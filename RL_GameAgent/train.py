"""
train.py
--------
Training pipeline.

* `Trainer`      - runs one experiment: training loop, periodic greedy evaluation,
                   best-checkpoint selection, CSV + JSON run tracking, final evaluation.
* `TrainingJob`  - runs a Trainer in a background thread (used by the dashboard so the
                   UI stays responsive and training can be stopped).

CLI:
    python train.py --game CartPole-v1 --variant full --episodes 300
    python train.py --game Acrobot-v1 --variant dqn --seed 1
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from collections import deque
from datetime import datetime

import gymnasium as gym
import numpy as np

import configs
from agent import DQNAgent
from configs import GAMES, VARIANTS, TrainConfig, make_config
from utils import evaluate_policy, get_logger, set_seed, summarize_scores, write_meta

LOG_FIELDS = ["episode", "reward", "avg20", "avg100", "epsilon", "beta", "loss", "mean_q",
              "steps", "total_steps", "eval_mean", "time"]
NAN = float("nan")


class Trainer:
    def __init__(self, cfg: TrainConfig, progress_cb=None, stop_event: threading.Event | None = None):
        self.cfg = cfg
        self.progress_cb = progress_cb
        self.stop_event = stop_event
        self.log = get_logger()

    def run(self) -> dict:
        cfg, log = self.cfg, self.log
        set_seed(cfg.seed)
        env, eval_env = gym.make(cfg.env_id), gym.make(cfg.env_id)
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg)

        csv_file = configs.log_csv_path(cfg.run_name)
        best_path = configs.model_file(cfg.run_name, "best")
        final_path = configs.model_file(cfg.run_name, "final")
        meta = {"run_name": cfg.run_name, "game": cfg.game, "variant": cfg.variant, "seed": cfg.seed,
                "created": datetime.now().isoformat(timespec="seconds"), "status": "running",
                "config": cfg.to_dict(), "summary": {}}
        write_meta(cfg.run_name, meta)
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

        last20, last100 = deque(maxlen=20), deque(maxlen=100)
        best_eval, first_solved = -float("inf"), None
        total_steps, episodes_run, avg100 = 0, 0, NAN
        status, t0 = "finished", time.time()
        policy = lambda s: agent.act(s, greedy=True)

        log.info("▶ %s | %s | seed %d | %d episodes", cfg.game, VARIANTS[cfg.variant]["label"],
                 cfg.seed, cfg.episodes)
        try:
            for ep in range(1, cfg.episodes + 1):
                if self.stop_event is not None and self.stop_event.is_set():
                    status = "stopped"
                    break

                state, _ = env.reset(seed=cfg.seed + ep)
                total, steps, done = 0.0, 0, False
                losses, qs = [], []
                while not done:
                    action = agent.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    # `terminated` (not `done`) controls bootstrapping: time-limit truncation is not a real end
                    agent.observe(state, action, reward, next_state, terminated, done)
                    stats = agent.learn()
                    if stats:
                        losses.append(stats["loss"])
                        qs.append(stats["mean_q"])
                    state, total, steps = next_state, total + reward, steps + 1

                agent.end_episode(progress=ep / cfg.episodes)
                total_steps += steps
                episodes_run = ep
                last20.append(total)
                last100.append(total)
                avg20, avg100 = float(np.mean(last20)), float(np.mean(last100))

                eval_mean = NAN
                if ep % cfg.eval_every == 0:
                    eval_mean = float(np.mean(evaluate_policy(eval_env, policy, cfg.eval_episodes, seed=10_000)))
                    if eval_mean >= best_eval:                      # best checkpoint = best greedy score
                        best_eval = eval_mean
                        agent.save(best_path, {"episode": ep, "eval_mean": eval_mean})

                if first_solved is None and len(last100) == 100 and avg100 >= cfg.solved_score:
                    first_solved = ep

                rec = {
                    "episode": ep, "reward": total, "avg20": avg20, "avg100": avg100,
                    "epsilon": agent.epsilon, "beta": agent.beta,
                    "loss": float(np.mean(losses)) if losses else NAN,
                    "mean_q": float(np.mean(qs)) if qs else NAN,
                    "steps": steps, "total_steps": total_steps, "eval_mean": eval_mean,
                    "time": time.time() - t0,
                }
                with open(csv_file, "a", newline="") as f:
                    csv.writer(f).writerow([rec[k] for k in LOG_FIELDS])
                if self.progress_cb:
                    self.progress_cb(dict(rec, episodes=cfg.episodes,
                                          best_eval=best_eval if np.isfinite(best_eval) else NAN))
                if ep % 10 == 0 or ep == 1:
                    log.info("ep %4d/%d | reward %7.1f | avg20 %7.1f | avg100 %7.1f | eps %.3f | loss %.4f",
                             ep, cfg.episodes, total, avg20, avg100, agent.epsilon, rec["loss"])

                if cfg.early_stop and first_solved is not None:
                    log.info("✅ solved at episode %d (avg100 = %.1f) - early stop", ep, avg100)
                    break

            # ---- wrap-up: save, then rigorous final evaluation of the BEST checkpoint
            agent.save(final_path, {"episode": episodes_run})
            if not np.isfinite(best_eval):
                agent.save(best_path, {"episode": episodes_run})
            best_agent = DQNAgent.load(best_path)
            final = summarize_scores(evaluate_policy(
                eval_env, lambda s: best_agent.act(s, greedy=True), cfg.final_eval_episodes, seed=20_000))

            summary = {
                "episodes_run": episodes_run, "total_steps": total_steps,
                "solved": first_solved is not None, "first_solved_episode": first_solved,
                "best_eval": best_eval if np.isfinite(best_eval) else None,
                "final_avg100": avg100 if np.isfinite(avg100) else None,
                "final_eval_mean": final["mean"], "final_eval_std": final["std"],
                "final_eval_min": final["min"], "final_eval_max": final["max"],
                "seconds": time.time() - t0,
            }
            meta.update(status=status, summary=summary)
            write_meta(cfg.run_name, meta)
            log.info("■ %s | final eval %.1f ± %.1f over %d episodes | %.0fs", status,
                     final["mean"], final["std"], cfg.final_eval_episodes, summary["seconds"])
            return dict(summary, run_name=cfg.run_name, status=status,
                        best_model=best_path, final_model=final_path)
        except Exception as exc:
            meta.update(status="error", error=repr(exc))
            write_meta(cfg.run_name, meta)
            raise
        finally:
            env.close()
            eval_env.close()


class TrainingJob:
    """Runs a `Trainer` in a daemon thread; safe to poll from the Streamlit script."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.history: list[dict] = []
        self.result: dict | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"train-{cfg.run_name}")

    def start(self) -> "TrainingJob":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.result = Trainer(self.cfg, progress_cb=self.history.append, stop_event=self._stop).run()
        except Exception as exc:  # surfaced in the UI
            self.error = repr(exc)

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    @property
    def stopping(self) -> bool:
        return self.running and self._stop.is_set()


def main() -> None:
    p = argparse.ArgumentParser(description="Train a DQN-family agent")
    p.add_argument("--game", default="CartPole-v1", choices=list(GAMES))
    p.add_argument("--variant", default="full", choices=list(VARIANTS))
    p.add_argument("--episodes", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float)
    p.add_argument("--gamma", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--n_step", type=int, help="override the variant's n-step value")
    p.add_argument("--no_early_stop", action="store_true")
    a = p.parse_args()

    cfg = make_config(a.game, a.variant, episodes=a.episodes, seed=a.seed, lr=a.lr, gamma=a.gamma,
                      batch_size=a.batch_size, n_step=a.n_step,
                      early_stop=False if a.no_early_stop else None)
    result = Trainer(cfg).run()
    print("\nResult:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in result.items()})


if __name__ == "__main__":
    main()
