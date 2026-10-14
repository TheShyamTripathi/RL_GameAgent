"""
play.py
-------
Watch / evaluate trained agents.

    python play.py --game CartPole-v1 --episodes 3            # latest best model, pygame window
    python play.py --game Acrobot-v1 --model models/<file>.pt
    python play.py --game CartPole-v1 --random                # random baseline
    python play.py --game CartPole-v1 --evaluate 20           # no window, prints statistics
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym

from agent import DQNAgent
from configs import GAMES
from utils import evaluate_policy, latest_model, summarize_scores


def _make_env(game: str, render_mode: str | None):
    return gym.make(GAMES[game].env_id, render_mode=render_mode)


def _policy(agent: DQNAgent | None, env):
    return (lambda s: env.action_space.sample()) if agent is None else (lambda s: agent.act(s, greedy=True))


def episode_stream(game: str, model_file: str | None = None, seed: int | None = None):
    """Generator for the dashboard: yields the RGB frame, step, score and done flag after every step."""
    env = _make_env(game, "rgb_array")
    policy = _policy(DQNAgent.load(model_file) if model_file else None, env)
    state, _ = env.reset(seed=seed)
    score, step, done = 0.0, 0, False
    yield {"frame": env.render(), "step": 0, "score": 0.0, "done": False}
    while not done:
        state, reward, terminated, truncated, _ = env.step(policy(state))
        done = terminated or truncated
        score += float(reward)
        step += 1
        yield {"frame": env.render(), "step": step, "score": score, "done": done}
    env.close()


def evaluate(game: str, model_file: str | None = None, episodes: int = 10, seed: int = 1000) -> list[float]:
    env = _make_env(game, None)
    agent = DQNAgent.load(model_file) if model_file else None
    scores = evaluate_policy(env, _policy(agent, env), episodes, seed)
    env.close()
    return scores


def play_window(game: str, model_file: str | None, episodes: int) -> None:
    env = _make_env(game, "human")
    policy = _policy(DQNAgent.load(model_file) if model_file else None, env)
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        done, total = False, 0.0
        while not done:
            state, r, term, trunc, _ = env.step(policy(state))
            done, total = term or trunc, total + float(r)
        print(f"Episode {ep}: score = {total:.1f}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Play / evaluate a trained agent")
    p.add_argument("--game", default="CartPole-v1", choices=list(GAMES))
    p.add_argument("--model", help="path to a .pt checkpoint (default: newest best model of the game)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--random", action="store_true", help="random policy baseline")
    p.add_argument("--evaluate", type=int, default=0, help="evaluate N episodes without rendering")
    a = p.parse_args()

    model_file = None if a.random else (a.model or latest_model(a.game, "best"))
    if not a.random and (not model_file or not os.path.exists(model_file)):
        raise SystemExit(f"No trained model found. Run:  python train.py --game {a.game}")

    if a.evaluate:
        s = summarize_scores(evaluate(a.game, model_file, a.evaluate))
        print(f"mean {s['mean']:.1f} ± {s['std']:.1f}  (min {s['min']:.1f}, max {s['max']:.1f})")
    else:
        play_window(a.game, model_file, a.episodes)
