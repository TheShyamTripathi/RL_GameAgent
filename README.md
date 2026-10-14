# 🎮 RL GameAgent — a modular Deep Q-Network family on classic-control games

Train, watch and analyse reinforcement-learning agents on **CartPole-v1** and **Acrobot-v1**
with a clean, testable code base and a Streamlit dashboard.

The project is built around an **ablation-friendly DQN stack**: every extension from the
DQN literature can be switched on/off through a preset, so you can measure what each
idea actually contributes.

| Extension | Paper | Where |
|---|---|---|
| DQN (replay + target network) | Mnih et al., 2015 | `agent.py` |
| Double Q-learning | van Hasselt et al., 2016 | `agent.py` |
| Dueling architecture | Wang et al., 2016 | `networks.py` |
| Prioritized Experience Replay | Schaul et al., 2016 | `replay_buffer.py` |
| N-step returns | Sutton, 1988 | `replay_buffer.py` |

## Project structure

```
RL_GameAgent/
├── requirements.txt
├── configs.py          # paths, GameSpec, algorithm VARIANTS, typed TrainConfig
├── networks.py         # Q-network (MLP / dueling head)
├── replay_buffer.py    # n-step accumulator, uniform buffer, sum-tree, prioritized buffer
├── agent.py            # DQNAgent: acting, learning step, save/load
├── train.py            # Trainer (experiment runner) + TrainingJob (background thread) + CLI
├── play.py             # watch / evaluate agents (window, dashboard stream, CLI)
├── dashboard.py        # Streamlit UI
├── compare.py          # ablation runner: variants x seeds -> aggregate table
├── utils.py            # seeding, logging, evaluation, run registry
├── tests/              # pytest suite (replay logic, config/registry, end-to-end smoke)
├── models/             # (auto-created) checkpoints:  <run>_best.pt, <run>_final.pt
└── logs/               # (auto-created) per-run metrics CSV + metadata JSON
```

## Quick start

```bash
pip install -r requirements.txt        # Python 3.10+
streamlit run dashboard.py             # UI

# or from the command line
python train.py --game CartPole-v1 --variant full
python train.py --game Acrobot-v1  --variant dqn --seed 1
python play.py  --game CartPole-v1 --episodes 3
python play.py  --game CartPole-v1 --evaluate 20
python compare.py --game CartPole-v1 --variants dqn ddqn full --seeds 3   # ablation study
pytest                                                                      # tests
```

## Dashboard

* **Overview** – architecture diagram, component table, ablation presets.
* **Train** – choose a variant, tune hyper-parameters, train in a *background thread*
  with live charts and a **Stop** button (UI stays responsive, best model is saved as it improves).
* **Play** – watch any saved checkpoint (or a random agent) play, frame by frame.
* **Analytics** – leaderboard, aggregated ablation table, multi-run comparison curves,
  per-run diagnostics (loss, Q-values, ε, PER β), and trained-vs-random evaluation.

## Design decisions

* **Typed configuration** – one `TrainConfig` dataclass is saved with every run and every checkpoint,
  so results are reproducible and checkpoints are self-describing.
* **Run tracking** – each run writes `logs/<run>.csv` (per-episode metrics), `logs/<run>.json`
  (config, status, summary) and `models/<run>_{best,final}.pt`. The dashboard simply reads this registry,
  so CLI runs and UI runs are interchangeable.
* **Honest model selection** – the best checkpoint is chosen by *fixed-seed greedy evaluation*, not by the
  noisy ε-greedy training reward. The reported score comes from a separate final evaluation of that
  checkpoint on different seeds.
* **Correct bootstrapping** – only true termination stops bootstrapping; time-limit truncation does not.
  N-step transitions store their own discount γᵏ, including shortened windows at episode end.
* **Torch-free core logic** – `replay_buffer.py`, `configs.py` and `utils.py` need only numpy, which keeps the
  tests fast and the UI start-up light (torch is imported lazily).

## Method summary

For a batch of (possibly n-step) transitions with weights wᵢ:

```
a*      = argmax_a Q_online(s', a)                      (Double DQN)
y       = R_n + γⁿ · Q_target(s', a*) · (1 − terminated)
δ       = y − Q_online(s, a)
L       = mean( wᵢ · Huber(δᵢ) )
pᵢ      = (|δᵢ| + ε)^α      →  P(i) = pᵢ / Σ pₖ ;  wᵢ = (N·P(i))^(−β) / maxⱼ wⱼ     (PER)
θ_target ← (1 − τ) θ_target + τ θ_online                (soft update)
```

β is annealed linearly from 0.4 to 1.0 over training; ε decays geometrically per episode.

## Reproducing the experiments

```bash
python compare.py --game CartPole-v1 --variants dqn ddqn dueling_ddqn per_dueling_ddqn full --seeds 3
python compare.py --game Acrobot-v1  --variants dqn full --seeds 3
```

Results are written to `logs/compare_<game>_<timestamp>.csv` and appear in the dashboard's
*Analytics → Ablation summary*. Use several seeds: single DQN runs vary a lot.

## Extending

* **New game** – add a `GameSpec` to `GAMES` in `configs.py` (any discrete-action Gymnasium env with a
  vector observation).
* **New variant** – add an entry to `VARIANTS` (a set of flags), then implement the flag in `agent.py`.

## Limitations / future work

* Vector observations and discrete actions only (no CNN / continuous control).
* Possible next steps: noisy nets, distributional (C51) head, and multi-environment training.
