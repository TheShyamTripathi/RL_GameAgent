"""
configs.py
----------
Single source of truth for paths, game specifications, algorithm variants and
the typed `TrainConfig` used by every other module.

Adding a new discrete-action Gymnasium game = one new `GameSpec` in `GAMES`.
Adding a new algorithm variant              = one new entry in `VARIANTS`.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from functools import lru_cache
from typing import Any

# ----------------------------------------------------------------------------
# Paths (models/ and logs/ are auto-created)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def get_device():
    """Lazy import so that non-torch modules (UI, utils, tests) stay light."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Algorithm variants  (used for ablation studies)
# ----------------------------------------------------------------------------
VARIANTS: dict[str, dict[str, Any]] = {
    "dqn": {
        "label": "Vanilla DQN",
        "short": "DQN",
        "description": "Baseline DQN (Mnih et al., 2015): uniform replay, 1-step targets, plain MLP.",
        "flags": dict(double_dqn=False, dueling=False, prioritized=False, n_step=1),
    },
    "ddqn": {
        "label": "Double DQN",
        "short": "DDQN",
        "description": "Adds Double Q-learning: the online net picks the action, the target net evaluates it. "
                       "Reduces over-estimation of Q-values.",
        "flags": dict(double_dqn=True, dueling=False, prioritized=False, n_step=1),
    },
    "dueling_ddqn": {
        "label": "Dueling Double DQN",
        "short": "Dueling DDQN",
        "description": "Double DQN + dueling head that separates state value V(s) from advantages A(s,a).",
        "flags": dict(double_dqn=True, dueling=True, prioritized=False, n_step=1),
    },
    "per_dueling_ddqn": {
        "label": "Dueling DDQN + Prioritized Replay",
        "short": "PER + Dueling DDQN",
        "description": "Adds Prioritized Experience Replay (sum-tree, importance-sampling weights): "
                       "learns more from surprising transitions.",
        "flags": dict(double_dqn=True, dueling=True, prioritized=True, n_step=1),
    },
    "full": {
        "label": "Full (Double + Dueling + PER + 3-step)",
        "short": "Full",
        "description": "Everything above plus 3-step returns for faster reward propagation "
                       "(a 'Rainbow-lite' agent).",
        "flags": dict(double_dqn=True, dueling=True, prioritized=True, n_step=3),
    },
}


# ----------------------------------------------------------------------------
# Game specifications
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class GameSpec:
    env_id: str
    emoji: str
    goal: str
    state: str
    actions: str
    reward: str
    solved_text: str
    solved_score: float      # mean reward over the last 100 episodes
    episodes: int            # default training budget
    lr: float
    eps_decay: float


GAMES: dict[str, GameSpec] = {
    "CartPole-v1": GameSpec(
        env_id="CartPole-v1",
        emoji="🛒",
        goal="Balance a pole on a moving cart for as long as possible.",
        state="4 values: cart position, cart velocity, pole angle, pole angular velocity",
        actions="2 actions: push cart left / push cart right",
        reward="+1 for every step the pole stays up (max 500 per episode)",
        solved_text="Average reward ≥ 475 over 100 episodes",
        solved_score=475.0, episodes=300, lr=1e-3, eps_decay=0.99,
    ),
    "Acrobot-v1": GameSpec(
        env_id="Acrobot-v1",
        emoji="🦾",
        goal="Swing a two-link robot arm so that its tip rises above the target line.",
        state="6 values: sin/cos of both joint angles + two angular velocities",
        actions="3 actions: apply torque -1 / 0 / +1 on the middle joint",
        reward="-1 for every step until the goal is reached (max 500 steps)",
        solved_text="Average reward ≥ -100 over 100 episodes",
        solved_score=-100.0, episodes=200, lr=5e-4, eps_decay=0.98,
    ),
}


# ----------------------------------------------------------------------------
# Typed training configuration
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- experiment identity
    game: str = "CartPole-v1"
    env_id: str = "CartPole-v1"
    variant: str = "full"
    run_name: str = ""
    seed: int = 42
    # --- budget / stopping
    episodes: int = 300
    solved_score: float = 475.0
    early_stop: bool = True
    # --- network
    hidden: tuple = (128, 128)
    dueling: bool = True
    # --- optimisation
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    grad_clip: float = 10.0
    tau: float = 0.005            # soft target update (Polyak averaging)
    double_dqn: bool = True
    # --- exploration (epsilon decays per episode)
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.99
    # --- replay
    buffer_size: int = 50_000
    warmup_steps: int = 1_000
    n_step: int = 3
    prioritized: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4   # annealed linearly to 1.0 over training
    # --- evaluation
    eval_every: int = 10          # greedy evaluation every N episodes
    eval_episodes: int = 5
    final_eval_episodes: int = 20

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        if "hidden" in clean:
            clean["hidden"] = tuple(clean["hidden"])
        return cls(**clean)


def new_run_name(game: str, variant: str, seed: int) -> str:
    return f"{game}__{variant}__s{seed}__{datetime.now():%Y%m%d-%H%M%S}"


def make_config(game: str, variant: str = "full", **overrides: Any) -> TrainConfig:
    """Build a config:  defaults  <-  game spec  <-  variant flags  <-  overrides."""
    if game not in GAMES:
        raise ValueError(f"Unknown game '{game}'. Choose from {list(GAMES)}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(VARIANTS)}")

    spec = GAMES[game]
    cfg = TrainConfig(
        game=game, env_id=spec.env_id, variant=variant,
        solved_score=spec.solved_score, episodes=spec.episodes,
        lr=spec.lr, eps_decay=spec.eps_decay,
    )
    for k, v in VARIANTS[variant]["flags"].items():
        setattr(cfg, k, v)

    valid = {f.name for f in fields(TrainConfig)}
    for k, v in overrides.items():
        if k not in valid:
            raise ValueError(f"Unknown config option '{k}'")
        if v is not None:
            setattr(cfg, k, v)
    cfg.hidden = tuple(cfg.hidden)
    if not cfg.run_name:
        cfg.run_name = new_run_name(cfg.game, cfg.variant, cfg.seed)
    return cfg


# ----------------------------------------------------------------------------
# File naming helpers (functions, so tests can monkeypatch the directories)
# ----------------------------------------------------------------------------
def log_csv_path(run_name: str) -> str:
    return os.path.join(LOGS_DIR, f"{run_name}.csv")


def meta_path(run_name: str) -> str:
    return os.path.join(LOGS_DIR, f"{run_name}.json")


def model_file(run_name: str, kind: str = "best") -> str:
    return os.path.join(MODELS_DIR, f"{run_name}_{kind}.pt")
