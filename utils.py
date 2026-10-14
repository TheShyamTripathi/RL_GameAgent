"""
utils.py
--------
Torch-free helpers: seeding, logging, policy evaluation, run registry
(metadata + CSV logs on disk) and display labels.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
from typing import Callable

import numpy as np

import configs


# ----------------------------------------------------------------------------
# reproducibility / logging
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str = "rl_gameagent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
def evaluate_policy(env, policy_fn: Callable, episodes: int = 5, seed: int = 10_000) -> list[float]:
    """Run `episodes` full episodes with fixed seeds; returns the list of returns."""
    scores = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done, total = False, 0.0
        while not done:
            state, reward, terminated, truncated, _ = env.step(policy_fn(state))
            done = terminated or truncated
            total += float(reward)
        scores.append(total)
    return scores


def summarize_scores(scores: list[float]) -> dict[str, float]:
    arr = np.asarray(scores, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max())}


# ----------------------------------------------------------------------------
# run registry (one JSON metadata file + one CSV log + models per run)
# ----------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)


def write_meta(run_name: str, meta: dict) -> None:
    """Atomic write so the dashboard never reads a half-written file."""
    path = configs.meta_path(run_name)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def list_runs(game: str | None = None) -> list[dict]:
    """All runs on disk (newest first), optionally filtered by game."""
    runs = []
    for path in glob.glob(os.path.join(configs.LOGS_DIR, "*.json")):
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if "run_name" not in meta:
            continue
        if game is None or meta.get("game") == game:
            runs.append(meta)
    runs.sort(key=lambda m: m.get("created", ""), reverse=True)
    return runs


def load_run_log(run_name: str):
    """Training log of a run as a DataFrame (None if missing/empty)."""
    import pandas as pd

    path = configs.log_csv_path(run_name)
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if len(df) else None


def run_label(meta: dict) -> str:
    variant = meta.get("variant", "?")
    short = configs.VARIANTS.get(variant, {}).get("short", variant)
    created = str(meta.get("created", ""))[5:16].replace("T", " ")
    return f"{short} · seed {meta.get('seed', '?')} · {created}"


def available_models(game: str) -> list[dict]:
    """Saved checkpoints for a game, newest run first (best before final)."""
    out = []
    for meta in list_runs(game):
        for kind in ("best", "final"):
            path = configs.model_file(meta["run_name"], kind)
            if os.path.exists(path):
                out.append({"label": f"{run_label(meta)} · {kind}", "path": path,
                            "run": meta["run_name"], "kind": kind})
    return out


def latest_model(game: str, kind: str = "best") -> str | None:
    for m in available_models(game):
        if m["kind"] == kind:
            return m["path"]
    return None
