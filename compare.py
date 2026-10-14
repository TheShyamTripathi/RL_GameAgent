"""
compare.py
----------
Ablation study: trains several variants on several seeds and prints an aggregate table.

    python compare.py --game CartPole-v1 --variants dqn ddqn full --seeds 3
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd

import configs
from configs import GAMES, VARIANTS, make_config
from train import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation: variants x seeds")
    p.add_argument("--game", default="CartPole-v1", choices=list(GAMES))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--seeds", type=int, default=3, help="number of seeds per variant")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--episodes", type=int)
    a = p.parse_args()

    rows = []
    for variant in a.variants:
        for k in range(a.seeds):
            cfg = make_config(a.game, variant, seed=a.base_seed + k, episodes=a.episodes)
            res = Trainer(cfg).run()
            rows.append({"variant": variant, "seed": cfg.seed, "solved_at": res["first_solved_episode"],
                         "final_eval": res["final_eval_mean"], "seconds": res["seconds"]})

    df = pd.DataFrame(rows)
    agg = df.groupby("variant", sort=False).agg(
        runs=("seed", "count"),
        solved_rate=("solved_at", lambda x: x.notna().mean()),
        episodes_to_solve=("solved_at", "mean"),
        final_eval_mean=("final_eval", "mean"),
        final_eval_std=("final_eval", "std"),
        seconds=("seconds", "mean"),
    ).round(2)
    out = os.path.join(configs.LOGS_DIR, f"compare_{a.game}_{datetime.now():%Y%m%d-%H%M%S}.csv")
    agg.to_csv(out)
    print("\n" + agg.to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
