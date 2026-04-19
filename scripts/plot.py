"""Plot one metric across one or more runs.

Usage:
    python scripts/plot.py runs/ppo_* runs/trpo_* --key mean_return_100 --out compare.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run directories containing progress.csv.")
    p.add_argument("--key", default="mean_return_100")
    p.add_argument("--x", default="global_step")
    p.add_argument("--smooth", type=int, default=1, help="Rolling-mean window (1 = none).")
    p.add_argument("--out", default="plot.png")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(8, 5))
    for run in args.runs:
        run = Path(run)
        csv_path = run / "progress.csv"
        if not csv_path.exists():
            print(f"[warn] skipping {run} (no progress.csv)")
            continue
        df = pd.read_csv(csv_path)
        if args.key not in df.columns:
            print(f"[warn] {run}: missing column '{args.key}'")
            continue
        y = df[args.key]
        if args.smooth > 1:
            y = y.rolling(args.smooth, min_periods=1).mean()
        ax.plot(df[args.x], y, label=run.name)

    ax.set_xlabel(args.x)
    ax.set_ylabel(args.key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
