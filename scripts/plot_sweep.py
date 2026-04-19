"""Aggregate plot: mean +/- std across seeds, grouped by (algo, env).

Reads each run's config.yaml to pick up (algo, env_id) tags, then draws one
subplot per env and one line per algo with a shaded std band.

Usage
-----
    python scripts/plot_sweep.py runs/*
    python scripts/plot_sweep.py runs/* --smooth 20 --out sweep.png
    python scripts/plot_sweep.py runs/* --key approx_kl
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run directories (each containing progress.csv).")
    p.add_argument("--key", default="mean_return_100")
    p.add_argument("--x", default="global_step")
    p.add_argument("--smooth", type=int, default=1)
    p.add_argument("--bins", type=int, default=200, help="Resampling bins on x.")
    p.add_argument("--out", default="sweep.png")
    args = p.parse_args()

    # env_id -> algo -> list[DataFrame]
    groups: dict[str, dict[str, list[pd.DataFrame]]] = defaultdict(lambda: defaultdict(list))
    for run_path in args.runs:
        run = Path(run_path)
        cfg_path = run / "config.yaml"
        csv_path = run / "progress.csv"
        if not cfg_path.exists() or not csv_path.exists():
            print(f"[warn] skipping {run} (missing config.yaml or progress.csv)")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        df = pd.read_csv(csv_path)
        if args.key not in df.columns or args.x not in df.columns:
            print(f"[warn] {run}: missing column '{args.key}' or '{args.x}'")
            continue
        if len(df) < 2:
            continue
        groups[cfg["env_id"]][cfg["algo"]].append(df)

    if not groups:
        print("No valid runs found.")
        return

    envs = sorted(groups.keys())
    n = len(envs)
    fig, axes = plt.subplots(1, n, figsize=(max(6, 6 * n), 4.5), squeeze=False)
    axes = axes[0]

    default_colors = {"ppo": "tab:blue", "trpo": "tab:orange"}

    for ax, env in zip(axes, envs):
        for algo, dfs in sorted(groups[env].items()):
            x_lo = max(df[args.x].min() for df in dfs)
            x_hi = min(df[args.x].max() for df in dfs)
            if x_hi <= x_lo:
                continue
            xs = np.linspace(x_lo, x_hi, args.bins)
            ys = np.stack([
                np.interp(xs, df[args.x].to_numpy(), df[args.key].to_numpy())
                for df in dfs
            ])
            ys = np.stack([_smooth(y, args.smooth) for y in ys])
            mean = ys.mean(axis=0)
            std = ys.std(axis=0)
            color = default_colors.get(algo)
            label = f"{algo} (n={len(dfs)})"
            ax.plot(xs, mean, label=label, color=color, linewidth=2.0)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)
        ax.set_title(env)
        ax.set_xlabel(args.x)
        ax.set_ylabel(args.key)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}  (envs: {envs})")


if __name__ == "__main__":
    main()
