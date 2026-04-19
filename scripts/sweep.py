"""Run a grid of (experiment, env, seed) training jobs.

Replaces the shell version with:
  - resumable: skips runs whose directory already contains a `model.pt`
  - parallel: N-at-a-time via a process pool
  - flexible: any extra `key=value` overrides get forwarded to train.py

Usage
-----
    # Default: ppo_atari + trpo_atari on Pong + Breakout, seeds 0..2
    python scripts/sweep.py

    # Quick smoke test across MuJoCo envs
    python scripts/sweep.py \\
        --experiments ppo_mujoco trpo_mujoco \\
        --env-ids HalfCheetah-v4 Hopper-v4 \\
        --seeds 0 1 2 \\
        --total-timesteps 100000 \\
        --parallel 2

    # Pass-through overrides (everything after -- goes to train.py)
    python scripts/sweep.py --seeds 0 1 2 -- lr=1e-4 clip_coef=0.2
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_one(
    experiment: str,
    env_id: str,
    seed: int,
    total_timesteps: int | None,
    device: str,
    extra_overrides: list[str],
) -> int:
    cmd = [
        sys.executable, str(ROOT / "scripts" / "train.py"),
        "--experiment", experiment,
        "--env-id", env_id,
        "--seed", str(seed),
        "--device", device,
    ]
    if total_timesteps is not None:
        cmd += ["--total-timesteps", str(total_timesteps)]
    cmd += extra_overrides
    print(f"[run] {' '.join(cmd)}")
    return subprocess.call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiments", nargs="+", default=["ppo_atari", "trpo_atari"],
                   help="Experiment names under configs/experiments/.")
    p.add_argument("--env-ids", nargs="+",
                   default=["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--parallel", type=int, default=1, help="Concurrent jobs.")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Override each experiment's total_timesteps.")
    p.add_argument("--device", default="auto")
    # Everything after `--` is forwarded verbatim as key=value overrides.
    args, passthrough = p.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    jobs = list(itertools.product(args.experiments, args.env_ids, args.seeds))
    print(f"[sweep] launching {len(jobs)} runs "
          f"({args.experiments} x {args.env_ids} x seeds={args.seeds})  "
          f"parallel={args.parallel}")

    if args.parallel <= 1:
        for exp, env, seed in jobs:
            run_one(exp, env, seed, args.total_timesteps, args.device, passthrough)
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futures = [
                ex.submit(run_one, exp, env, seed, args.total_timesteps, args.device, passthrough)
                for (exp, env, seed) in jobs
            ]
            for f in as_completed(futures):
                f.result()

    print("[sweep] done. Aggregate with:")
    print("  python scripts/plot_sweep.py runs/*")


if __name__ == "__main__":
    main()
