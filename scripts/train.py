"""Single entry point for training PPO or TRPO on Atari or MuJoCo.

Usage
-----
    # By experiment name (configs/experiments/<name>.yaml)
    python scripts/train.py --experiment ppo_atari --env-id PongNoFrameskip-v4 --seed 0

    # By explicit config path, with CLI overrides (YAML-parsed)
    python scripts/train.py --config configs/experiments/trpo_mujoco.yaml \\
        --seed 1  total_timesteps=500000  lr=1e-4

The training loop is identical for PPO/TRPO and for Atari/MuJoCo; the
per-algorithm / per-domain bits live in env / network / agent factories.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# Make `pcrl` importable when run as `python scripts/train.py` without install.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcrl.agents import PPO, TRPO  # noqa: E402
from pcrl.buffer import RolloutBuffer, torch_obs_dtype  # noqa: E402
from pcrl.config import apply_overrides, load_config  # noqa: E402
from pcrl.device import get_device  # noqa: E402
from pcrl.envs import make_env  # noqa: E402
from pcrl.learners import BackpropLearner  # noqa: E402
from pcrl.logger import CSVLogger  # noqa: E402
from pcrl.networks import make_actor_critic  # noqa: E402
from pcrl.run_utils import default_run_name, write_run_metadata  # noqa: E402
from pcrl.seeding import set_seed  # noqa: E402


CONFIG_ROOT = ROOT / "configs"


def resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config).resolve()
    if args.experiment:
        return (CONFIG_ROOT / "experiments" / f"{args.experiment}.yaml").resolve()
    raise SystemExit("Provide --experiment NAME or --config PATH.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=str, default=None,
                   help="Name under configs/experiments/ (e.g. ppo_atari).")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config file (overrides --experiment).")
    p.add_argument("--env-id", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--total-timesteps", type=int, default=None)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-interval", type=int, default=10,
                   help="Print progress every N updates.")
    p.add_argument("overrides", nargs="*",
                   help="Extra key=value overrides (YAML-parsed).")
    return p.parse_args()


def _to_tensor(x, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    arr = np.asarray(x)
    if arr.dtype == np.float64 and dtype != torch.float64:
        arr = arr.astype(np.float32)
    return torch.as_tensor(arr, device=device).to(dtype)


def build_learner(params, lr: float, learner_cfg: dict) -> BackpropLearner:
    """Instantiate a Learner from cfg['learner']. Default: backprop."""
    t = (learner_cfg or {}).get("type", "backprop")
    if t == "backprop":
        return BackpropLearner(
            params, lr=lr, eps=learner_cfg.get("eps", 1e-5) if learner_cfg else 1e-5,
        )
    # Hook point for future learners (predictive_coding, ...).
    raise ValueError(f"Unknown learner type: {t!r}")


def build_agent(cfg: dict, network: torch.nn.Module, device: torch.device):
    algo = cfg["algo"]
    learner_cfg = cfg.get("learner", {"type": "backprop"})
    if algo == "ppo":
        learner = build_learner(network.parameters(), cfg["lr"], learner_cfg)
        return PPO(network, cfg, learner, device)
    if algo == "trpo":
        # Only the value regression uses a Learner; policy step is bespoke.
        value_learner = build_learner(network.value_parameters(), cfg["vf_lr"], learner_cfg)
        return TRPO(network, cfg, value_learner, device)
    raise ValueError(f"Unknown algo: {algo!r}")


def main() -> None:
    args = parse_args()
    cfg_path = resolve_config_path(args)
    cfg = load_config(cfg_path)

    # CLI overrides (typed args first, then generic key=value).
    if args.env_id:
        cfg["env_id"] = args.env_id
    if args.total_timesteps:
        cfg["total_timesteps"] = args.total_timesteps
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    run_name = args.run_name or default_run_name(cfg["algo"], cfg["env_id"], args.seed)
    run_dir = ROOT / "runs" / run_name
    write_run_metadata(
        run_dir,
        {**cfg, "seed": args.seed},
        extra={"experiment_config": str(cfg_path.relative_to(ROOT))
               if cfg_path.is_relative_to(ROOT) else str(cfg_path)},
    )

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"[config] {cfg_path.name}   algo={cfg['algo']}   env={cfg['env_id']}   seed={args.seed}")
    print(f"[device] {device}")
    print(f"[run]    {run_dir}")

    # -------- env, network, agent, buffer ------------------------------------
    envs = make_env(cfg["env_type"], cfg["env_id"], args.seed, cfg["num_envs"])
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space
    obs_dtype = torch_obs_dtype(obs_space.dtype)

    if act_space.__class__.__name__ == "Discrete":
        action_dtype, action_shape = torch.long, ()
    else:
        action_dtype, action_shape = torch.float32, tuple(act_space.shape)

    network = make_actor_critic(envs, share_backbone=cfg["share_backbone"]).to(device)
    agent = build_agent(cfg, network, device)
    buffer = RolloutBuffer(
        num_steps=cfg["num_steps"],
        num_envs=cfg["num_envs"],
        obs_shape=tuple(obs_space.shape),
        obs_dtype=obs_dtype,
        action_dtype=action_dtype,
        action_shape=action_shape,
        device=device,
    )
    logger = CSVLogger(run_dir / "progress.csv")

    # -------- training loop --------------------------------------------------
    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = _to_tensor(next_obs_np, obs_dtype, device)
    next_done = torch.zeros(cfg["num_envs"], device=device)

    global_step = 0
    batch_per_update = cfg["num_envs"] * cfg["num_steps"]
    total_updates = max(1, cfg["total_timesteps"] // batch_per_update)
    ep_returns: deque[float] = deque(maxlen=100)
    ep_lengths: deque[int] = deque(maxlen=100)
    total_episodes = 0
    start = time.time()

    for update in range(1, total_updates + 1):
        # ---- collect rollout ------------------------------------------------
        for step in range(cfg["num_steps"]):
            global_step += cfg["num_envs"]
            with torch.no_grad():
                action, logprob, _, value = network.get_action_and_value(next_obs)

            obs_np, reward, terms, truncs, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terms, truncs)

            buffer.add(
                step=step,
                obs=next_obs,
                action=action,
                logprob=logprob,
                reward=_to_tensor(reward, torch.float32, device),
                done=next_done,
                value=value,
            )
            next_obs = _to_tensor(obs_np, obs_dtype, device)
            next_done = _to_tensor(done.astype(np.float32), torch.float32, device)

            if "episode" in infos:
                mask = infos.get("_episode", np.zeros(cfg["num_envs"], dtype=bool))
                for i in range(cfg["num_envs"]):
                    if mask[i]:
                        ep_returns.append(float(infos["episode"]["r"][i]))
                        ep_lengths.append(int(infos["episode"]["l"][i]))
                        total_episodes += 1

        # ---- bootstrap + GAE ------------------------------------------------
        with torch.no_grad():
            next_value = network.get_value(next_obs)
        buffer.compute_gae(next_value, next_done, cfg["gamma"], cfg["gae_lambda"])

        progress_remaining = 1.0 - (update - 1) / total_updates
        stats = agent.update(buffer, progress_remaining=progress_remaining)

        # ---- log ------------------------------------------------------------
        elapsed = time.time() - start
        fps = int(global_step / max(elapsed, 1e-6))
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        row = {
            "global_step": global_step,
            "update": update,
            "elapsed": round(elapsed, 2),
            "fps": fps,
            "mean_return_100": mean_ret,
            "mean_length_100": mean_len,
            "total_episodes": total_episodes,
            **stats,
        }
        logger.log(row)
        if update == 1 or update % args.log_interval == 0 or update == total_updates:
            if cfg["algo"] == "ppo":
                extra = f" kl={stats['approx_kl']:.4f} clipfrac={stats['clipfrac']:.3f}"
            else:
                extra = (f" kl={stats['kl']:.4f} ls={stats['line_search_success']} "
                         f"step={stats['step_size']:.3f}")
            print(
                f"[{update:>5}/{total_updates}] step={global_step:>9}  fps={fps:>5}  "
                f"ret100={mean_ret:>8.2f}  eps={total_episodes:>5}{extra}"
            )

    torch.save(network.state_dict(), run_dir / "model.pt")
    envs.close()
    logger.close()
    print(f"[done] model saved to {run_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
