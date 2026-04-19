"""Roll out a trained agent for N episodes and print returns.

Usage:
    python scripts/evaluate.py runs/ppo_PongNoFrameskip-v4_s0_.../ --episodes 10
    python scripts/evaluate.py runs/trpo_Hopper-v4_s0_... --render --deterministic
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcrl.buffer import torch_obs_dtype  # noqa: E402
from pcrl.device import get_device  # noqa: E402
from pcrl.envs import make_env  # noqa: E402
from pcrl.networks import make_actor_critic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--render", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="Use mean/argmax action.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device)
    envs = make_env(
        cfg["env_type"], cfg["env_id"],
        seed=args.seed, num_envs=1,
        render_mode=("human" if args.render else None),
    )
    obs_dtype = torch_obs_dtype(envs.single_observation_space.dtype)

    network = make_actor_critic(envs, share_backbone=cfg["share_backbone"]).to(device)
    network.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    network.eval()

    returns = []
    for ep in range(args.episodes):
        obs, _ = envs.reset(seed=args.seed + ep)
        ep_ret = 0.0
        done = False
        while not done:
            obs_t = torch.as_tensor(np.asarray(obs), device=device).to(obs_dtype)
            with torch.no_grad():
                if args.deterministic:
                    action = network.deterministic_action(obs_t)
                else:
                    action, _, _, _ = network.get_action_and_value(obs_t)
            obs, reward, terms, truncs, _ = envs.step(action.cpu().numpy())
            ep_ret += float(reward[0])
            done = bool(terms[0] or truncs[0])
        returns.append(ep_ret)
        print(f"  ep {ep + 1:2d}: return = {ep_ret:.2f}")
    print(f"\nMean return over {args.episodes} episodes: "
          f"{np.mean(returns):.2f} +- {np.std(returns):.2f}")
    envs.close()


if __name__ == "__main__":
    main()
