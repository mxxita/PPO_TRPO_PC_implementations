"""On-policy rollout buffer with GAE advantage estimation.

`obs_dtype` is configurable (uint8 for Atari, float32 for MuJoCo). Everything
else stays on the training device so the algorithm update is host-transfer-free.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


_NP_TO_TORCH = {
    np.dtype("uint8"): torch.uint8,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float32,  # downcast: MPS has poor float64 support
}


def torch_obs_dtype(np_dtype) -> torch.dtype:
    return _NP_TO_TORCH.get(np.dtype(np_dtype), torch.float32)


class RolloutBuffer:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        obs_dtype: torch.dtype,
        action_dtype: torch.dtype,
        action_shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, dtype=obs_dtype, device=device)
        self.actions = torch.zeros(
            (num_steps, num_envs) + action_shape, dtype=action_dtype, device=device
        )
        self.logprobs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)

    def add(self, step, obs, action, logprob, reward, done, value):
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value

    @torch.no_grad()
    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor,
                    gamma: float, gae_lambda: float):
        lastgaelam = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
        self.returns[:] = self.advantages + self.values

    def get(self) -> Dict[str, torch.Tensor]:
        """Flatten (num_steps, num_envs, ...) -> (num_steps * num_envs, ...)."""
        return dict(
            obs=self.obs.reshape((-1,) + self.obs.shape[2:]),
            actions=self.actions.reshape((-1,) + self.actions.shape[2:]),
            logprobs=self.logprobs.reshape(-1),
            values=self.values.reshape(-1),
            advantages=self.advantages.reshape(-1),
            returns=self.returns.reshape(-1),
        )
