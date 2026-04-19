"""Agent ABC — the RL algorithm interface.

An Agent owns the policy/value network and the algorithm-specific update rule
(PPO clipped surrogate, TRPO natural gradient, ...). It delegates the actual
parameter updates to one or more `Learner`s so that the same algorithm can
run under backprop or predictive coding without code changes.

If you implement a new algorithm, inherit from `Agent` and implement
`.update(buffer, progress_remaining)`. `.act()` works out of the box as long
as the network exposes `get_action_and_value`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class Agent(ABC):
    """Minimal interface every RL algorithm in this repo must satisfy."""

    network: torch.nn.Module
    cfg: Dict[str, Any]
    device: torch.device

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        """Sample an action; return (action, logprob, value)."""
        action, logprob, _, value = self.network.get_action_and_value(obs)
        return action, logprob, value

    @abstractmethod
    def update(self, buffer, progress_remaining: float = 1.0) -> Dict[str, float]:
        """Consume a full rollout buffer, update parameters, return scalar stats.

        `progress_remaining` is 1.0 at the start of training and 0.0 at the end;
        algorithms that anneal LR / clip coefficient use it as their schedule.
        """
