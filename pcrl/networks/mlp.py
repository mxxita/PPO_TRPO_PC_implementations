"""MuJoCo / continuous control actor-critic: MLP + diagonal Gaussian.

Matches the architecture used for MuJoCo in both papers:
  - two hidden layers of 64 units, tanh
  - separate policy / value towers
  - state-independent log-std, initialised to 0
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Independent, Normal

from .nature_cnn import layer_init


def _mlp_trunk(sizes, activation=nn.Tanh) -> nn.Sequential:
    """Feature extractor: activation after every layer (including last)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(layer_init(nn.Linear(sizes[i], sizes[i + 1])))
        layers.append(activation())
    return nn.Sequential(*layers)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...] = (64, 64),
        share_backbone: bool = False,
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.share_backbone = share_backbone
        self.policy_trunk = _mlp_trunk([obs_dim, *hidden_sizes])
        self.value_trunk = self.policy_trunk if share_backbone else _mlp_trunk([obs_dim, *hidden_sizes])

        self.actor_mean = layer_init(nn.Linear(hidden_sizes[-1], act_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.ones(1, act_dim) * log_std_init)
        self.critic = layer_init(nn.Linear(hidden_sizes[-1], 1), std=1.0)

    def _policy_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_trunk(x)

    def _value_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_trunk(x)

    def get_dist(self, x: torch.Tensor) -> Independent:
        mean = self.actor_mean(self._policy_features(x))
        std = self.actor_log_std.expand_as(mean).exp()
        return Independent(Normal(mean, std), 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self._value_features(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        dist = self.get_dist(x)
        value = self.get_value(x)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self._policy_features(x))

    def policy_parameters(self):
        params = list(self.policy_trunk.parameters())
        params += list(self.actor_mean.parameters())
        params.append(self.actor_log_std)
        return params

    def value_parameters(self):
        return list(self.value_trunk.parameters()) + list(self.critic.parameters())
