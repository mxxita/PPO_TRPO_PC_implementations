"""Atari actor-critic: Nature-DQN CNN (Mnih et al., 2015) + Categorical head."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureEncoder(nn.Module):
    """(N, 4, 84, 84) uint8 -> (N, 512) features."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)), nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float() / 255.0)


class AtariActorCritic(nn.Module):
    """CNN actor-critic with a Categorical policy head.

    `share_backbone=True` mirrors the ppo2 default (shared CNN trunk).
    TRPO needs `share_backbone=False` so its natural-gradient step can target
    the policy parameters without touching the value head.
    """

    def __init__(self, num_actions: int, share_backbone: bool = True, in_channels: int = 4):
        super().__init__()
        self.share_backbone = share_backbone
        self.policy_encoder = NatureEncoder(in_channels)
        self.value_encoder = self.policy_encoder if share_backbone else NatureEncoder(in_channels)
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def get_dist(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=self.actor(self.policy_encoder(x)))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.value_encoder(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        logits = self.actor(self.policy_encoder(x))
        value = self.critic(self.value_encoder(x)).squeeze(-1)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.policy_encoder(x)).argmax(dim=-1)

    def policy_parameters(self):
        return list(self.policy_encoder.parameters()) + list(self.actor.parameters())

    def value_parameters(self):
        # With share_backbone=True these overlap with policy_parameters; that's
        # fine for PPO (one optimizer over everything) but wrong for TRPO.
        return list(self.value_encoder.parameters()) + list(self.critic.parameters())
