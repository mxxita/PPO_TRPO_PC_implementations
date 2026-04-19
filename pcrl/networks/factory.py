"""Pick the right actor-critic class from the env's observation/action spaces."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch.nn as nn

from .mlp import MLPActorCritic
from .nature_cnn import AtariActorCritic


def make_actor_critic(envs: gym.vector.VectorEnv, share_backbone: bool) -> nn.Module:
    obs_space = envs.single_observation_space
    act_space = envs.single_action_space

    if isinstance(act_space, gym.spaces.Discrete):
        in_channels = obs_space.shape[0]  # Atari: (C, H, W) uint8
        return AtariActorCritic(
            num_actions=int(act_space.n),
            share_backbone=share_backbone,
            in_channels=in_channels,
        )
    if isinstance(act_space, gym.spaces.Box):
        return MLPActorCritic(
            obs_dim=int(np.prod(obs_space.shape)),
            act_dim=int(np.prod(act_space.shape)),
            share_backbone=share_backbone,
        )
    raise ValueError(f"Unsupported action space: {act_space}")
