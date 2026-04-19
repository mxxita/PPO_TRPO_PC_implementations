"""Env factories for Atari and MuJoCo, both returning a SyncVectorEnv.

Atari output:  uint8, shape (num_envs, 4, 84, 84)  -- standard Mnih preprocessing.
MuJoCo output: float32, shape (num_envs, obs_dim) with running-mean normalisation.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

# Register ALE envs on recent gymnasium versions.
try:
    import ale_py  # noqa: F401
    if hasattr(gym, "register_envs"):
        gym.register_envs(ale_py)
except ImportError:
    pass

# FrameStack was renamed to FrameStackObservation in gymnasium 1.x.
try:
    from gymnasium.wrappers import FrameStackObservation as FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStack  # type: ignore


# -----------------------------------------------------------------------------
# Helper wrappers (avoid gymnasium API drift between 0.29 and 1.x)
# -----------------------------------------------------------------------------

class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset for envs that require it (e.g., Breakout)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self._has_fire = len(action_meanings) >= 3 and action_meanings[1] == "FIRE"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if not self._has_fire:
            return obs, info
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1} (Mnih et al. 2015)."""

    def reward(self, reward):
        return float(np.sign(reward))


class ClipObservation(gym.ObservationWrapper):
    """Clamp observations to a symmetric range (used after NormalizeObservation)."""

    def __init__(self, env: gym.Env, low: float = -10.0, high: float = 10.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def observation(self, obs):
        return np.clip(obs, self.low, self.high).astype(np.float32)


class ClipReward(gym.RewardWrapper):
    def __init__(self, env: gym.Env, low: float = -10.0, high: float = 10.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def reward(self, reward):
        return float(np.clip(reward, self.low, self.high))


# -----------------------------------------------------------------------------
# Atari
# -----------------------------------------------------------------------------

def _make_atari_single(env_id: str, seed: int, idx: int, render_mode: str | None):
    def thunk():
        env = gym.make(env_id, frameskip=1, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            scale_obs=False,  # keep uint8
        )
        env = FireResetEnv(env)
        env = FrameStack(env, 4)
        env = ClipRewardEnv(env)
        env.action_space.seed(seed + idx)
        return env
    return thunk


def make_atari_env(env_id: str, seed: int, num_envs: int, render_mode: str | None = None):
    fns = [_make_atari_single(env_id, seed, i, render_mode) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(fns)


# -----------------------------------------------------------------------------
# MuJoCo
# -----------------------------------------------------------------------------

def _make_mujoco_single(env_id: str, seed: int, idx: int, render_mode: str | None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = ClipObservation(env, -10.0, 10.0)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = ClipReward(env, -10.0, 10.0)
        env.action_space.seed(seed + idx)
        return env
    return thunk


def make_mujoco_env(env_id: str, seed: int, num_envs: int, render_mode: str | None = None):
    fns = [_make_mujoco_single(env_id, seed, i, render_mode) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(fns)


# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------

def make_env(env_type: str, env_id: str, seed: int, num_envs: int, render_mode: str | None = None):
    env_type = env_type.lower()
    if env_type == "atari":
        return make_atari_env(env_id, seed, num_envs, render_mode)
    if env_type == "mujoco":
        return make_mujoco_env(env_id, seed, num_envs, render_mode)
    raise ValueError(f"Unknown env_type: {env_type!r} (expected 'atari' or 'mujoco')")
