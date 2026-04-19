"""Minimal YAML config loader with a Hydra-lite `defaults:` composition.

Example experiment file:

    # configs/experiments/ppo_atari.yaml
    defaults:
      - ../base/algo/ppo.yaml
      - ../base/env/atari.yaml
      - ../base/learner/backprop.yaml

    env_id: BreakoutNoFrameskip-v4
    total_timesteps: 10_000_000

Defaults are resolved relative to the config file's own directory, loaded in
listed order, and merged with a deep update. The top-level file's explicit
keys override anything from defaults. Defaults can themselves have defaults.

This is intentionally ~30 lines of Python so you can read it end-to-end. If
you outgrow it, migrate to Hydra — the config format is forward-compatible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, Mapping):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config, resolving `defaults:` recursively."""
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    defaults = raw.pop("defaults", []) or []
    out: Dict[str, Any] = {}
    for default in defaults:
        default_path = (path.parent / default).resolve()
        _deep_update(out, load_config(default_path))
    _deep_update(out, raw)
    return out


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """Apply CLI overrides of the form `key=value` or `nested.key=value`.

    Values are YAML-parsed so booleans, numbers, lists all round-trip:
        lr=1e-4   num_envs=16   anneal_lr=false   learner.eps=1.0e-7
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be KEY=VALUE, got: {ov!r}")
        k, v = ov.split("=", 1)
        parsed = yaml.safe_load(v)
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = parsed
    return cfg
