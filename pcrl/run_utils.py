"""Run-directory setup + reproducibility metadata.

Every run dir ends up with:

    config.yaml     -- the resolved, composed config (with CLI overrides applied)
    env.yaml        -- git hash, python/torch/gymnasium versions, host, command
    progress.csv    -- per-update metrics (written by CSVLogger during training)
    model.pt        -- final weights (written by train.py at exit)

`env.yaml` is the piece most people skip and regret later: without it you
can't reproduce a run six months from now when your dependency versions
have drifted.
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _git_info(repo_root: Path) -> Dict[str, Any]:
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return {"git_hash": rev, "git_dirty": bool(status)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_hash": None, "git_dirty": None}


def _lib_versions() -> Dict[str, Optional[str]]:
    import importlib
    versions: Dict[str, Optional[str]] = {}
    for pkg in ("torch", "numpy", "gymnasium", "pandas", "matplotlib"):
        try:
            versions[pkg] = importlib.import_module(pkg).__version__
        except Exception:
            versions[pkg] = None
    return versions


def write_run_metadata(run_dir: Path, cfg: Dict, extra: Optional[Dict] = None) -> None:
    """Freeze the resolved config + run environment into `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent
    meta = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cmd": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        **_git_info(repo_root),
        "libs": _lib_versions(),
    }
    if extra:
        meta.update(extra)
    with open(run_dir / "env.yaml", "w") as f:
        yaml.dump(meta, f, sort_keys=False)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)


def default_run_name(algo: str, env_id: str, seed: int) -> str:
    stamp = int(time.time())
    return f"{algo}_{env_id.replace('/', '-')}_s{seed}_{stamp}"
