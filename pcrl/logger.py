"""Minimal CSV logger.

First call establishes the schema; subsequent calls must share the same keys
(missing keys fill with empty strings). Swap for a richer backend (W&B,
TensorBoard) without touching training code: the agent hands a dict to
`.log()` and doesn't care what's underneath.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file = None
        self._writer = None
        self._keys = None

    def log(self, data: Dict) -> None:
        if self._writer is None:
            self._file = open(self.path, "w", newline="")
            self._keys = list(data.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._keys)
            self._writer.writeheader()
        row = {k: data.get(k, "") for k in self._keys}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
