"""Standard autograd + Adam. The baseline learner.

Matches what every canonical RL implementation does internally: call
``loss.backward()``, optionally clip the global gradient norm, then
``optimizer.step()``.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch.optim import Adam

from .base import Learner


class BackpropLearner(Learner):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        eps: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
    ):
        self._params = list(params)
        self.optimizer = Adam(self._params, lr=lr, eps=eps, betas=betas)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def step(self, max_grad_norm: Optional[float] = None) -> None:
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._params, max_grad_norm)
        self.optimizer.step()

    def set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def parameters(self):
        return self._params
