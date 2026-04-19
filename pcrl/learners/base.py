"""Learner ABC — the *how* of parameter updates.

The agent (PPO, TRPO, ...) builds a loss. The Learner decides how to turn that
loss into weight changes. For backprop this is just Adam + autograd. For
predictive coding it's iterative inference + local updates. Same interface.

Keep this API small on purpose. If you're tempted to add methods, that signal
usually means the agent should own more of its update logic, not the learner.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch


class Learner(ABC):
    """Optimizer-like API. Semantics identical to torch.optim for the backprop
    case; a PC learner may perform iterative inference inside `backward`."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Reset whatever state accumulates between updates (grads, errors, ...)."""

    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Compute the update direction for this loss.

        For backprop: runs ``loss.backward()``.
        For PC: injects ``loss`` as a top-layer target, runs inference, and
        leaves local prediction errors on each layer ready for `step()`.
        """

    @abstractmethod
    def step(self, max_grad_norm: Optional[float] = None) -> None:
        """Apply one update. `max_grad_norm` is a convenience; backprop learners
        clip the global L2 norm; others may ignore it."""

    @abstractmethod
    def set_lr(self, lr: float) -> None:
        """Used by algorithms that anneal the learning rate (e.g. PPO)."""

    @abstractmethod
    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """The parameters this learner is responsible for."""
