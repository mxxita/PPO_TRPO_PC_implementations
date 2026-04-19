"""Network interface used by every agent in this repo.

This is a Protocol (duck-typed) rather than an abstract base class so that
both ``nn.Module``-based networks and future PC networks (which may have a
different forward semantics) can satisfy it without forced inheritance.

Any policy/value net must provide:

    get_dist(obs)                       -> torch.distributions.Distribution
    get_value(obs)                      -> Tensor (batch,)
    get_action_and_value(obs, action=None)
                                         -> (action, logprob, entropy, value)
    deterministic_action(obs)           -> Tensor   (argmax / mean)
    policy_parameters()                 -> list[nn.Parameter]
    value_parameters()                  -> list[nn.Parameter]

TRPO specifically needs ``policy_parameters`` and ``value_parameters`` to be
disjoint (no shared backbone) because its natural-gradient step must not touch
value-head weights.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch.distributions import Distribution


@runtime_checkable
class Policy(Protocol):
    def get_dist(self, obs: torch.Tensor) -> Distribution: ...
    def get_value(self, obs: torch.Tensor) -> torch.Tensor: ...
    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ): ...
    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor: ...
    def policy_parameters(self) -> list: ...
    def value_parameters(self) -> list: ...
