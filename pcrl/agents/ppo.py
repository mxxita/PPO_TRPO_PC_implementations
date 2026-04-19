"""PPO (Schulman et al., 2017) with clipped surrogate objective.

Eq. (7) for L^CLIP, eq. (9) for the combined actor-critic-entropy loss.
The optimizer is wrapped behind a Learner: swap `BackpropLearner` for a PC
learner to run the same algorithm with predictive coding.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from ..learners.base import Learner
from .base import Agent


class PPO(Agent):
    def __init__(self, network: nn.Module, cfg: Dict, learner: Learner, device: torch.device):
        self.network = network
        self.cfg = cfg
        self.device = device
        self.learner = learner

    def update(self, buffer, progress_remaining: float = 1.0) -> Dict:
        cfg = self.cfg

        # Paper: lr and clip both scaled by alpha (progress_remaining).
        lr = cfg["lr"] * (progress_remaining if cfg.get("anneal_lr", True) else 1.0)
        self.learner.set_lr(lr)
        clip_coef = cfg["clip_coef"] * (progress_remaining if cfg.get("anneal_clip", True) else 1.0)

        data = buffer.get()
        batch_size = data["obs"].shape[0]
        mb_size = cfg["minibatch_size"]
        inds = np.arange(batch_size)

        target_kl = cfg.get("target_kl", None)  # auxiliary detail #3: early stopping

        sum_pg_loss = sum_v_loss = sum_ent = 0.0
        sum_approx_kl = sum_clipfrac = 0.0
        n_updates = 0

        for _epoch in range(cfg["update_epochs"]):
            np.random.shuffle(inds)
            for start in range(0, batch_size, mb_size):
                mb = inds[start:start + mb_size]

                _, new_logprob, entropy, new_value = self.network.get_action_and_value(
                    data["obs"][mb], data["actions"][mb]
                )
                logratio = new_logprob - data["logprobs"][mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    mb_approx_kl = ((ratio - 1) - logratio).mean().item()  # k3 estimator
                    mb_clipfrac = (torch.abs(ratio - 1) > clip_coef).float().mean().item()

                if target_kl is not None and mb_approx_kl > 1.5 * target_kl:
                    break

                adv = data["advantages"][mb]
                if cfg.get("norm_adv", True):
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Clipped surrogate (eq. 7): pessimistic (min) -> negate to minimise.
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)
                if cfg.get("clip_vloss", True):
                    v_unclipped = (new_value - data["returns"][mb]) ** 2
                    v_clipped = data["values"][mb] + torch.clamp(
                        new_value - data["values"][mb], -clip_coef, clip_coef
                    )
                    v_clipped_loss = (v_clipped - data["returns"][mb]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_clipped_loss).mean()
                else:
                    v_loss = 0.5 * ((new_value - data["returns"][mb]) ** 2).mean()

                ent = entropy.mean()
                loss = pg_loss - cfg["ent_coef"] * ent + cfg["vf_coef"] * v_loss

                self.learner.zero_grad()
                self.learner.backward(loss)
                self.learner.step(max_grad_norm=cfg["max_grad_norm"])

                sum_pg_loss += pg_loss.item()
                sum_v_loss += v_loss.item()
                sum_ent += ent.item()
                sum_approx_kl += mb_approx_kl
                sum_clipfrac += mb_clipfrac
                n_updates += 1
            else:
                continue
            break  # inner loop broke (early stopping) -> break outer too

        return dict(
            policy_loss=sum_pg_loss / max(n_updates, 1),
            value_loss=sum_v_loss / max(n_updates, 1),
            entropy=sum_ent / max(n_updates, 1),
            approx_kl=sum_approx_kl / max(n_updates, 1),
            clipfrac=sum_clipfrac / max(n_updates, 1),
            lr=lr,
            clip_coef=clip_coef,
        )
