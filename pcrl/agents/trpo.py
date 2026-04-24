"""TRPO (Schulman et al., 2015), single-path.

Solves the constrained problem (eq. 12):
    max_theta  L(theta) = E[(pi_theta / pi_old) * A]
    s.t.       E[KL(pi_old || pi_theta)] <= delta

via natural gradient:
  1. Value regression (delegated to a Learner — backprop by default, PC swappable).
  2. Policy gradient  g  at theta_old.
  3. Search direction s = F^{-1} g  via conjugate gradient; F is Fisher
     (Hessian of KL at theta_old) computed through Hessian-vector products.
  4. Step size       beta = sqrt(2 delta / s^T F s).
  5. Backtracking line search: accept first alpha = 0.5^j where KL <= delta
     AND surrogate improved; else revert.

The policy step is bespoke autograd (CG + line search) so it does NOT go
through a Learner — only the value-function regression does.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
from torch.distributions import kl_divergence

from ..learners.base import Learner
from .base import Agent


# ---------- helpers on flat parameter vectors --------------------------------
# this section helps to compute the flat gradients... in TRPO the weights are represented in one huge vector
# the next three functions are used to compute the flat gradients, get and set the flat parameters and compute the fisher vector produc
def get_flat_params(params: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in params])


def set_flat_params(params: List[torch.Tensor], flat: torch.Tensor) -> None:
    idx = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def flat_grad(
    loss: torch.Tensor,
    params: List[torch.Tensor],
    create_graph: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph, allow_unused=True,
    )
    out = []
    for g, p in zip(grads, params):
        out.append((g if g is not None else torch.zeros_like(p)).contiguous().view(-1))
    return torch.cat(out)


# Textbook CG implementaion: (Ax = b -> x = A^{-1}b) without materialising A, only using Av(v) = A @ v products.)
# explanation for dummies: Conjugate Gradient allows TRPO to find the most efficient update direction by solving a complex system of equations using only matrix-vector products, avoiding the impossible task of calculating and storing a massive matrix inverse.
def conjugate_gradients(
    Av: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    n_iters: int = 10,
    tol: float = 1e-10,
) -> torch.Tensor:
    """Solve A x = b without materialising A; Av(v) returns A @ v."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_iters):
        Ap = Av(p)
        alpha = rdotr / (torch.dot(p, Ap) + 1e-10)
        x = x + alpha * p
        r = r - alpha * Ap
        new_rdotr = torch.dot(r, r)
        if new_rdotr < tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x


# ---------- TRPO -------------------------------------------------------------

class TRPO(Agent):
    def __init__(
        self,
        network: torch.nn.Module,
        cfg: Dict,
        value_learner: Learner,
        device: torch.device,
    ):
        self.network = network
        self.cfg = cfg
        self.device = device
        # Only value-function regression is delegated to a Learner. The natural-
        # gradient policy step is implemented directly below.
        self.value_learner = value_learner

    def update(self, buffer, progress_remaining: float = 1.0) -> Dict:
        data = buffer.get()
        v_stats = self._update_value(data)
        p_stats = self._update_policy(data)
        return {**p_stats, **v_stats}

    # ---- value regression ---------------------------------------------------
    # MSE regression on the retuns. The only part of TRPO that  uses the learner, because it is the only part touching hte loss calucaltion... This is where you would swap for PC
    def _update_value(self, data) -> Dict:
        cfg = self.cfg
        obs = data["obs"]
        returns = data["returns"]
        batch_size = obs.shape[0]
        inds = np.arange(batch_size)
        losses = []
        for _ in range(cfg["vf_iters"]):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg["vf_minibatch_size"]):
                mb = inds[start:start + cfg["vf_minibatch_size"]]
                values = self.network.get_value(obs[mb])
                loss = 0.5 * ((values - returns[mb]) ** 2).mean()
                self.value_learner.zero_grad()
                self.value_learner.backward(loss)
                self.value_learner.step(max_grad_norm=cfg["max_grad_norm"])
                losses.append(loss.item())
        return {"value_loss": float(np.mean(losses))}

    # ---- policy step --------------------------------------------------------
    def _update_policy(self, data) -> Dict:
        cfg = self.cfg
        obs = data["obs"]
        actions = data["actions"]
        old_logprobs = data["logprobs"].detach()
        adv = data["advantages"]
        if cfg.get("norm_adv", True):
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Snapshot old distribution (no graph attached). Works for Categorical
        # (Atari) and Independent(Normal) (MuJoCo) alike via kl_divergence dispatch.
        with torch.no_grad():
            old_dist = self.network.get_dist(obs)

        params = self.network.policy_parameters()

        # L(theta) in the paper. i..e the ratio-weighted advantage. 
        def neg_surrogate() -> torch.Tensor:
            new_dist = self.network.get_dist(obs)
            new_logprob = new_dist.log_prob(actions)
            ratio = (new_logprob - old_logprobs).exp()
            return -(ratio * adv).mean()

        def mean_kl() -> torch.Tensor:
            new_dist = self.network.get_dist(obs)
            return kl_divergence(old_dist, new_dist).mean()

        # (1) g = grad L(theta_old)
        loss0 = neg_surrogate()
        g = -flat_grad(loss0, params, retain_graph=False)

        # (2) Fisher-vector product via cached first-order KL gradient
        kl = mean_kl()
        kl_grads = torch.autograd.grad(kl, params, create_graph=True)
        flat_kl_grads = torch.cat([gr.view(-1) for gr in kl_grads])

        def Fv(v: torch.Tensor) -> torch.Tensor:
            gv = (flat_kl_grads * v).sum()
            hvp = torch.autograd.grad(gv, params, retain_graph=True)
            return torch.cat([h.contiguous().view(-1) for h in hvp]) + cfg["cg_damping"] * v

        # (3) Conjugate gradient -> search direction
        step_dir = conjugate_gradients(Fv, g, n_iters=cfg["cg_iters"])

        # (4) Step size from the KL trust region
        shs = 0.5 * torch.dot(step_dir, Fv(step_dir))
        shs = shs.clamp(min=1e-10)
        lm = torch.sqrt(shs / cfg["delta"])
        full_step = step_dir / lm
        expected_improve = torch.dot(g, full_step).item()

        # (5) Backtracking line search
        old_flat = get_flat_params(params)
        old_L = -loss0.item()
        success = False
        step_size = 0.0
        final_kl = 0.0
        new_L = old_L
        for i in range(cfg["line_search_steps"]):
            alpha = 0.5 ** i
            set_flat_params(params, old_flat + alpha * full_step)
            with torch.no_grad():
                new_L = -neg_surrogate().item()
                final_kl = mean_kl().item()
            if final_kl <= cfg["delta"] and new_L > old_L:
                success = True
                step_size = alpha
                break
        if not success:
            set_flat_params(params, old_flat)
            final_kl = 0.0
            new_L = old_L

        return dict(
            policy_loss=float(-old_L),
            surrogate_improve=float(new_L - old_L),
            expected_improve=float(expected_improve),
            kl=float(final_kl),
            line_search_success=int(success),
            step_size=float(step_size),
        )
