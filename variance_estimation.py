"""
Paper Section 5.1.2 — Variance estimation with noisy data and weights.

Setup (verbatim from the paper):
    - Constant inputs, mean 0, additive Gaussian noise with variance 10.
    - 30% dropout on the top-down prediction at every trial (MC dropout).
    - Two conditions:
        (a) fixed activity posterior   -> only Sigma is updated
        (b) fixed top-down predictions -> only Sigma is updated
    - Variance learning rate 0.1, 10-100 updates, Fig 5 shows ~800 trials.

What we're testing:
    Does a learnable diagonal variance sigma^2, optimized by gradient descent
    on the Gaussian NLL of the prediction error, converge to the true noise
    variance of that error? This is the precision-estimator-as-unit-test.

What this file is NOT:
    It's not a full PC network. There's one prediction edge, one error node,
    and a learnable 2-element variance vector. That's deliberate — 5.1.2
    isolates the precision estimator from everything else.
"""
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# The minimal precision estimator
# ------------------------------------------------------------------
#
# We have:
#   phi: the "observed activity" at the lower layer (2-dim, noisy)
#   v:   the "top-down prediction" (2-dim, also possibly noisy via dropout)
#   eps = phi - v
#   sigma^2 = exp(log_var): learnable per-dim variance
#
# Free energy for one error edge (Gaussian NLL, per sample):
#   F = 0.5 * sum_d [ eps_d^2 / sigma_d^2 + log(sigma_d^2) ]
#
# We drop log(2*pi) as a constant. The log-variance parameter is initialized
# at log(1) = 0 so variance starts at 1 (the paper doesn't specify, so I
# pick a neutral init; try log(0.1) or log(5) to see convergence from below
# or above — both should land at the true variance).


def nll(eps, log_var):
    """Gaussian NLL averaged over batch, summed over dims."""
    var = log_var.exp()
    return 0.5 * ((eps ** 2 / var).sum(dim=1).mean() + log_var.sum())


# ------------------------------------------------------------------
# Experiment (a): fixed activity posterior
# ------------------------------------------------------------------
# phi is clamped to a fresh noisy observation each trial.
# v is computed by a fixed linear "top-down" function with dropout.
# Only log_var gets updated.
#
# The true variance of eps = phi - v has two components:
#   Var(phi) = noise_var (= 10)
#   Var(v)   = dropout contribution (depends on dropout rate and v's scale)
# These add because phi and v are independent across trials.

def experiment_fixed_phi(
    n_trials=800,
    batch_per_trial=64,
    noise_var=10.0,
    dropout_p=0.3,
    v_lr=0.1,
    init_log_var=0.0,
    seed=0,
):
    g = torch.Generator().manual_seed(seed)

    # Constant underlying signals for the two input dims (mean 0 as per paper)
    true_mean = torch.zeros(2)

    # "Top-down prediction" base value — we pick a constant here. What matters
    # for 5.1.2 is that v has some spread once dropout is applied; the paper
    # doesn't specify v's magnitude so we use 1.0.
    v_base = torch.ones(2)

    log_var = torch.zeros(2, requires_grad=True)
    log_var.data.fill_(init_log_var)

    traces = {"phi": [], "v": [], "sigma2": [], "eps": []}

    for t in range(n_trials):
        B = batch_per_trial
        # phi: batch of clamped noisy observations
        phi = true_mean + torch.randn(B, 2, generator=g) * (noise_var ** 0.5)

        # v: base prediction with dropout applied independently per sample.
        mask = (torch.rand(B, 2, generator=g) > dropout_p).float() / (1 - dropout_p)
        v = v_base.unsqueeze(0) * mask

        eps = phi - v

        # One gradient step on log_var using this batch's error statistics
        F = nll(eps, log_var)
        grad = torch.autograd.grad(F, log_var)[0]
        with torch.no_grad():
            log_var.data.add_(grad, alpha=-v_lr)

        # Log the batch mean for visualization (so the input/prediction traces
        # aren't pure noise)
        traces["phi"].append(phi.mean(0).tolist())
        traces["v"].append(v.mean(0).tolist())
        traces["sigma2"].append(log_var.exp().detach().tolist())
        traces["eps"].append(eps.mean(0).tolist())

    return traces


# ------------------------------------------------------------------
# Experiment (b): fixed top-down predictions
# ------------------------------------------------------------------
# Here we freeze v (still with dropout, since the paper explicitly mentions
# dropout is active in this condition too — see "despite the noisiness (30%
# dropout) of the top-down prediction" in the Fig 5 caption).
# phi is the noisy observation. This is structurally identical to (a) in
# our minimal setup — the real difference in the paper is which quantity is
# being updated alongside sigma (phi posterior updates vs. prediction updates
# in the full PC network). With everything frozen except sigma, the two
# conditions produce the same traces. I implement (b) separately anyway
# so the file is a faithful reading of the paper and so you can modify one
# without touching the other.

def experiment_fixed_v(**kwargs):
    return experiment_fixed_phi(**kwargs)  # same in this minimal setup


# ------------------------------------------------------------------
# Plotting — reproduces the style of Figure 5
# ------------------------------------------------------------------

def plot_trace(ax, traces, title, noise_var):
    n = len(traces["sigma2"])
    t = range(n)
    phi = torch.tensor(traces["phi"])
    v = torch.tensor(traces["v"])
    sigma2 = torch.tensor(traces["sigma2"])

    # Match Figure 5's legend: Input 1/2, Variance 1/2, Prediction 1/2, True variance
    ax.plot(t, phi[:, 0], color="tab:blue",   alpha=0.4, lw=0.8, label="Input 1")
    ax.plot(t, phi[:, 1], color="tab:orange", alpha=0.4, lw=0.8, label="Input 2")
    ax.plot(t, sigma2[:, 0], color="tab:green", lw=2, label="Variance 1")
    ax.plot(t, sigma2[:, 1], color="tab:red",   lw=2, label="Variance 2")
    ax.plot(t, v[:, 0], color="tab:purple", alpha=0.5, lw=0.8, label="Prediction 1")
    ax.plot(t, v[:, 1], color="tab:brown",  alpha=0.5, lw=0.8, label="Prediction 2")
    ax.axhline(noise_var, color="k", ls="--", lw=1, label="True variance")

    ax.set_title(title)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Value")
    ax.set_ylim(-12, 14)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def main():
    torch.manual_seed(0)

    traces_a = experiment_fixed_phi(seed=0)
    traces_b = experiment_fixed_v(seed=1)

    # Report convergence
    final_a = traces_a["sigma2"][-1]
    final_b = traces_b["sigma2"][-1]
    print(f"(a) fixed phi   final sigma^2: {final_a}  (target ~= 10 + dropout)")
    print(f"(b) fixed v     final sigma^2: {final_b}  (target ~= 10 + dropout)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_trace(axes[0], traces_a,
               "Variance estimation with fixed inputs", noise_var=10.0)
    plot_trace(axes[1], traces_b,
               "Variance estimation with fixed predictions", noise_var=10.0)
    fig.tight_layout()
    out_path = "fig5_variance_estimation.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()