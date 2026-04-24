"""
Minimal gradient-based predictive coding on MNIST.

No precision estimation yet — we just set Sigma = I (identity) everywhere,
which makes the free energy reduce to a sum of squared prediction errors.
This gets you the core PC inference+learning loop so that layering precision
on top later is a small change, not a rewrite.

Mental model:
    Layer 0: input (clamped to image)
    Layer 1: hidden activity phi_1
    Layer 2: hidden activity phi_2
    Layer 3: top activity phi_3, clamped to one-hot label during training

    Predictions flow top-down:
        v_l = f_l(phi_{l+1}) = W_l @ nonlinearity(phi_{l+1})

    Errors are layer-local:
        eps_l = phi_l - v_l

    Free energy (with Sigma = I):
        F = sum_l  0.5 * ||eps_l||^2

Two-loop structure:
    Inner loop (inference): for a fixed batch, hold weights frozen and do
        gradient descent on F wrt the *activities* phi_1, phi_2 until they
        settle. This is the bidirectional message passing.
    Outer loop (learning): after inference settles, do one gradient step on F
        wrt the *weights* W_l.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class PCNet(nn.Module):
    """
    A stack of linear prediction functions f_l(phi_{l+1}) = W_l @ tanh(phi_{l+1}).

    We store one weight matrix per inter-layer edge. Activities phi_l are NOT
    parameters of the module — they are per-batch state created fresh in the
    inference loop (see `infer`).
    """
    def __init__(self, layer_sizes):
        super().__init__()
        # layer_sizes[0] = input dim (784), layer_sizes[-1] = top dim (10)
        self.layer_sizes = layer_sizes
        # W[l] maps layer l+1 down to layer l  (top-down prediction)
        self.W = nn.ModuleList([
            nn.Linear(layer_sizes[l + 1], layer_sizes[l], bias=True)
            for l in range(len(layer_sizes) - 1)
        ])
        self.act = torch.tanh  # nonlinearity applied to phi_{l+1} before W_l

    def predict(self, phi_above, l):
        """Top-down prediction of layer l given activity at layer l+1."""
        return self.W[l](self.act(phi_above))

    def free_energy(self, phis):
        """
        F = sum_l 0.5 * ||phi_l - f_l(phi_{l+1})||^2

        phis is a list [phi_0, phi_1, ..., phi_L]. phi_0 is input-clamped,
        phi_L is target-clamped (during training).
        """
        total = 0.0
        for l in range(len(phis) - 1):
            v_l = self.predict(phis[l + 1], l)
            eps_l = phis[l] - v_l
            total = total + 0.5 * (eps_l ** 2).sum(dim=1).mean()
        return total


# --------------------------------------------------------------------------
# Inference (inner loop) and learning (outer loop)
# --------------------------------------------------------------------------

def _init_phis_topdown(model, x, y_onehot, device):
    """Initialize intermediate activities by sweeping top-down predictions,
    so the inner loop starts from something sensible rather than zeros."""
    sizes = model.layer_sizes
    L = len(sizes) - 1
    B = x.shape[0]
    phis = [None] * (L + 1)
    phis[0] = x
    if y_onehot is not None:
        phis[L] = y_onehot
    else:
        phis[L] = torch.zeros(B, sizes[L], device=device)
    # fill intermediates by predicting downward from the top
    with torch.no_grad():
        for l in range(L - 1, 0, -1):
            phis[l] = model.predict(phis[l + 1], l).clone()
    return phis


def infer(model, x, y_onehot, n_inner_steps=50, inner_lr=0.5):
    """
    Inner loop: hold weights frozen, optimize intermediate activities to
    minimize F. phi_0 clamped to x, phi_L clamped to y_onehot.
    """
    device = x.device
    sizes = model.layer_sizes
    L = len(sizes) - 1

    phis = _init_phis_topdown(model, x, y_onehot, device)
    # Make intermediates into leaf tensors with grad
    for l in range(1, L):
        phis[l] = phis[l].detach().clone().requires_grad_(True)

    free_phis = [phis[l] for l in range(1, L)]

    for _ in range(n_inner_steps):
        F_val = model.free_energy(phis)
        grads = torch.autograd.grad(F_val, free_phis, create_graph=False)
        with torch.no_grad():
            for p, g in zip(free_phis, grads):
                p.data.add_(g, alpha=-inner_lr)

    return [p.detach() for p in phis]


def learn_step(model, phis_settled, weight_optim):
    """
    Given settled activities, compute F once more (this time with weights
    in the graph) and take one optimizer step on the weights.
    """
    weight_optim.zero_grad()
    # Re-wrap activities as non-learnable tensors; only W parameters carry grads
    phis = [p.detach() for p in phis_settled]
    F_val = model.free_energy(phis)
    F_val.backward()
    weight_optim.step()
    return F_val.item()


# --------------------------------------------------------------------------
# Evaluation: classify by finding the label that minimizes F
# --------------------------------------------------------------------------

def classify(model, x, n_classes=10, n_inner_steps=100, inner_lr=0.5):
    """
    Test-time inference: clamp phi_0 = x, let phi_L be free, run the inner
    loop, then take argmax of phi_L.
    """
    device = x.device
    sizes = model.layer_sizes
    L = len(sizes) - 1

    phis = _init_phis_topdown(model, x, y_onehot=None, device=device)
    for l in range(1, L + 1):
        phis[l] = phis[l].detach().clone().requires_grad_(True)
    free_phis = [phis[l] for l in range(1, L + 1)]

    for _ in range(n_inner_steps):
        F_val = model.free_energy(phis)
        grads = torch.autograd.grad(F_val, free_phis, create_graph=False)
        with torch.no_grad():
            for p, g in zip(free_phis, grads):
                p.data.add_(g, alpha=-inner_lr)

    return phis[L].detach().argmax(dim=1)


# --------------------------------------------------------------------------
# Training script
# --------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t.view(-1)),  # flatten 28*28 -> 784
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    # 3-layer PC network (matches "Figure 2" structure: 3 prediction edges)
    #   phi_0 (784)  <-  phi_1 (256)  <-  phi_2 (64)  <-  phi_3 (10)
    model = PCNet([784, 256, 64, 10]).to(device)
    weight_optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 2
    for epoch in range(n_epochs):
        model.train()
        running_F = 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y_oh = F.one_hot(y, num_classes=10).float()
            phis = infer(model, x, y_oh, n_inner_steps=50, inner_lr=0.5)
            f_val = learn_step(model, phis, weight_optim)
            running_F += f_val
            if i % 100 == 0:
                print(f"epoch {epoch} step {i:4d}  F={f_val:.4f}")

        # quick eval on a subset for speed
        model.eval()
        correct, total = 0, 0
        for j, (x, y) in enumerate(test_loader):
            if j >= 8:
                break
            x, y = x.to(device), y.to(device)
            preds = classify(model, x, n_inner_steps=100, inner_lr=0.5)
            correct += (preds == y).sum().item()
            total += y.numel()
        print(f"[epoch {epoch}] test accuracy on {total} images: "
              f"{100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()