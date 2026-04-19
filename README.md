# pcrl — PPO & TRPO Paper Implementations

Clean PyTorch implementations of **PPO** ([Schulman 2017](https://arxiv.org/abs/1707.06347))
and **TRPO** ([Schulman 2015](https://arxiv.org/abs/1502.05477), single-path)
on **Atari** and **MuJoCo**. Auto-detects MPS / CUDA / CPU.

Factored so the training loop, networks, and buffer are shared — and so
the optimizer is a plug (swap Adam backprop for predictive coding without
touching the algorithms).

Should be extended to include pc too - TODO

---

## Quickstart

```bash
source /Users/maritaberger/venvs/pc/bin/activate
pip install -e .           # + [dev] for pytest / ruff

# smoke test (~1 min)
python scripts/train.py --experiment ppo_atari \
  --env-id PongNoFrameskip-v4 --total-timesteps 50_000 --seed 0
```

Each run writes to `runs/<name>/`:

| file | what |
|------|------|
| `config.yaml` | fully-resolved hyperparams |
| `env.yaml` | git hash, library versions, host, command line |
| `progress.csv` | one row per update |
| `model.pt` | final weights |

---

## Common commands

```bash
# full PPO Atari run
python scripts/train.py --experiment ppo_atari --env-id BreakoutNoFrameskip-v4 --seed 0

# TRPO on MuJoCo with CLI overrides (YAML-parsed, nested keys OK)
python scripts/train.py --experiment trpo_mujoco \
    --env-id Hopper-v4  vf_lr=5e-4  learner.eps=1e-7

# evaluate a checkpoint
python scripts/evaluate.py runs/ppo_PongNoFrameskip-v4_s0_... --episodes 5 --render

# sweep: 3 seeds × 2 experiments × 2 envs, 2 in parallel
python scripts/sweep.py --parallel 2

# plot: one line per run, or mean ± std across seeds
python scripts/plot.py runs/ppo_Pong* --smooth 20 --out pong.png
python scripts/plot_sweep.py runs/* --out sweep.png
```

---

## Repo layout

```
pcrl/
├── agents/       Agent ABC + PPO + TRPO         ← what to optimize
├── learners/     Learner ABC + BackpropLearner  ← how to optimize (swap for PC here)
├── networks/     Policy Protocol + CNN + MLP
├── envs.py       Atari / MuJoCo wrappers + make_env
├── buffer.py     RolloutBuffer + GAE
├── config.py     YAML loader with `defaults:` composition
├── run_utils.py  metadata freeze (git hash, library versions, cmd)
└── {logger,device,seeding}.py

scripts/          train.py · evaluate.py · plot.py · plot_sweep.py · sweep.py
configs/
├── base/         algo/{ppo,trpo}.yaml · env/{atari,mujoco}.yaml · learner/backprop.yaml
└── experiments/  {ppo,trpo}_{atari,mujoco}.yaml   ← compose the base fragments
```

### What actually differs between PPO and TRPO?

Only `update()`:

- **PPO** — K epochs of minibatch SGD on the clipped surrogate (eq. 7).
- **TRPO** — Adam regression on the value head, then one natural-gradient
  policy step via conjugate gradient + backtracking line search, enforcing
  `KL(π_old ‖ π_new) ≤ δ` exactly.

---

## Configs: compose, don't copy

```yaml
# configs/experiments/ppo_atari.yaml
defaults:
  - ../base/algo/ppo.yaml
  - ../base/env/atari.yaml
  - ../base/learner/backprop.yaml

lr: 2.5e-4
clip_coef: 0.1
update_epochs: 4
# ...
```

Later entries in `defaults:` override earlier ones; top-level keys
override everything. Full loader is ~30 lines in
[`pcrl/config.py`](pcrl/config.py). Forward-compatible with real Hydra.

---

## Extending with predictive coding (or any non-backprop learner)

The **Learner** abstraction is the hook point:

```python
# pcrl/learners/base.py
class Learner(ABC):
    def zero_grad(self): ...
    def backward(self, loss): ...            # or: propagate errors locally
    def step(self, max_grad_norm=None): ...  # or: one PC inference pass
    def set_lr(self, lr): ...
    def parameters(self): ...
```

To add a PC learner:

1. Subclass `Learner` under `pcrl/learners/predictive_coding.py`.
2. Add `configs/base/learner/predictive_coding.yaml` with
   `learner.type: predictive_coding`.
3. Extend `build_learner()` in [`scripts/train.py`](scripts/train.py) to
   dispatch on `learner.type`.

**PPO plugs straight in** (it only ever calls
`learner.{zero_grad, backward, step, set_lr}`). **TRPO** takes a learner
only for its *value-function* regression; the policy step is bespoke
because Fisher-vector products need direct autograd access, which is
incompatible with a local-update learner. See
[`pcrl/agents/trpo.py`](pcrl/agents/trpo.py).

To add a new algorithm: subclass `Agent`, implement `update()`, extend
`build_agent()` in `scripts/train.py`.

---

## Reproducibility

Every run freezes: git commit + dirty flag, torch/numpy/gymnasium/pandas
versions, python version, platform, hostname, and the exact launch
command, into `runs/<name>/env.yaml`. Paired with `config.yaml` and the
seed in the run name, this is enough to reproduce bit-for-bit on the same
hardware.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: pcrl` | `pip install -e .` inside the `pc` venv |
| Atari ROMs missing | `AutoROM --accept-license` |
| `operator X not implemented for MPS` | `--device cpu`; update PyTorch |
| MuJoCo: "Could not find module 'mujoco'" | `pip install mujoco "gymnasium[mujoco]"` |
| TRPO line search fails every iteration | decrease `delta` or `cg_damping` |
| Runs look noisy | average ≥3 seeds; use `plot_sweep.py` |

---

## References

- Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- Schulman et al. *Trust Region Policy Optimization.* ICML 2015 / arXiv:1502.05477.
- Schulman et al. *GAE.* arXiv:1506.02438, 2015.
- Mnih et al. *Human-level control through deep reinforcement learning.* Nature 518, 2015.
- Huang et al. *The 37 Implementation Details of PPO.* ICLR Blog Track, 2022.
