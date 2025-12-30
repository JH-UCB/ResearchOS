# sciml_pipeline_template

A lightweight, research-grade experiment runner for **Scientific Machine Learning (SciML)**.

This template is intentionally small (pure Python + PyTorch) but is structured to scale:
- supervised ML (classification/regression)
- PDE-constrained learning (PINNs / residual minimization)
- inverse problems (data misfit + regularization, with optional learned priors)
- operator- and geometry-aware modeling (Fourier features, coordinate networks, etc.)

It is built around *three* core abstractions:

1. **Task** (problem definition)
   - data generation / sampling / loading
   - loss terms (data loss, physics residual, constraints)
   - metrics and evaluation logic
2. **Model** (parameterization)
   - an `nn.Module` that maps inputs to outputs (or directly stores unknowns, e.g. ParamVector)
3. **Trainer** (optimization loop)
   - generic training loop with checkpointing, early stopping, and experiment logging

## Quickstart

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run an example

#### (A) PINN: 1D Poisson problem
```bash
python -m sciml_pipeline.run --config configs/pinn_poisson1d.yaml
```

#### (B) Inverse problem: 1D deconvolution
```bash
python -m sciml_pipeline.run --config configs/inverse_deconvolution1d.yaml
```

#### (C) Supervised baseline: classification from NPZ
Create an `.npz` file with arrays:
- `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`
then run:
```bash
python -m sciml_pipeline.run --config configs/classification_npz.yaml
```

## Output layout

Runs are stored under `runs/<run_id>/`:

- `config.yaml`              Copy of the configuration
- `events.jsonl`             Step/epoch logs (JSON lines)
- `summary.json`             Final metrics + metadata
- `checkpoints/best.pt`      Best model checkpoint
- `checkpoints/last.pt`      Last model checkpoint
- `artifacts/`               Task-specific outputs (plots, arrays, etc.)

At the project root, `runs/experiments.csv` tracks run status and key metrics.

## Extending the template

### Add a new task
1. Create `sciml_pipeline/tasks/my_task.py`
2. Define:
   - a `MyTaskConfig` dataclass
   - a `MyTask(Task)` implementing the required methods
3. Register it via `@register_task("my_task")`
4. Add config parsing to `TaskConfigFactory` in `config.py`

### Add a new model
1. Create `sciml_pipeline/models/my_model.py`
2. Define:
   - a `MyModelConfig` dataclass
   - a `MyModel(nn.Module)`
3. Register via `@register_model("my_model")`
4. Add config parsing to `ModelConfigFactory` in `config.py`

## Notes on SciML scaling

This template keeps the interfaces general enough to support:
- PDE residual losses (PINNs)
- hybrid solvers: differentiable solvers + neural components
- reduced-order modeling (POD/PCA-like compression)
- manifold-valued learning (via projection hooks in the trainer)

For multi-physics / coupled systems, define the Task loss as a weighted sum of terms
and treat constraints as additional loss components (or projections).

## Sweeps

Run every YAML config in a directory:
```bash
python -m sciml_pipeline.sweep --config_dir configs
```

Configs starting with `_` are ignored by convention.

## SciML primitives included

- `sciml_pipeline/operators.py`: grad/jacobian/divergence/laplacian/hessian via autograd
- `sciml_pipeline/sampling.py`: uniform / Sobol / Latin hypercube sampling + hypercube boundary sampling
- `sciml_pipeline/solvers/poisson1d_fd.py`: finite-difference solver baseline (no SciPy)
- `sciml_pipeline/manifolds.py`: sphere manifold with exp/log/dist + in-place projection helper

## Directory Structure 

```bash
.
├── configs
│   ├── classification_npz.yaml
│   ├── inverse_deconvolution1d.yaml
│   └── pinn_poisson1d.yaml
├── LICENSE
├── README.md
├── requirements.txt
├── runs
│   ├── 6c48cf6307dc
│   │   ├── artifacts
│   │   ├── checkpoints
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   ├── config.yaml
│   │   ├── events.jsonl
│   │   └── summary.json
│   ├── e8c5355cbc8e
│   │   ├── artifacts
│   │   ├── checkpoints
│   │   ├── config.yaml
│   │   └── events.jsonl
│   └── experiments.csv
└── sciml_pipeline
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-310.pyc
    │   ├── config.cpython-310.pyc
    │   ├── hashing.cpython-310.pyc
    │   ├── logging.cpython-310.pyc
    │   ├── metrics.cpython-310.pyc
    │   ├── paths.cpython-310.pyc
    │   ├── registry.cpython-310.pyc
    │   ├── run.cpython-310.pyc
    │   ├── seed.cpython-310.pyc
    │   └── utils.cpython-310.pyc
    ├── config.py
    ├── hashing.py
    ├── logging.py
    ├── manifolds.py
    ├── metrics.py
    ├── models
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── cnn1d.cpython-310.pyc
    │   │   ├── fourier_mlp.cpython-310.pyc
    │   │   ├── mlp.cpython-310.pyc
    │   │   └── param_vector.cpython-310.pyc
    │   ├── cnn1d.py
    │   ├── fourier_mlp.py
    │   ├── mlp.py
    │   └── param_vector.py
    ├── operators.py
    ├── paths.py
    ├── registry.py
    ├── run.py
    ├── sampling.py
    ├── seed.py
    ├── solvers
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── poisson1d_fd.cpython-310.pyc
    │   └── poisson1d_fd.py
    ├── sweep.py
    ├── tasks
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── classification_npz.cpython-310.pyc
    │   │   ├── inverse_deconvolution1d.cpython-310.pyc
    │   │   └── pinn_poisson1d.cpython-310.pyc
    │   ├── classification_npz.py
    │   ├── inverse_deconvolution1d.py
    │   └── pinn_poisson1d.py
    ├── trainers
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   └── torch_trainer.cpython-310.pyc
    │   └── torch_trainer.py
    └── utils.py

18 directories, 64 files
```