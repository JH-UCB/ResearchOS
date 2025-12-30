"""
Configuration system for sciml_pipeline.

Design goals:
- Human-editable experiment specs (YAML)
- Deterministic run identification (hash of config)
- Strong defaults + explicit parameters (research reproducibility)

We intentionally avoid heavy config frameworks; YAML + dataclasses are enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import yaml

from .seed import SeedConfig


# =============================================================================
# RUN / TRAINER / OPTIM CONFIGS
# =============================================================================

@dataclass(frozen=True)
class RunConfig:
    """Run-level configuration (filesystem + resume semantics)."""
    root_dir: str = "runs"
    experiment_name: str = "experiment"
    notes: str = ""
    resume_if_completed: bool = True
    overwrite_run_dir: bool = False  # if True, delete/overwrite run_dir (dangerous)


@dataclass(frozen=True)
class TrainerConfig:
    """Generic training-loop settings."""
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    max_epochs: int = 200
    steps_per_epoch: Optional[int] = None  # used for iterable/sampler tasks
    grad_clip_norm: Optional[float] = None
    early_stopping_patience: int = 50
    log_every_epochs: int = 1
    eval_every_epochs: int = 1
    mixed_precision: bool = False  # only effective on CUDA


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer + scheduler settings (Adam + optional ReduceLROnPlateau)."""
    lr: float = 1e-3
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler: ReduceLROnPlateau
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 20
    plateau_min_lr: float = 1e-6


# =============================================================================
# TASK CONFIGS
# =============================================================================

@dataclass(frozen=True)
class TaskConfigBase:
    name: str


@dataclass(frozen=True)
class ClassificationNPZTaskConfig(TaskConfigBase):
    """Supervised classification from a `.npz` file."""
    name: str = "classification_npz"

    # Path to npz file containing arrays:
    # X_train, y_train, X_val, y_val, X_test, y_test
    npz_path: str = "data/dataset.npz"

    # DataLoader settings
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False

    # Labels
    num_classes: int = 2


@dataclass(frozen=True)
class PINNPoisson1DTaskConfig(TaskConfigBase):
    """1D Poisson PINN: -u''(x) = f(x) on x in [0,1], u(0)=u(1)=0."""
    name: str = "pinn_poisson1d"

    # Sampling
    n_interior: int = 512        # interior collocation points per epoch
    n_boundary: int = 64         # boundary points per epoch (split across boundaries)
    steps_per_epoch: int = 50    # gradient steps per epoch

    # Loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0

    # Evaluation grid (for metrics)
    n_eval: int = 256

    # Which forcing function / exact solution family
    # We implement a default where u*(x) = sin(pi x) and f(x) = pi^2 sin(pi x).
    forcing: Literal["sin_pi"] = "sin_pi"


@dataclass(frozen=True)
class InverseDeconvolution1DTaskConfig(TaskConfigBase):
    """1D deconvolution inverse problem y = k * x + noise."""
    name: str = "inverse_deconvolution1d"

    # Signal generation (synthetic demo)
    signal_length: int = 256
    kernel_sigma: float = 2.0
    noise_std: float = 0.03

    # Loss weights
    w_data: float = 1.0
    w_tv: float = 0.01  # total variation regularization strength

    # Training
    steps_per_epoch: int = 200  # gradient steps per epoch (single observation)
    n_eval: int = 256           # points used for evaluation/plots

    # If True, save arrays to artifacts/
    save_artifacts: bool = True

    # If True, also save plots (requires matplotlib)
    save_plots: bool = False


# =============================================================================
# MODEL CONFIGS
# =============================================================================

@dataclass(frozen=True)
class ModelConfigBase:
    name: str


@dataclass(frozen=True)
class MLPModelConfig(ModelConfigBase):
    """General MLP. Works for classification or coordinate networks."""
    name: str = "mlp"
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    activation: Literal["relu", "tanh", "silu"] = "tanh"
    dropout: float = 0.0
    use_batchnorm: bool = False


@dataclass(frozen=True)
class FourierMLPModelConfig(ModelConfigBase):
    """Fourier-feature MLP for coordinate-based learning (PINNs, implicit reps)."""
    name: str = "fourier_mlp"
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: Literal["relu", "tanh", "silu"] = "tanh"
    dropout: float = 0.0
    use_batchnorm: bool = False

    # Fourier feature settings: x -> [sin(2πBx), cos(2πBx)]
    fourier_features: int = 64
    fourier_scale: float = 10.0


@dataclass(frozen=True)
class CNN1DModelConfig(ModelConfigBase):
    """Simple 1D CNN. Useful as a learned prior for inverse problems."""
    name: str = "cnn1d"
    channels: List[int] = field(default_factory=lambda: [32, 64, 64])
    kernel_size: int = 5
    activation: Literal["relu", "silu"] = "silu"
    use_batchnorm: bool = True


@dataclass(frozen=True)
class ParamVectorModelConfig(ModelConfigBase):
    """A learnable vector parameter: the 'model' is the unknown itself."""
    name: str = "param_vector"
    length: int = 256
    init: Literal["zeros", "normal"] = "zeros"
    init_scale: float = 0.01



@dataclass(frozen=True)
class _UnsetTaskConfig(TaskConfigBase):
    """Placeholder task config used before parsing."""
    name: str = "unset"


@dataclass(frozen=True)
class _UnsetModelConfig(ModelConfigBase):
    """Placeholder model config used before parsing."""
    name: str = "unset"


# =============================================================================
# TOP-LEVEL EXPERIMENT CONFIG
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    task: TaskConfigBase = field(default_factory=_UnsetTaskConfig)   # overwritten by from_dict
    model: ModelConfigBase = field(default_factory=_UnsetModelConfig)  # overwritten by from_dict

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a nested dict suitable for hashing and saving."""
        def dc_to_dict(x: Any) -> Dict[str, Any]:
            # dataclasses.asdict is fine, but we want to avoid importing it here.
            return x.__dict__.copy()

        return {
            "run": dc_to_dict(self.run),
            "seed": dc_to_dict(self.seed),
            "trainer": dc_to_dict(self.trainer),
            "optim": dc_to_dict(self.optim),
            "task": dc_to_dict(self.task),
            "model": dc_to_dict(self.model),
        }


# =============================================================================
# FACTORIES: dict -> dataclass
# =============================================================================

class TaskConfigFactory:
    _MAP: Dict[str, Type[TaskConfigBase]] = {
        "classification_npz": ClassificationNPZTaskConfig,
        "pinn_poisson1d": PINNPoisson1DTaskConfig,
        "inverse_deconvolution1d": InverseDeconvolution1DTaskConfig,
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TaskConfigBase:
        name = d.get("name", None)
        if name not in cls._MAP:
            raise KeyError(f"Unknown task name '{name}'. Known: {sorted(cls._MAP.keys())}")
        T = cls._MAP[str(name)]
        return T(**d)  # type: ignore[arg-type]


class ModelConfigFactory:
    _MAP: Dict[str, Type[ModelConfigBase]] = {
        "mlp": MLPModelConfig,
        "fourier_mlp": FourierMLPModelConfig,
        "cnn1d": CNN1DModelConfig,
        "param_vector": ParamVectorModelConfig,
    }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ModelConfigBase:
        name = d.get("name", None)
        if name not in cls._MAP:
            raise KeyError(f"Unknown model name '{name}'. Known: {sorted(cls._MAP.keys())}")
        T = cls._MAP[str(name)]
        return T(**d)  # type: ignore[arg-type]


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a Python dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {path}")
    return data


def build_experiment_config(data: Dict[str, Any]) -> ExperimentConfig:
    """Build ExperimentConfig from a nested dict.

    Expected top-level keys:
    - run, seed, trainer, optim, task, model
    """
    run = RunConfig(**(data.get("run", {}) or {}))
    seed = SeedConfig(**(data.get("seed", {}) or {}))
    trainer = TrainerConfig(**(data.get("trainer", {}) or {}))
    optim = OptimConfig(**(data.get("optim", {}) or {}))

    task_dict = data.get("task", None)
    model_dict = data.get("model", None)
    if not isinstance(task_dict, dict):
        raise ValueError("Config must include a 'task' mapping")
    if not isinstance(model_dict, dict):
        raise ValueError("Config must include a 'model' mapping")

    task = TaskConfigFactory.from_dict(task_dict)
    model = ModelConfigFactory.from_dict(model_dict)

    # Small compatibility sanity checks
    if isinstance(task, PINNPoisson1DTaskConfig) and isinstance(model, ParamVectorModelConfig):
        raise ValueError("PINNPoisson1DTask does not support ParamVectorModel (needs a coordinate network).")

    if isinstance(task, InverseDeconvolution1DTaskConfig) and isinstance(model, ParamVectorModelConfig):
        if model.length != task.signal_length:
            raise ValueError(
                f"ParamVector length ({model.length}) must match task.signal_length ({task.signal_length})"
            )

    if isinstance(task, PINNPoisson1DTaskConfig):
        # If user forgot to set trainer.steps_per_epoch, we can derive it from task.
        if trainer.steps_per_epoch is None:
            trainer = TrainerConfig(**{**trainer.__dict__, "steps_per_epoch": task.steps_per_epoch})

    if isinstance(task, InverseDeconvolution1DTaskConfig):
        if trainer.steps_per_epoch is None:
            trainer = TrainerConfig(**{**trainer.__dict__, "steps_per_epoch": task.steps_per_epoch})

    return ExperimentConfig(run=run, seed=seed, trainer=trainer, optim=optim, task=task, model=model)

