"""
Registries for tasks and models.

A registry is the simplest scalable plugin system:
- new tasks/models can be added without modifying the trainer/runner
- configuration selects implementations by name
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, TypeVar

import torch
import torch.nn as nn

# -----------------------------
# Protocols (interfaces)
# -----------------------------

Batch = Dict[str, torch.Tensor]


class Task(Protocol):
    """A problem definition (data + loss + metrics)."""

    name: str

    def build(self) -> None:
        """Optional one-time setup (load files, generate observations, etc.)."""
        ...

    def get_dataloaders(self) -> Dict[str, Any]:
        """Return dict with keys: 'train', optional 'val', optional 'test'.

        Each value can be:
        - torch.utils.data.DataLoader
        - any iterable yielding batches: Dict[str, Tensor]

        The trainer treats these as iterables; it does not assume random access.
        """
        ...

    def loss_and_metrics(self, model: nn.Module, batch: Batch, stage: str) -> tuple[torch.Tensor, Dict[str, float]]:
        """Compute scalar loss and a dictionary of scalar metrics for logging."""
        ...

    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        """Optional: compute expensive / global metrics for the current model.

        For SciML tasks, evaluation often means sampling a grid and comparing to
        known solution or checking conservation errors. This method is called
        at epoch boundaries.
        """
        ...

    def primary_metric(self) -> tuple[str, str]:
        """Return (metric_name, mode) where mode is 'min' or 'max'."""
        ...


class ModelFactory(Protocol):
    def __call__(self, input_dim: Optional[int], output_dim: Optional[int], device: torch.device, cfg: Any) -> nn.Module: ...


# -----------------------------
# Global registries
# -----------------------------

_TASK_REGISTRY: Dict[str, Callable[[Any], Task]] = {}
_MODEL_REGISTRY: Dict[str, Callable[[Any, torch.device, Optional[int], Optional[int]], nn.Module]] = {}


def register_task(name: str) -> Callable[[Callable[[Any], Task]], Callable[[Any], Task]]:
    """Decorator to register a Task builder."""
    def decorator(fn: Callable[[Any], Task]) -> Callable[[Any], Task]:
        if name in _TASK_REGISTRY:
            raise KeyError(f"Task '{name}' already registered")
        _TASK_REGISTRY[name] = fn
        return fn
    return decorator


def register_model(name: str) -> Callable[[Callable[[Any, torch.device, Optional[int], Optional[int]], nn.Module]], Callable[[Any, torch.device, Optional[int], Optional[int]], nn.Module]]:
    """Decorator to register a Model builder."""
    def decorator(fn: Callable[[Any, torch.device, Optional[int], Optional[int]], nn.Module]) -> Callable[[Any, torch.device, Optional[int], Optional[int]], nn.Module]:
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def make_task(task_cfg: Any) -> Task:
    """Instantiate a registered task from a config object."""
    name = getattr(task_cfg, "name", None)
    if not isinstance(name, str):
        raise ValueError("Task config must have a string attribute 'name'")
    if name not in _TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Registered: {sorted(_TASK_REGISTRY.keys())}")
    return _TASK_REGISTRY[name](task_cfg)


def make_model(model_cfg: Any, device: torch.device, input_dim: Optional[int], output_dim: Optional[int]) -> nn.Module:
    """Instantiate a registered model from a config object."""
    name = getattr(model_cfg, "name", None)
    if not isinstance(name, str):
        raise ValueError("Model config must have a string attribute 'name'")
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Registered: {sorted(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](model_cfg, device, input_dim, output_dim)


def registered_tasks() -> list[str]:
    return sorted(_TASK_REGISTRY.keys())


def registered_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())

