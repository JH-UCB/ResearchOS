"""
General utilities used across tasks, models, and trainers.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def get_device(device: str = "auto") -> torch.device:
    """Resolve a torch.device.

    Args:
        device:
            - "auto": choose cuda if available, else mps if available, else cpu
            - "cpu" | "cuda" | "cuda:0" | "mps"
    """
    device = device.lower().strip()
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass (possibly nested) into a plain dict."""
    if not is_dataclass(obj):
        raise TypeError(f"Expected dataclass, got {type(obj)}")

    def convert(x: Any) -> Any:
        if is_dataclass(x):
            return {k: convert(v) for k, v in asdict(x).items()}
        if isinstance(x, dict):
            return {str(k): convert(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [convert(v) for v in x]
        return x

    return convert(obj)


def mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean((a - b) ** 2)


def grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Compute ∂outputs/∂inputs using autograd.

    Args:
        outputs: Tensor of shape (N, ...) that depends on inputs.
        inputs: Tensor with requires_grad=True.
        create_graph: If True, keep graph for higher-order derivatives.

    Returns:
        Gradient tensor with same shape as inputs.
    """
    if not inputs.requires_grad:
        raise ValueError("inputs must have requires_grad=True for autograd gradient computation")

    g = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return g


def second_derivative_1d(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute d²u/dx² for 1D coordinate networks.

    Args:
        u: Tensor of shape (N, 1) or (N,)
        x: Tensor of shape (N, 1) with requires_grad=True.

    Returns:
        Tensor of shape (N, 1) containing second derivatives.
    """
    if u.ndim == 1:
        u = u[:, None]
    du_dx = grad(u, x, create_graph=True)
    d2u_dx2 = grad(du_dx, x, create_graph=True)
    return d2u_dx2


def as_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach a torch tensor and convert to numpy on CPU."""
    return x.detach().cpu().numpy()

