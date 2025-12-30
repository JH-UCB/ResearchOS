"""
PINN task: 1D Poisson equation.

We solve:
    -u''(x) = f(x),   x in (0, 1)
    u(0) = u(1) = 0

Default forcing family:
    u*(x) = sin(pi x)
    f(x) = pi^2 sin(pi x)

This is a classic "hello world" for PDE-constrained learning because:
- second derivatives are required (tests higher-order autograd)
- boundary conditions introduce constraint handling
- evaluation is easy (known exact solution)
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..metrics import l2_relative_error, linf_error
from ..solvers import solve_poisson_dirichlet_1d
from ..registry import register_task
from ..utils import mse, second_derivative_1d
from ..config import PINNPoisson1DTaskConfig


def forcing_sin_pi(x: torch.Tensor) -> torch.Tensor:
    """f(x) = pi^2 sin(pi x)."""
    return (math.pi ** 2) * torch.sin(math.pi * x)


def exact_solution_sin_pi(x: torch.Tensor) -> torch.Tensor:
    """u*(x) = sin(pi x)."""
    return torch.sin(math.pi * x)


class _InfinitePoissonSampler:
    """Infinite iterator yielding collocation + boundary batches.

    Yields dict:
        x_interior: (n_interior, 1) requires_grad=True
        x_boundary: (n_boundary, 1) requires_grad=True (boundary points)
        u_boundary: (n_boundary, 1) (zeros)
    """

    def __init__(self, cfg: PINNPoisson1DTaskConfig):
        self.cfg = cfg

    def __iter__(self):
        while True:
            n_i = int(self.cfg.n_interior)
            n_b = int(self.cfg.n_boundary)

            # Interior: uniform samples in (0,1)
            x_i = torch.rand(n_i, 1)
            x_i.requires_grad_(True)

            # Boundary: half at 0, half at 1
            n0 = n_b // 2
            n1 = n_b - n0
            x_b0 = torch.zeros(n0, 1)
            x_b1 = torch.ones(n1, 1)
            x_b = torch.cat([x_b0, x_b1], dim=0)
            x_b.requires_grad_(True)

            u_b = torch.zeros(n_b, 1)

            yield {"x_interior": x_i, "x_boundary": x_b, "u_boundary": u_b}


class PINNPoisson1DTask:
    name: str = "pinn_poisson1d"

    def __init__(self, cfg: PINNPoisson1DTaskConfig):
        self.cfg = cfg
        self.input_dim = 1
        self.output_dim = 1

        # Choose forcing
        if cfg.forcing != "sin_pi":
            raise ValueError(f"Unknown forcing: {cfg.forcing}")

        self._f = forcing_sin_pi
        self._u_exact = exact_solution_sin_pi

        self._train_iter = _InfinitePoissonSampler(cfg)

        # Fixed eval grid on CPU (moved to device when evaluating)
        x = torch.linspace(0.0, 1.0, int(cfg.n_eval)).view(-1, 1)
        self._x_eval = x
        self._u_eval = self._u_exact(x)

        # Finite-difference baseline (classical numerical solver)
        # This is useful for validating the PDE setup independently of autograd.
        import numpy as np

        def f_np(x_np: np.ndarray) -> np.ndarray:
            return (math.pi ** 2) * np.sin(math.pi * x_np)

        x_fd, u_fd = solve_poisson_dirichlet_1d(f_np, n=int(cfg.n_eval), u0=0.0, u1=0.0)
        self._u_fd = torch.from_numpy(u_fd.astype(np.float32)).view(-1, 1)

    def build(self) -> None:
        # Nothing to load; task is procedural.
        return

    def get_dataloaders(self) -> Dict[str, Any]:
        # Train is infinite; trainer must use steps_per_epoch.
        return {"train": iter(self._train_iter)}

    def loss_and_metrics(self, model: nn.Module, batch: Dict[str, torch.Tensor], stage: str) -> tuple[torch.Tensor, Dict[str, float]]:
        x_i = batch["x_interior"]
        x_b = batch["x_boundary"]
        u_b = batch["u_boundary"]

        # Ensure requires_grad is set after device transfer for higher-order derivatives
        if not x_i.requires_grad:
            x_i = x_i.detach().requires_grad_(True)

        # Model output: u(x) must be (N,1)
        u_i = model(x_i)
        u_b_pred = model(x_b)

        if u_i.ndim == 1:
            u_i = u_i[:, None]
        if u_b_pred.ndim == 1:
            u_b_pred = u_b_pred[:, None]

        if u_i.shape[1] != 1 or u_b_pred.shape[1] != 1:
            raise ValueError("PINNPoisson1DTask expects model output_dim == 1")

        # PDE residual: -u''(x) - f(x) = 0
        u_xx = second_derivative_1d(u_i, x_i)  # (N,1)
        f = self._f(x_i)  # (N,1)
        if f.ndim == 1:
            f = f[:, None]

        residual = -u_xx - f
        loss_pde = torch.mean(residual ** 2)

        # Boundary condition loss
        loss_bc = mse(u_b_pred, u_b)

        loss = float(self.cfg.w_pde) * loss_pde + float(self.cfg.w_bc) * loss_bc

        # Lightweight metrics (per-batch)
        metrics = {
            "loss_pde": float(loss_pde.detach().item()),
            "loss_bc": float(loss_bc.detach().item()),
        }
        return loss, metrics

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        # Evaluate on fixed grid
        x = self._x_eval.to(device)
        u_true = self._u_eval.to(device)

        u_pred = model(x)
        if u_pred.ndim == 1:
            u_pred = u_pred[:, None]

        rel_l2 = l2_relative_error(u_pred, u_true)
        linf = linf_error(u_pred, u_true)

        # Compare to FD baseline as well (should match exact for this case)
        u_fd = self._u_fd.to(device)
        rel_l2_fd = l2_relative_error(u_pred, u_fd)

        return {
            "eval_rel_l2": float(rel_l2),
            "eval_linf": float(linf),
            "eval_rel_l2_fd": float(rel_l2_fd),
        }

    def primary_metric(self) -> tuple[str, str]:
        # In PDEs we minimize error; relative L2 is a good baseline.
        return ("eval_rel_l2", "min")


@register_task("pinn_poisson1d")
def build_task(cfg: PINNPoisson1DTaskConfig) -> PINNPoisson1DTask:
    return PINNPoisson1DTask(cfg)

