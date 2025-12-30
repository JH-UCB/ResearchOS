"""
Differential operators via PyTorch autograd.

This module is one of the main "SciML primitives" in the template.
It provides reusable building blocks for PDE residuals in 1D/2D/3D, including:
- gradient
- Jacobian
- divergence
- Laplacian
- Hessian

Notes on performance:
- These operators are correct but may be slow for high-dimensional outputs.
- For large-scale SciML, consider vector-Jacobian products (vJPs),
  structured derivatives, or specialized libraries (functorch, torch.func).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return x[:, None]
    return x


def grad_scalar(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Gradient of scalar field u(x) w.r.t. x.

    Args:
        u: Tensor of shape (N, 1) or (N,)
        x: Tensor of shape (N, d), requires_grad=True
        create_graph: Whether to create higher-order graph.

    Returns:
        du/dx of shape (N, d)
    """
    x = _ensure_2d(x)
    u = _ensure_2d(u)

    if u.shape[1] != 1:
        raise ValueError("grad_scalar expects u to be scalar per sample (N,1).")
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True.")

    g = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=False,
    )[0]
    return g


def jacobian(y: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Jacobian dy/dx for vector-valued y.

    Args:
        y: Tensor (N, m) or (N,) where m is output dimension.
        x: Tensor (N, d), requires_grad=True

    Returns:
        J of shape (N, m, d)
    """
    x = _ensure_2d(x)
    y = _ensure_2d(y)

    N, m = y.shape
    d = x.shape[1]
    if not x.requires_grad:
        raise ValueError("x must have requires_grad=True.")

    J = []
    for j in range(m):
        yj = y[:, j:j+1]
        gj = torch.autograd.grad(
            outputs=yj,
            inputs=x,
            grad_outputs=torch.ones_like(yj),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]  # (N, d)
        J.append(gj[:, None, :])  # (N, 1, d)

    return torch.cat(J, dim=1)  # (N, m, d)


def hessian(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Hessian of scalar field u(x).

    Args:
        u: Tensor (N,1) or (N,)
        x: Tensor (N,d), requires_grad=True

    Returns:
        H of shape (N, d, d)
    """
    x = _ensure_2d(x)
    u = _ensure_2d(u)
    if u.shape[1] != 1:
        raise ValueError("hessian expects scalar u per sample.")

    g = grad_scalar(u, x, create_graph=True)  # (N, d)
    N, d = g.shape
    H_rows = []
    for i in range(d):
        gi = g[:, i:i+1]
        Hi = torch.autograd.grad(
            outputs=gi,
            inputs=x,
            grad_outputs=torch.ones_like(gi),
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=False,
        )[0]  # (N, d)
        H_rows.append(Hi[:, None, :])  # (N, 1, d)
    return torch.cat(H_rows, dim=1)  # (N, d, d)


def divergence(v: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Divergence of a vector field v(x).

    Args:
        v: Tensor (N, d)
        x: Tensor (N, d), requires_grad=True

    Returns:
        div v of shape (N, 1)
    """
    x = _ensure_2d(x)
    v = _ensure_2d(v)
    if v.shape[1] != x.shape[1]:
        raise ValueError("divergence expects v and x to have same second dimension (d).")

    J = jacobian(v, x, create_graph=create_graph)  # (N, d, d)
    div = torch.diagonal(J, dim1=1, dim2=2).sum(dim=1, keepdim=True)  # (N,1)
    return div


def laplacian(u: torch.Tensor, x: torch.Tensor, create_graph: bool = True) -> torch.Tensor:
    """Laplacian Δu for scalar u(x)."""
    H = hessian(u, x, create_graph=create_graph)  # (N, d, d)
    lap = torch.diagonal(H, dim1=1, dim2=2).sum(dim=1, keepdim=True)
    return lap

