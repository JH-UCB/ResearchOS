"""
Manifold primitives.

Many scientific problems have natural manifold structure:
- unit vectors (sphere S^{n-1})
- rotations (SO(3))
- SPD matrices (covariances, diffusion tensors)
- low-rank manifolds (model reduction)

This module provides a small, correct baseline implementation for the sphere.
It is designed to integrate with the trainer via the optional
`task.post_optimizer_step(model)` hook, where parameters can be projected.

For more advanced work, you can extend this pattern to SO(n), Stiefel, SPD, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch


class Manifold(Protocol):
    def proj(self, x: torch.Tensor) -> torch.Tensor: ...
    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...
    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...
    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class Sphere:
    """Unit sphere S^{n-1} embedded in R^n.

    Points satisfy ||x|| = 1 (per sample).
    Tangent vectors satisfy x · v = 0 (per sample).
    """

    eps: float = 1e-12

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        """Project to the sphere by normalization."""
        norm = torch.linalg.norm(x, dim=-1, keepdim=True)
        return x / (norm + self.eps)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Geodesic distance: arccos(clamp(x·y))."""
        x = self.proj(x)
        y = self.proj(y)
        dot = torch.sum(x * y, dim=-1).clamp(-1.0, 1.0)
        return torch.arccos(dot)

    def exp(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on the sphere.

        exp_x(v) = cos(||v||) x + sin(||v||) v / ||v||
        """
        x = self.proj(x)
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        v_unit = v / (v_norm + self.eps)
        return self.proj(torch.cos(v_norm) * x + torch.sin(v_norm) * v_unit)

    def log(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logarithmic map on the sphere."""
        x = self.proj(x)
        y = self.proj(y)

        dot = torch.sum(x * y, dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.arccos(dot)  # (..,1)

        # Project y onto tangent space at x:
        u = y - dot * x
        u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)
        u_unit = u / (u_norm + self.eps)

        return theta * u_unit


def project_parameter_(param: torch.nn.Parameter, manifold: Sphere) -> None:
    """In-place manifold projection for a parameter tensor."""
    with torch.no_grad():
        param.copy_(manifold.proj(param))

