"""
Sampling utilities for domains used in SciML.

Domain sampling is a first-class design choice in PINNs / collocation methods:
- uniform iid sampling is simple but can miss sharp regions
- low-discrepancy sequences (Sobol) provide more even coverage
- Latin hypercube can reduce variance in low dimensions

This module provides small, dependency-free sampling baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def uniform_sample(n: int, dim: int, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """Uniform iid sample in [low, high]^dim."""
    n = int(n)
    dim = int(dim)
    return (high - low) * torch.rand(n, dim) + low


def sobol_sample(
    n: int,
    dim: int,
    low: float = 0.0,
    high: float = 1.0,
    scramble: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Sobol low-discrepancy sample in [low, high]^dim."""
    n = int(n)
    dim = int(dim)
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=bool(scramble), seed=seed)
    x01 = engine.draw(n)
    return (high - low) * x01 + low


def latin_hypercube_sample(n: int, dim: int, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
    """Latin hypercube sampling in [low, high]^dim.

    This is a lightweight implementation:
    - stratify each dimension into n bins
    - randomly permute bin assignments per dimension
    """
    n = int(n)
    dim = int(dim)

    # Bin edges in [0,1]
    bins = torch.linspace(0.0, 1.0, n + 1)
    # Random offset within each bin
    u = torch.rand(n, dim)
    x = torch.empty(n, dim)

    for j in range(dim):
        perm = torch.randperm(n)
        x[:, j] = bins[:-1][perm] + (bins[1:] - bins[:-1])[perm] * u[:, j]

    return (high - low) * x + low


@dataclass(frozen=True)
class HypercubeDomain:
    """Axis-aligned hypercube domain [low, high]^dim."""
    dim: int
    low: float = 0.0
    high: float = 1.0

    def sample_interior(self, n: int, method: str = "sobol", seed: Optional[int] = None) -> torch.Tensor:
        method = method.lower()
        if method == "uniform":
            return uniform_sample(n, self.dim, self.low, self.high)
        if method == "sobol":
            return sobol_sample(n, self.dim, self.low, self.high, seed=seed)
        if method == "lhs":
            return latin_hypercube_sample(n, self.dim, self.low, self.high)
        raise ValueError(f"Unknown sampling method: {method}")

    def sample_boundary(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        """Sample boundary points by picking a random face per point."""
        n = int(n)
        x = sobol_sample(n, self.dim, self.low, self.high, seed=seed)
        # Choose a face: (axis, side)
        faces = torch.randint(low=0, high=2 * self.dim, size=(n,))
        for i in range(n):
            axis = int(faces[i] // 2)
            side = int(faces[i] % 2)
            x[i, axis] = self.low if side == 0 else self.high
        return x

