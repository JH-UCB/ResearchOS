"""
Finite difference solver for 1D Poisson with Dirichlet boundary conditions.

Solve:
    -u''(x) = f(x), x in (0, 1)
    u(0) = u0, u(1) = u1

We use a standard second-order central difference discretization on a uniform grid.

This is useful as:
- a sanity-check reference for PINNs
- a data generator for supervised operator learning
- a baseline forward solver for inverse problems

No SciPy dependency: we solve the tridiagonal system using NumPy.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def solve_poisson_dirichlet_1d(
    f: Callable[[np.ndarray], np.ndarray],
    n: int = 256,
    u0: float = 0.0,
    u1: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve -u'' = f on [0,1] with Dirichlet boundary conditions.

    Args:
        f: callable taking x array (n,) in [0,1] and returning f(x) array (n,)
        n: number of grid points (including boundaries)
        u0: boundary value at x=0
        u1: boundary value at x=1

    Returns:
        x: grid points shape (n,)
        u: solution values shape (n,)
    """
    n = int(n)
    if n < 3:
        raise ValueError("n must be >= 3")

    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]

    # Interior indices 1..n-2
    xi = x[1:-1]
    fi = f(xi).astype(np.float64)

    # Discretization:
    # -u''(x_i) ≈ -(u_{i-1} - 2u_i + u_{i+1}) / h^2 = f_i
    # => (-1/h^2) u_{i-1} + (2/h^2) u_i + (-1/h^2) u_{i+1} = f_i
    m = n - 2
    a = (-1.0 / h**2) * np.ones(m - 1)  # subdiagonal
    b = (2.0 / h**2) * np.ones(m)       # diagonal
    c = (-1.0 / h**2) * np.ones(m - 1)  # superdiagonal

    rhs = fi.copy()
    rhs[0] -= a[0] * u0
    rhs[-1] -= c[-1] * u1

    # Thomas algorithm (tridiagonal solve)
    cp = np.zeros_like(c)
    dp = np.zeros_like(rhs)

    cp[0] = c[0] / b[0]
    dp[0] = rhs[0] / b[0]

    for i in range(1, m - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (rhs[i] - a[i - 1] * dp[i - 1]) / denom

    denom_last = b[m - 1] - a[m - 2] * cp[m - 2]
    dp[m - 1] = (rhs[m - 1] - a[m - 2] * dp[m - 2]) / denom_last

    u_interior = np.zeros(m)
    u_interior[m - 1] = dp[m - 1]
    for i in range(m - 2, -1, -1):
        u_interior[i] = dp[i] - cp[i] * u_interior[i + 1]

    u = np.zeros(n)
    u[0] = u0
    u[-1] = u1
    u[1:-1] = u_interior
    return x, u

