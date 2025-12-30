"""Classical (non-learned) numerical solvers used as baselines."""

from .poisson1d_fd import solve_poisson_dirichlet_1d

__all__ = ["solve_poisson_dirichlet_1d"]

