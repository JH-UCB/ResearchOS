"""
Reproducibility utilities: seeding Python, NumPy, and PyTorch.

SciML runs can be sensitive to initialization; this module centralizes
seed control so experiment replay is reliable.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedConfig:
    """Seed configuration.

    Attributes:
        seed: The base seed (int).
        deterministic_torch: If True, forces deterministic PyTorch ops where possible.
            This can reduce performance and may raise errors for some ops.
    """
    seed: int = 0
    deterministic_torch: bool = False


def set_global_seed(cfg: SeedConfig) -> None:
    """Set seeds for Python, NumPy, and PyTorch (if installed)."""
    seed = int(cfg.seed)

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Hash seed affects iteration order and some randomized structures.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if cfg.deterministic_torch:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

