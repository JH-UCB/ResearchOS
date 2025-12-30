"""
ParamVector model.

In inverse problems, a powerful baseline is to optimize the unknown directly
(with regularization) rather than training a neural network.

This model stores a learnable vector `x` and returns it.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import ParamVectorModelConfig
from ..registry import register_model


class ParamVector(nn.Module):
    def __init__(self, length: int, init: str = "zeros", init_scale: float = 0.01):
        super().__init__()
        self.length = int(length)

        if init == "zeros":
            x0 = torch.zeros(self.length)
        elif init == "normal":
            x0 = torch.randn(self.length) * float(init_scale)
        else:
            raise ValueError(f"Unknown init: {init}")

        self.x = nn.Parameter(x0)

    def forward(self, batch_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Return shape (length,) if no batch_input provided.
        # If batch_input provided with shape (B, ...), expand to (B, length).
        if batch_input is None:
            return self.x
        if batch_input.ndim == 1:
            B = 1
        else:
            B = int(batch_input.shape[0])
        return self.x[None, :].expand(B, -1)


@register_model("param_vector")
def build_param_vector(cfg: ParamVectorModelConfig, device: torch.device, input_dim: Optional[int], output_dim: Optional[int]) -> nn.Module:
    model = ParamVector(length=int(cfg.length), init=str(cfg.init), init_scale=float(cfg.init_scale))
    return model.to(device)

