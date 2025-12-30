"""
Fourier-feature MLP for coordinate-based learning.

This is a standard baseline for PINNs and implicit neural representations:
x -> [sin(2πBx), cos(2πBx)] -> MLP -> u

Where B is a random Gaussian matrix scaled by `fourier_scale`.
"""

from __future__ import annotations

from typing import List, Optional

import math
import torch
import torch.nn as nn

from ..config import FourierMLPModelConfig
from ..registry import register_model


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class FourierFeatures(nn.Module):
    """Random Fourier feature embedding."""
    def __init__(self, in_dim: int, num_features: int, scale: float):
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be > 0")
        if num_features <= 0:
            raise ValueError("num_features must be > 0")

        B = torch.randn(in_dim, num_features) * float(scale)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_dim)
        proj = 2.0 * math.pi * x @ self.B  # (N, num_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*num_features)


class FourierFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str,
        dropout: float,
        use_batchnorm: bool,
        fourier_features: int,
        fourier_scale: float,
    ):
        super().__init__()
        self.embed = FourierFeatures(input_dim, int(fourier_features), float(fourier_scale))
        ff_dim = 2 * int(fourier_features)

        layers: list[nn.Module] = []
        prev = ff_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, int(h)))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(int(h)))
            layers.append(_make_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)

        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        return self.net(z)


@register_model("fourier_mlp")
def build_fourier_mlp(cfg: FourierMLPModelConfig, device: torch.device, input_dim: Optional[int], output_dim: Optional[int]) -> nn.Module:
    if input_dim is None or output_dim is None:
        raise ValueError("fourier_mlp requires input_dim and output_dim to be provided by the Task.")
    model = FourierFeatureMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=list(cfg.hidden_layers),
        activation=str(cfg.activation),
        dropout=float(cfg.dropout),
        use_batchnorm=bool(cfg.use_batchnorm),
        fourier_features=int(cfg.fourier_features),
        fourier_scale=float(cfg.fourier_scale),
    )
    return model.to(device)

