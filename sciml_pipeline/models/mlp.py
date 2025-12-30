"""
MLP model.

Used for:
- supervised classification/regression on vector features
- coordinate networks in SciML (if used with low-dimensional inputs)
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

import torch
import torch.nn as nn

from ..config import MLPModelConfig
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


class MLP(nn.Module):
    """A configurable MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        layers: list[nn.Module] = []
        prev = input_dim
        act = _make_activation(activation)

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
        return self.net(x)


@register_model("mlp")
def build_mlp(cfg: MLPModelConfig, device: torch.device, input_dim: Optional[int], output_dim: Optional[int]) -> nn.Module:
    if input_dim is None or output_dim is None:
        raise ValueError("MLP requires input_dim and output_dim to be provided by the Task.")
    model = MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=list(cfg.hidden_layers),
        activation=str(cfg.activation),
        dropout=float(cfg.dropout),
        use_batchnorm=bool(cfg.use_batchnorm),
    )
    return model.to(device)

