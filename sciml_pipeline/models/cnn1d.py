"""
A small 1D CNN.

This can be used as a learned prior / denoiser / inverse map in 1D signal tasks.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..config import CNN1DModelConfig
from ..registry import register_model


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


class CNN1D(nn.Module):
    """A simple residual-ish 1D CNN mapping y -> x_hat."""

    def __init__(
        self,
        channels: List[int],
        kernel_size: int = 5,
        activation: str = "silu",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve length via symmetric padding")

        act = _make_activation(activation)
        padding = kernel_size // 2

        layers: list[nn.Module] = []
        in_ch = 1
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, int(ch), kernel_size=kernel_size, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(int(ch)))
            layers.append(_make_activation(activation))
            in_ch = int(ch)

        layers.append(nn.Conv1d(in_ch, 1, kernel_size=kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (batch, length) or (batch, 1, length)
        if y.ndim == 2:
            y = y[:, None, :]
        x = self.net(y)
        return x[:, 0, :]


@register_model("cnn1d")
def build_cnn1d(cfg: CNN1DModelConfig, device: torch.device, input_dim: Optional[int], output_dim: Optional[int]) -> nn.Module:
    # input_dim/output_dim are signal lengths; CNN itself doesn't need them, but we keep them for compatibility checks.
    model = CNN1D(
        channels=list(cfg.channels),
        kernel_size=int(cfg.kernel_size),
        activation=str(cfg.activation),
        use_batchnorm=bool(cfg.use_batchnorm),
    )
    return model.to(device)

