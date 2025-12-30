"""
Inverse problem task: 1D deconvolution.

We generate a synthetic ground truth signal x_true and observe:
    y_obs = k * x_true + noise

We then recover x by minimizing:
    L = w_data * ||k * x_hat - y_obs||^2 + w_tv * TV(x_hat)

Where TV(x) = mean |x[i+1] - x[i]| (anisotropic 1D TV).

This task supports different models:
- param_vector: optimize x directly (classic inverse-problem baseline)
- cnn1d: learn y -> x map (learned prior baseline)
- mlp: learn y -> x map (vector baseline)

The forward operator (convolution) is implemented with torch.nn.functional.conv1d.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import InverseDeconvolution1DTaskConfig
from ..metrics import l2_relative_error, linf_error
from ..registry import register_task


def make_gaussian_kernel1d(sigma: float) -> torch.Tensor:
    """Create a normalized 1D Gaussian kernel tensor of shape (1,1,K)."""
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    # choose K ~ 6*sigma, ensure odd and at least 3
    K = max(3, int(round(6.0 * sigma)) | 1)  # bitwise-or 1 makes it odd
    half = K // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k = k / torch.sum(k)
    return k.view(1, 1, K)


def tv1d(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularizer: mean |x[i+1] - x[i]|."""
    return torch.mean(torch.abs(x[..., 1:] - x[..., :-1]))


class _InfiniteSingleObservationSampler:
    def __init__(self, y_obs: torch.Tensor):
        self.y_obs = y_obs

    def __iter__(self):
        while True:
            yield {"y_obs": self.y_obs}


class InverseDeconvolution1DTask:
    name: str = "inverse_deconvolution1d"

    def __init__(self, cfg: InverseDeconvolution1DTaskConfig):
        self.cfg = cfg
        self.signal_length = int(cfg.signal_length)

        self.input_dim = self.signal_length
        self.output_dim = self.signal_length

        self._built = False

        self._x_true: Optional[torch.Tensor] = None
        self._y_obs: Optional[torch.Tensor] = None
        self._kernel: Optional[torch.Tensor] = None

        self._train_iter: Optional[_InfiniteSingleObservationSampler] = None

    def build(self) -> None:
        # Synthetic ground truth: a sparse-ish mixture of bumps + sinusoids.
        L = self.signal_length
        t = torch.linspace(0.0, 1.0, L)

        x = (
            0.6 * torch.sin(2 * math.pi * 3.0 * t)
            + 0.3 * torch.sin(2 * math.pi * 7.0 * t + 0.3)
        )

        # Add localized bumps
        for center, amp, width in [(0.25, 1.0, 0.03), (0.63, -0.8, 0.05), (0.82, 0.5, 0.02)]:
            x = x + float(amp) * torch.exp(-((t - float(center)) ** 2) / (2.0 * float(width) ** 2))

        x = x.to(torch.float32)

        kernel = make_gaussian_kernel1d(float(self.cfg.kernel_sigma))
        y_clean = self._convolve(x[None, :], kernel)[0]
        noise = torch.randn_like(y_clean) * float(self.cfg.noise_std)
        y_obs = y_clean + noise

        self._x_true = x
        self._y_obs = y_obs
        self._kernel = kernel

        self._train_iter = _InfiniteSingleObservationSampler(y_obs[None, :])  # batch=1

        self._built = True

    def _convolve(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        # x: (B, L), kernel: (1,1,K)
        x1 = x[:, None, :]
        pad = int(kernel.shape[-1] // 2)
        y = F.conv1d(x1, kernel, padding=pad)
        return y[:, 0, :]  # (B, L)

    def get_dataloaders(self) -> Dict[str, Any]:
        if not self._built:
            self.build()
        assert self._train_iter is not None
        return {"train": iter(self._train_iter)}

    def _predict_x(self, model: nn.Module, y_obs_batch: torch.Tensor) -> torch.Tensor:
        # Try calling model with y_obs; supports CNN/MLP/ParamVector.
        x_hat = model(y_obs_batch)
        if x_hat.ndim == 1:
            x_hat = x_hat[None, :]
        return x_hat

    def loss_and_metrics(self, model: nn.Module, batch: Dict[str, torch.Tensor], stage: str) -> tuple[torch.Tensor, Dict[str, float]]:
        if not self._built:
            self.build()
        assert self._kernel is not None

        y_obs = batch["y_obs"]  # (B, L)
        x_hat = self._predict_x(model, y_obs)

        y_pred = self._convolve(x_hat, self._kernel.to(y_obs.device))
        loss_data = torch.mean((y_pred - y_obs) ** 2)

        loss_tv = tv1d(x_hat)

        loss = float(self.cfg.w_data) * loss_data + float(self.cfg.w_tv) * loss_tv

        return loss, {
            "loss_data": float(loss_data.detach().item()),
            "loss_tv": float(loss_tv.detach().item()),
        }

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        if not self._built:
            self.build()
        assert self._x_true is not None and self._y_obs is not None and self._kernel is not None

        x_true = self._x_true.to(device)[None, :]
        y_obs = self._y_obs.to(device)[None, :]

        x_hat = self._predict_x(model, y_obs)
        rel_l2 = l2_relative_error(x_hat, x_true)
        linf = linf_error(x_hat, x_true)

        y_pred = self._convolve(x_hat, self._kernel.to(device))
        data_mse = torch.mean((y_pred - y_obs) ** 2).item()

        return {
            "eval_rel_l2": float(rel_l2),
            "eval_linf": float(linf),
            "eval_data_mse": float(data_mse),
        }

    def finalize(self, artifacts_dir: Path, model: nn.Module, device: torch.device) -> None:
        """Optional: save artifacts (arrays and plots)."""
        if not self._built:
            self.build()
        assert self._x_true is not None and self._y_obs is not None and self._kernel is not None

        if not bool(self.cfg.save_artifacts):
            return

        artifacts_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            x_true = self._x_true.to(device)[None, :]
            y_obs = self._y_obs.to(device)[None, :]
            x_hat = self._predict_x(model, y_obs)

        np.save(artifacts_dir / "x_true.npy", x_true.squeeze(0).detach().cpu().numpy())
        np.save(artifacts_dir / "y_obs.npy", y_obs.squeeze(0).detach().cpu().numpy())
        np.save(artifacts_dir / "x_hat.npy", x_hat.squeeze(0).detach().cpu().numpy())
        np.save(artifacts_dir / "kernel.npy", self._kernel.squeeze().detach().cpu().numpy())

        # Optional plot if matplotlib is available
        if not bool(self.cfg.save_plots):
            return

        # Matplotlib plots are optional; can be slow on first import (font cache).
        try:
            import matplotlib.pyplot as plt  # type: ignore

            t = np.linspace(0.0, 1.0, self.signal_length)
            plt.figure(figsize=(10, 4))
            plt.plot(t, x_true.squeeze(0).detach().cpu().numpy(), label="x_true")
            plt.plot(t, x_hat.squeeze(0).detach().cpu().numpy(), label="x_hat")
            plt.legend()
            plt.title("Deconvolution: recovered signal")
            plt.tight_layout()
            plt.savefig(artifacts_dir / "reconstruction.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 4))
            plt.plot(t, y_obs.squeeze(0).detach().cpu().numpy(), label="y_obs")
            plt.legend()
            plt.title("Observation")
            plt.tight_layout()
            plt.savefig(artifacts_dir / "observation.png", dpi=150)
            plt.close()
        except Exception:
            # Matplotlib not installed or runtime backend issue: ignore.
            pass

    def primary_metric(self) -> tuple[str, str]:
        # In inverse problems, error is minimized.
        return ("eval_rel_l2", "min")


@register_task("inverse_deconvolution1d")
def build_task(cfg: InverseDeconvolution1DTaskConfig) -> InverseDeconvolution1DTask:
    return InverseDeconvolution1DTask(cfg)

