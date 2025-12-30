"""
Supervised classification task loading data from an NPZ file.

Expected NPZ keys:
- X_train: float32 array (N_train, D)
- y_train: int64 array (N_train,)
- X_val, y_val (optional)
- X_test, y_test (optional)

This task is included as a "sanity check" baseline: if the runner cannot
solve a simple supervised task, it's not ready for SciML.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import ClassificationNPZTaskConfig
from ..metrics import classification_metrics
from ..registry import register_task


class _DictBatchLoader:
    """Re-iterable wrapper converting (x, y) tuples to dict batches.
    
    This class creates a new generator on each __iter__ call,
    allowing the trainer to iterate over the same loader multiple epochs.
    """
    def __init__(self, loader: DataLoader):
        self.loader = loader
    
    def __iter__(self):
        for x, y in self.loader:
            yield {"x": x, "y": y}


class ClassificationNPZTask:
    name: str = "classification_npz"

    def __init__(self, cfg: ClassificationNPZTaskConfig):
        self.cfg = cfg
        self._built = False

        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        self._test_loader: Optional[DataLoader] = None

        self._criterion = nn.CrossEntropyLoss()

    def build(self) -> None:
        path = Path(self.cfg.npz_path)
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")

        data = np.load(path, allow_pickle=False)

        def require(key: str) -> np.ndarray:
            if key not in data:
                raise KeyError(f"Missing key '{key}' in NPZ: {path}")
            return data[key]

        X_train = require("X_train").astype(np.float32)
        y_train = require("y_train").astype(np.int64)

        X_val = data["X_val"].astype(np.float32) if "X_val" in data else None
        y_val = data["y_val"].astype(np.int64) if "y_val" in data else None

        X_test = data["X_test"].astype(np.float32) if "X_test" in data else None
        y_test = data["y_test"].astype(np.int64) if "y_test" in data else None

        if X_train.ndim != 2:
            raise ValueError("X_train must be a 2D array (N, D)")
        if y_train.ndim != 1:
            raise ValueError("y_train must be a 1D array (N,)")

        self.input_dim = int(X_train.shape[1])
        self.output_dim = int(self.cfg.num_classes)

        # Basic label sanity check
        if y_train.min() < 0 or y_train.max() >= self.output_dim:
            raise ValueError(f"y_train values must be in [0, {self.output_dim-1}]")

        # Build datasets/loaders
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        self._train_loader = DataLoader(
            train_ds,
            batch_size=int(self.cfg.batch_size),
            shuffle=True,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(self.cfg.pin_memory),
        )

        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
            self._val_loader = DataLoader(
                val_ds,
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=bool(self.cfg.pin_memory),
            )

        if X_test is not None and y_test is not None:
            test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
            self._test_loader = DataLoader(
                test_ds,
                batch_size=int(self.cfg.batch_size),
                shuffle=False,
                num_workers=int(self.cfg.num_workers),
                pin_memory=bool(self.cfg.pin_memory),
            )

        self._built = True

    def get_dataloaders(self) -> Dict[str, Any]:
        if not self._built:
            self.build()
        assert self._train_loader is not None
        loaders: Dict[str, Any] = {"train": self._wrap(self._train_loader)}
        if self._val_loader is not None:
            loaders["val"] = self._wrap(self._val_loader)
        if self._test_loader is not None:
            loaders["test"] = self._wrap(self._test_loader)
        return loaders

    def _wrap(self, loader: DataLoader):
        # Convert (x, y) tuples into dict batches expected by the trainer.
        # Returns a re-iterable object instead of a single-use generator
        # to allow multiple epochs to iterate over the same loader.
        return _DictBatchLoader(loader)

    def loss_and_metrics(self, model: nn.Module, batch: Dict[str, torch.Tensor], stage: str) -> tuple[torch.Tensor, Dict[str, float]]:
        x = batch["x"]
        y = batch["y"]
        logits = model(x)
        loss = self._criterion(logits, y)

        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == y).float().mean().item()

        return loss, {"accuracy": float(acc)}

    @torch.no_grad()
    def evaluate(self, model: nn.Module, device: torch.device) -> Dict[str, float]:
        # Evaluate on validation set if present; else on training set.
        if not self._built:
            self.build()

        loader = self._val_loader or self._train_loader
        assert loader is not None

        all_true = []
        all_pred = []
        total_loss = 0.0
        total_n = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = self._criterion(logits, y)
            total_loss += float(loss.item()) * int(x.shape[0])
            total_n += int(x.shape[0])

            preds = torch.argmax(logits, dim=-1)
            all_true.append(y.detach().cpu().numpy())
            all_pred.append(preds.detach().cpu().numpy())

        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        m = classification_metrics(y_true, y_pred, num_classes=int(self.output_dim or self.cfg.num_classes))
        m["loss"] = total_loss / max(total_n, 1)
        # Prefix to make it clear this is global evaluation
        return {f"eval_{k}": float(v) for k, v in m.items()}

    def primary_metric(self) -> tuple[str, str]:
        # Prefer evaluation accuracy if evaluate() runs.
        return ("eval_accuracy", "max")


@register_task("classification_npz")
def build_task(cfg: ClassificationNPZTaskConfig) -> ClassificationNPZTask:
    return ClassificationNPZTask(cfg)

