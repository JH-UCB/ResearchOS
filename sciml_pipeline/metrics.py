"""
Metrics utilities.

We avoid heavy dependencies (e.g., scikit-learn) to keep the template portable.
For classification, we implement common metrics directly from counts.

For SciML tasks, we include basic field error metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix (num_classes x num_classes)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Compute accuracy, macro-F1, weighted-F1, macro-precision, macro-recall.

    Args:
        y_true: shape (N,)
        y_pred: shape (N,)
        num_classes: number of classes

    Returns:
        Dict of scalar metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    support = cm.sum(axis=1).astype(np.float64)
    total = cm.sum().astype(np.float64)

    precision = np.array([_safe_div(tp[i], tp[i] + fp[i]) for i in range(num_classes)], dtype=np.float64)
    recall = np.array([_safe_div(tp[i], tp[i] + fn[i]) for i in range(num_classes)], dtype=np.float64)
    f1 = np.array([_safe_div(2 * precision[i] * recall[i], precision[i] + recall[i]) for i in range(num_classes)], dtype=np.float64)

    acc = float(tp.sum() / total) if total != 0 else 0.0

    macro_precision = float(precision.mean()) if num_classes > 0 else 0.0
    macro_recall = float(recall.mean()) if num_classes > 0 else 0.0
    macro_f1 = float(f1.mean()) if num_classes > 0 else 0.0

    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() != 0 else 0.0

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def l2_relative_error(u_pred: torch.Tensor, u_true: torch.Tensor, eps: float = 1e-12) -> float:
    """Relative L2 error: ||pred-true||_2 / (||true||_2 + eps)."""
    num = torch.linalg.norm(u_pred - u_true).item()
    den = torch.linalg.norm(u_true).item()
    return float(num / (den + eps))


def linf_error(u_pred: torch.Tensor, u_true: torch.Tensor) -> float:
    """L-infinity error: max |pred-true|."""
    return float(torch.max(torch.abs(u_pred - u_true)).item())

