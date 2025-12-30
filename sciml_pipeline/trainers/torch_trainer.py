"""
Generic PyTorch trainer.

This trainer is intentionally *task-driven*:
- The task defines what a "batch" is (supervised data, collocation points, etc.)
- The task defines the loss and metrics.
- The trainer performs optimization and handles:
  - checkpointing
  - early stopping
  - ReduceLROnPlateau scheduling
  - logging

This design is robust for SciML where "data" may be procedural.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..config import OptimConfig, TrainerConfig
from ..logging import JSONLRunLogger
from ..paths import RunPaths
from ..registry import Task
from ..utils import get_device


def _is_better(value: float, best: float, mode: str) -> bool:
    if mode == "min":
        return value < best
    if mode == "max":
        return value > best
    raise ValueError("mode must be 'min' or 'max'")


def _init_best(mode: str) -> float:
    if mode == "min":
        return float("inf")
    if mode == "max":
        return -float("inf")
    raise ValueError("mode must be 'min' or 'max'")


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            raise TypeError(f"Batch values must be torch.Tensor; got {type(v)} for key '{k}'")
    return out


def _aggregate_scalar_dict(dicts: list[Dict[str, float]]) -> Dict[str, float]:
    if not dicts:
        return {}
    keys = sorted({k for d in dicts for k in d.keys()})
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d]
        agg[k] = float(sum(vals) / max(len(vals), 1))
    return agg


class TorchTrainer:
    """Train/evaluate a model for a given task."""

    def __init__(self, trainer_cfg: TrainerConfig, optim_cfg: OptimConfig):
        self.trainer_cfg = trainer_cfg
        self.optim_cfg = optim_cfg
        self.device = get_device(trainer_cfg.device)

        # AMP only on CUDA
        self.use_amp = bool(trainer_cfg.mixed_precision and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _build_optimizer(self, task: Task, model: nn.Module) -> Adam:
        # Optional: let task provide custom param groups (e.g., different lrs)
        if hasattr(task, "optimizer_param_groups"):
            param_groups = task.optimizer_param_groups(model)  # type: ignore[attr-defined]
            return Adam(param_groups, lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay, betas=self.optim_cfg.betas)

        return Adam(model.parameters(), lr=self.optim_cfg.lr, weight_decay=self.optim_cfg.weight_decay, betas=self.optim_cfg.betas)

    def _build_scheduler(self, optimizer: Adam) -> Optional[ReduceLROnPlateau]:
        if not self.optim_cfg.use_plateau_scheduler:
            return None
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.optim_cfg.plateau_factor,
            patience=self.optim_cfg.plateau_patience,
            min_lr=self.optim_cfg.plateau_min_lr,
            verbose=False,
        )

    def fit(self, task: Task, model: nn.Module, logger: JSONLRunLogger, paths: RunPaths) -> Dict[str, Any]:
        """Train the model and return a summary dict."""
        model = model.to(self.device)
        model.train()

        loaders = task.get_dataloaders()
        train_loader = loaders.get("train", None)
        val_loader = loaders.get("val", None)

        if train_loader is None:
            raise ValueError("Task did not provide a 'train' dataloader/iterator.")

        primary_name, primary_mode = task.primary_metric()
        best_metric = _init_best(primary_mode)
        best_epoch = -1
        epochs_without_improve = 0

        optimizer = self._build_optimizer(task, model)
        scheduler = self._build_scheduler(optimizer)

        # For plateau scheduler: we use validation loss if present, else training loss.
        plateau_key = "loss"

        start_time = time.time()
        logger.log_message(f"Training start on device={self.device}, amp={self.use_amp}")

        for epoch in range(1, self.trainer_cfg.max_epochs + 1):
            # -----------------------
            # Train epoch
            # -----------------------
            train_metrics = self._run_epoch(
                task=task,
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                stage="train",
                steps_limit=self.trainer_cfg.steps_per_epoch,
            )

            if epoch % self.trainer_cfg.log_every_epochs == 0:
                logger.log_epoch_metrics(epoch, "train", train_metrics)

            # -----------------------
            # Validation epoch
            # -----------------------
            val_metrics: Dict[str, float] = {}
            if val_loader is not None and (epoch % self.trainer_cfg.eval_every_epochs == 0):
                val_metrics = self._run_epoch(
                    task=task,
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    stage="val",
                    steps_limit=None,  # usually finite
                )
                logger.log_epoch_metrics(epoch, "val", val_metrics)

            # Optional global eval at epoch boundaries (PDE grid evaluation, etc.)
            global_eval: Dict[str, float] = {}
            if epoch % self.trainer_cfg.eval_every_epochs == 0:
                try:
                    global_eval = task.evaluate(model, self.device) or {}
                    if global_eval:
                        logger.log_event("global_eval", {"epoch": epoch, "metrics": global_eval})
                except NotImplementedError:
                    pass

            # -----------------------
            # Select metric for early stopping
            # -----------------------
            metric_source = val_metrics if val_metrics else train_metrics
            # Prefer global_eval if it provides the primary metric.
            if global_eval and primary_name in global_eval:
                metric_source = {**metric_source, **global_eval}

            if primary_name not in metric_source:
                raise KeyError(
                    f"Primary metric '{primary_name}' not found in metrics. "
                    f"Available keys: {sorted(metric_source.keys())}"
                )

            current_primary = float(metric_source[primary_name])

            # -----------------------
            # Scheduler step (if enabled)
            # -----------------------
            if scheduler is not None:
                # If val_metrics present, use val loss; else train loss.
                sched_metrics = val_metrics if val_metrics else train_metrics
                sched_value = float(sched_metrics.get(plateau_key, current_primary))
                scheduler.step(sched_value)

            # -----------------------
            # Checkpointing and early stopping
            # -----------------------
            improved = _is_better(current_primary, best_metric, primary_mode)
            if improved:
                best_metric = current_primary
                best_epoch = epoch
                epochs_without_improve = 0
                self._save_checkpoint(paths.best_ckpt_path, model, optimizer, epoch, best_metric, primary_name)
                logger.log_event("checkpoint_best", {"epoch": epoch, "metric": best_metric, "metric_name": primary_name})
            else:
                epochs_without_improve += 1

            # Always save last checkpoint
            self._save_checkpoint(paths.last_ckpt_path, model, optimizer, epoch, current_primary, primary_name)

            if epochs_without_improve >= self.trainer_cfg.early_stopping_patience:
                logger.log_message(
                    "Early stopping triggered",
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_metric=best_metric,
                    primary_metric=primary_name,
                    mode=primary_mode,
                )
                break

        wall = time.time() - start_time

        summary = {
            "best_epoch": best_epoch,
            "best_metric": float(best_metric),
            "primary_metric": primary_name,
            "primary_mode": primary_mode,
            "wall_time_sec": float(wall),
        }
        return summary

    def _run_epoch(
        self,
        task: Task,
        model: nn.Module,
        loader: Any,
        optimizer: Optional[Adam],
        stage: str,
        steps_limit: Optional[int],
    ) -> Dict[str, float]:
        """Run one epoch over a loader/iterator."""
        is_train = optimizer is not None
        model.train(is_train)

        step_metrics: list[Dict[str, float]] = []
        step = 0

        # For iterators without __len__, we must rely on steps_limit or consume fully.
        for batch in loader:
            step += 1
            if steps_limit is not None and step > steps_limit:
                break

            batch = _move_batch_to_device(batch, self.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, metrics = task.loss_and_metrics(model, batch, stage="train")

                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

                self.scaler.scale(loss).backward()

                # Optional gradient clipping
                if self.trainer_cfg.grad_clip_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.trainer_cfg.grad_clip_norm)

                self.scaler.step(optimizer)
                self.scaler.update()

                # Optional projection/hook after optimizer step (manifolds, constraints)
                if hasattr(task, "post_optimizer_step"):
                    task.post_optimizer_step(model)  # type: ignore[attr-defined]

            else:
                with torch.no_grad():
                    loss, metrics = task.loss_and_metrics(model, batch, stage=stage)

            # Ensure loss is recorded
            metrics = dict(metrics)
            metrics["loss"] = float(loss.detach().item())
            step_metrics.append(metrics)

        return _aggregate_scalar_dict(step_metrics)

    def _save_checkpoint(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Adam,
        epoch: int,
        metric: float,
        metric_name: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": int(epoch),
            "metric": float(metric),
            "metric_name": str(metric_name),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(payload, path)

