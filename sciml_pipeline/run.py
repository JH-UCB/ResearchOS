"""
CLI runner: `python -m sciml_pipeline.run --config path/to/config.yaml`

This is the orchestrator: it composes
- config -> run_id + filesystem layout
- task + model registries
- trainer loop
- experiment logging

It is intentionally minimal and readable, as this is where most research
pipelines become brittle.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import build_experiment_config, load_yaml_config
from .hashing import config_hash
from .logging import CSVExperimentTracker, JSONLRunLogger, save_config_copy, save_summary
from .paths import RunPaths, ensure_run_dirs
from .seed import set_global_seed
from .trainers.torch_trainer import TorchTrainer
from .utils import get_device

# Side-effect imports: register tasks/models
from . import tasks as _tasks  # noqa: F401
from . import models as _models  # noqa: F401
from .registry import make_model, make_task


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sciml_pipeline runner")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    raw = load_yaml_config(args.config)
    cfg = build_experiment_config(raw)

    cfg_dict = cfg.to_dict()
    run_id = config_hash(cfg_dict, n_chars=12)

    root_dir = Path(cfg.run.root_dir)
    paths = RunPaths(root_dir=root_dir, run_id=run_id)

    # Trackers/loggers
    tracker = CSVExperimentTracker(root_dir=root_dir)
    logger = JSONLRunLogger(paths)

    # Resume semantics
    status = tracker.get_status(run_id)
    if status == "COMPLETED" and cfg.run.resume_if_completed:
        print(f"[SKIP] run_id={run_id} already COMPLETED (resume_if_completed=True).")
        return 0

    # Prepare run dirs
    if cfg.run.overwrite_run_dir and paths.run_dir.exists():
        import shutil
        shutil.rmtree(paths.run_dir)

    ensure_run_dirs(paths)
    save_config_copy(paths, cfg_dict)

    # Seed
    set_global_seed(cfg.seed)

    # Device
    device = get_device(cfg.trainer.device)

    # Instantiate task + model
    task = make_task(cfg.task)
    task.build()

    # Task may set input/output dims during build.
    input_dim = getattr(task, "input_dim", None)
    output_dim = getattr(task, "output_dim", None)

    model = make_model(cfg.model, device=device, input_dim=input_dim, output_dim=output_dim)

    # Start tracking
    tracker.start_run(
        run_id=run_id,
        experiment_name=cfg.run.experiment_name,
        task_name=getattr(task, "name", str(getattr(cfg.task, "name", ""))),
        model_name=str(getattr(cfg.model, "name", "")),
        notes=str(cfg.run.notes),
    )
    logger.log_event(
        "run_start",
        {
            "run_id": run_id,
            "experiment_name": cfg.run.experiment_name,
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": str(device),
        },
    )

    # Train
    try:
        trainer = TorchTrainer(cfg.trainer, cfg.optim)
        train_summary = trainer.fit(task=task, model=model, logger=logger, paths=paths)

        # Load best checkpoint if available
        if paths.best_ckpt_path.exists():
            ckpt = torch.load(paths.best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            logger.log_event("loaded_best_checkpoint", {"epoch": ckpt.get("epoch", None), "metric": ckpt.get("metric", None)})

        # Final evaluation
        eval_metrics = task.evaluate(model, device) or {}
        logger.log_event("final_eval", {"metrics": eval_metrics})

        # Optional finalize hook (save artifacts)
        if hasattr(task, "finalize"):
            task.finalize(paths.artifacts_dir, model, device)  # type: ignore[attr-defined]

        summary = {
            "run_id": run_id,
            "experiment_name": cfg.run.experiment_name,
            "task": cfg_dict["task"],
            "model": cfg_dict["model"],
            "train_summary": train_summary,
            "final_eval": eval_metrics,
        }
        save_summary(paths, summary)

        # Mark completed using primary metric
        primary_metric = float(train_summary["best_metric"])
        tracker.complete_run(run_id, primary_metric_name=str(train_summary["primary_metric"]), primary_metric_value=primary_metric)

        print(f"[DONE] run_id={run_id} best {train_summary['primary_metric']}={train_summary['best_metric']:.6g} at epoch {train_summary['best_epoch']}")
        return 0

    except BaseException as e:
        logger.log_exception(e)
        tracker.fail_run(run_id, e)
        print(f"[FAILED] run_id={run_id} error={type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

