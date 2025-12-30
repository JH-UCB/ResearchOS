"""
Experiment logging utilities.

This module provides two complementary logging streams:

1) A **global** CSV index: `runs/experiments.csv`
   - one row per run_id
   - status transitions: STARTED -> COMPLETED/FAILED
   - quick scan of key metrics

2) A **per-run** JSONL event log: `runs/<run_id>/events.jsonl`
   - append-only stream of structured events (epoch metrics, debug info)
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .paths import RunPaths


def _utc_timestamp() -> str:
    """UTC timestamp in ISO-like format (seconds resolution)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ExperimentRow:
    """A single row in experiments.csv."""
    run_id: str
    status: str  # STARTED | COMPLETED | FAILED
    started_at: str
    completed_at: str = ""
    failed_at: str = ""
    experiment_name: str = ""
    task_name: str = ""
    model_name: str = ""
    primary_metric_name: str = ""
    primary_metric_value: str = ""
    notes: str = ""


class CSVExperimentTracker:
    """A lightweight global CSV tracker.

    This mirrors the transactional semantics you already built:
    - start_run writes/updates a STARTED row
    - complete_run marks COMPLETED and stores key metric
    - fail_run marks FAILED and stores error summary
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root_dir / "experiments.csv"

        # Ensure header exists
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(ExperimentRow("","", "")).keys()))
                writer.writeheader()

    def _read_all(self) -> Dict[str, ExperimentRow]:
        rows: Dict[str, ExperimentRow] = {}
        if not self.csv_path.exists():
            return rows

        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                run_id = r.get("run_id", "")
                if run_id:
                    rows[run_id] = ExperimentRow(**r)  # type: ignore[arg-type]
        return rows

    def _write_all(self, rows: Dict[str, ExperimentRow]) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(ExperimentRow("","", "")).keys()))
            writer.writeheader()
            for run_id in sorted(rows.keys()):
                writer.writerow(asdict(rows[run_id]))

    def get_status(self, run_id: str) -> Optional[str]:
        rows = self._read_all()
        if run_id in rows:
            return rows[run_id].status
        return None

    def start_run(
        self,
        run_id: str,
        experiment_name: str,
        task_name: str,
        model_name: str,
        notes: str = "",
    ) -> None:
        rows = self._read_all()
        started_at = _utc_timestamp()

        row = rows.get(run_id)
        if row is None:
            row = ExperimentRow(
                run_id=run_id,
                status="STARTED",
                started_at=started_at,
                experiment_name=experiment_name,
                task_name=task_name,
                model_name=model_name,
                notes=notes,
            )
        else:
            # Update the row; keep started_at if already present.
            row.status = "STARTED"
            row.started_at = row.started_at or started_at
            row.experiment_name = experiment_name or row.experiment_name
            row.task_name = task_name or row.task_name
            row.model_name = model_name or row.model_name
            row.notes = notes or row.notes

        rows[run_id] = row
        self._write_all(rows)

    def complete_run(
        self,
        run_id: str,
        primary_metric_name: str,
        primary_metric_value: float,
    ) -> None:
        rows = self._read_all()
        row = rows.get(run_id)
        if row is None:
            row = ExperimentRow(run_id=run_id, status="COMPLETED", started_at=_utc_timestamp())

        row.status = "COMPLETED"
        row.completed_at = _utc_timestamp()
        row.primary_metric_name = primary_metric_name
        row.primary_metric_value = f"{primary_metric_value:.6g}"
        rows[run_id] = row
        self._write_all(rows)

    def fail_run(self, run_id: str, error: BaseException) -> None:
        rows = self._read_all()
        row = rows.get(run_id)
        if row is None:
            row = ExperimentRow(run_id=run_id, status="FAILED", started_at=_utc_timestamp())

        row.status = "FAILED"
        row.failed_at = _utc_timestamp()
        row.notes = f"{type(error).__name__}: {error}"
        rows[run_id] = row
        self._write_all(rows)


class JSONLRunLogger:
    """Append-only event logger for a single run."""
    def __init__(self, paths: RunPaths):
        self.paths = paths
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        event = {
            "ts": _utc_timestamp(),
            "event": event_type,
            **payload,
        }
        with open(self.paths.logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_epoch_metrics(self, epoch: int, stage: str, metrics: Dict[str, Any]) -> None:
        self.log_event("epoch_metrics", {"epoch": epoch, "stage": stage, "metrics": metrics})

    def log_message(self, message: str, **extra: Any) -> None:
        self.log_event("message", {"message": message, **extra})

    def log_exception(self, error: BaseException) -> None:
        tb = traceback.format_exc()
        self.log_event(
            "exception",
            {"error_type": type(error).__name__, "error": str(error), "traceback": tb},
        )


def save_config_copy(paths: RunPaths, config_dict: Dict[str, Any]) -> None:
    """Save the config as YAML inside the run directory."""
    with open(paths.config_copy_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)


def save_summary(paths: RunPaths, summary: Dict[str, Any]) -> None:
    """Save final summary as JSON."""
    with open(paths.summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

