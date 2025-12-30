"""
Filesystem layout for runs and artifacts.

This is the smallest unit of "deployment architecture":
a stable on-disk contract that every task/model can rely on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class RunPaths:
    """Resolved paths for a run.

    Attributes:
        root_dir: Root directory containing all runs (e.g. ./runs).
        run_id: Stable run identifier (hash).
        run_dir: Directory for this run (root_dir/run_id).
        checkpoints_dir: Directory for model checkpoints.
        artifacts_dir: Directory for task/model artifacts (plots, arrays).
        logs_path: JSONL log file.
        summary_path: Summary JSON file.
        config_copy_path: Stored config YAML.
    """
    root_dir: Path
    run_id: str

    @property
    def run_dir(self) -> Path:
        return self.root_dir / self.run_id

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def config_copy_path(self) -> Path:
        return self.run_dir / "config.yaml"

    @property
    def best_ckpt_path(self) -> Path:
        return self.checkpoints_dir / "best.pt"

    @property
    def last_ckpt_path(self) -> Path:
        return self.checkpoints_dir / "last.pt"


def ensure_run_dirs(paths: RunPaths) -> None:
    """Create the directory structure for a run."""
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)

