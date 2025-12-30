"""
Sweep runner: execute a directory of YAML configs sequentially.

This is a minimal replacement for more complex experiment management tools.
It is often enough for graduate research:
- you keep configs version-controlled
- you run sweeps on a workstation/HPC job array
- results are tracked by run_id hashes

Usage:
    python -m sciml_pipeline.sweep --config_dir configs
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .run import main as run_main


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a sweep of sciml_pipeline configs")
    p.add_argument("--config_dir", type=str, required=True, help="Directory containing YAML configs")
    p.add_argument("--pattern", type=str, default="*.yaml", help="Glob pattern (default: *.yaml)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"config_dir not found: {config_dir}")

    configs = sorted(config_dir.glob(args.pattern))
    if not configs:
        print(f"No configs matching {args.pattern} in {config_dir}")
        return 0

    any_fail = False
    for path in configs:
        if path.name.startswith("_"):
            # convention: ignore internal scratch configs
            continue
        print(f"\n=== RUN {path} ===")
        code = run_main(["--config", str(path)])
        if code != 0:
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

