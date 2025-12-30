"""
Run ID hashing and canonical serialization.

The key requirement for research engineering is that *runs are identifiable*
by their configuration. This module provides stable hashing so that:
- reruns can be detected
- artifacts can be cached per config
- experiment tracking is deterministic
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _normalize_for_json(obj: Any) -> Any:
    """Convert an object into a JSON-serializable structure.

    This function is conservative:
    - dataclasses: should be converted before calling this function
    - numpy scalars: converted to Python scalars
    - tuples: converted to lists

    Args:
        obj: Any object.

    Returns:
        A JSON-serializable object.
    """
    # Avoid importing numpy if not installed; but we depend on it anyway.
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if np is not None:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)

    if isinstance(obj, dict):
        return {str(k): _normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Fall back to string representation (keeps hashing stable but lossy).
    return str(obj)


def canonical_json(data: Dict[str, Any]) -> str:
    """Create a canonical JSON string (stable ordering, no whitespace)."""
    normalized = _normalize_for_json(data)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(data: Dict[str, Any], n_chars: int = 12) -> str:
    """Compute a stable short hash for a configuration dictionary.

    Args:
        data: Configuration dictionary.
        n_chars: Length of returned hash prefix.

    Returns:
        A hex string of length `n_chars`.
    """
    s = canonical_json(data).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return h[: int(n_chars)]

