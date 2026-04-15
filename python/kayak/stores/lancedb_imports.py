"""Owns optional-import helpers for the LanceDB public store adapter."""

from __future__ import annotations

from typing import Any


def require_lancedb() -> tuple[Any, Any]:
    """Import LanceDB and PyArrow with one user-facing failure mode."""

    try:
        import lancedb
        import pyarrow as pa
    except ImportError as exc:
        raise RuntimeError(
            "LanceDB store support requires optional dependencies: "
            "install `lancedb` and `pyarrow` in the same Python environment."
        ) from exc
    return lancedb, pa
