"""Owns backend selection policy for the high-level text retriever surface.

The public low-level search functions keep backend choice explicit.
The high-level retriever is the Mojo-first workflow entry point, so it prefers
the Mojo backend automatically when that backend is actually available.
"""

from __future__ import annotations

from kayak_bridge import (
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    backend_info,
)


def default_text_retriever_backend() -> str:
    if backend_info(MOJO_EXACT_CPU_BACKEND).available:
        return MOJO_EXACT_CPU_BACKEND
    return NUMPY_REFERENCE_BACKEND
