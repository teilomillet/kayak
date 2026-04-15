"""Owns explicit backend capability introspection for the public Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .layouts import (
    INDEX_LAYOUT_HYBRID_FLAT_DIM128,
    INDEX_LAYOUT_PACKED,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    QUERY_LAYOUT_FLAT_DIM128,
    QUERY_LAYOUT_NESTED,
)
from .mojo_exact_cpu import _detect_mojo_command


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Capability and availability facts for one public backend name."""

    name: str
    available: bool
    requires_mojo: bool
    query_layouts: tuple[str, ...]
    index_layouts: tuple[str, ...]
    availability_reason: str


@lru_cache(maxsize=1)
def _mojo_backend_availability_reason() -> str:
    try:
        command = _detect_mojo_command()
    except RuntimeError as error:
        return str(error)
    return f"Kayak can invoke Mojo via: {' '.join(command)}"


def backend_info(name: str) -> BackendInfo:
    """Return capability and availability metadata for one backend name."""
    query_layouts = (QUERY_LAYOUT_NESTED, QUERY_LAYOUT_FLAT_DIM128)
    index_layouts = (INDEX_LAYOUT_PACKED, INDEX_LAYOUT_HYBRID_FLAT_DIM128)

    if name == NUMPY_REFERENCE_BACKEND:
        return BackendInfo(
            name=name,
            available=True,
            requires_mojo=False,
            query_layouts=query_layouts,
            index_layouts=index_layouts,
            availability_reason="NumPy reference backend is always available in the package runtime.",
        )

    if name == MOJO_EXACT_CPU_BACKEND:
        reason = _mojo_backend_availability_reason()
        return BackendInfo(
            name=name,
            available=not reason.startswith("Kayak could not find"),
            requires_mojo=True,
            query_layouts=query_layouts,
            index_layouts=index_layouts,
            availability_reason=reason,
        )

    raise ValueError(f"unsupported backend: {name}")


def available_backends() -> tuple[str, ...]:
    """Return the backend names available in the current runtime."""
    names = [NUMPY_REFERENCE_BACKEND]
    if backend_info(MOJO_EXACT_CPU_BACKEND).available:
        names.append(MOJO_EXACT_CPU_BACKEND)
    return tuple(names)
