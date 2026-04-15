"""Owns narrow public diagnostics for the Python-to-Mojo bridge."""

from __future__ import annotations

from dataclasses import dataclass

from .backend_info import backend_info
from .bundled_mojopkg_metadata import BundledMojopkgMetadata
from .layouts import MOJO_EXACT_CPU_BACKEND
from .mojo_exact_cpu import (
    _active_mojo_version,
    _bundled_mojopkg,
    _bundled_mojopkg_metadata,
    _detect_mojo_command,
    _mojo_source_root,
    load_module,
)


@dataclass(frozen=True, slots=True)
class MojoBridgeInfo:
    """Public diagnostics for the Python-to-Mojo exact-backend bridge."""

    available: bool
    availability_reason: str
    command: tuple[str, ...] | None
    active_mojo_version: str | None
    bridge_source: str
    bundled_project_version: str | None
    bundled_mojo_version: str | None
    module_loaded: bool | None
    load_error: str | None


def _bridge_source_kind() -> str:
    if _mojo_source_root() is not None:
        return "repo_sources"
    if _bundled_mojopkg() is not None:
        return "bundled_artifact"
    return "none"


def _command_or_none() -> tuple[str, ...] | None:
    try:
        return tuple(_detect_mojo_command())
    except RuntimeError:
        return None


def _metadata_or_none() -> BundledMojopkgMetadata | None:
    return _bundled_mojopkg_metadata()


def mojo_bridge_info(*, probe_load: bool = False) -> MojoBridgeInfo:
    """Return explicit public diagnostics for the Mojo exact backend bridge.

    Reason:
    - `backend_info(...)` answers backend capability questions
    - users still need one stable public place to inspect Python-to-Mojo
      connectivity, bundled wheel metadata, and optional bridge loadability
      without importing internal modules

    `probe_load=True` intentionally goes further than capability introspection:
    it attempts to build or load the Mojo-backed Python extension and reports
    whether that succeeded.
    """

    info = backend_info(MOJO_EXACT_CPU_BACKEND)
    metadata = _metadata_or_none()

    module_loaded: bool | None = None
    load_error: str | None = None
    if probe_load:
        if not info.available:
            module_loaded = False
            load_error = info.availability_reason
        else:
            try:
                load_module()
            except Exception as error:  # pragma: no cover - exercised in tests
                module_loaded = False
                load_error = str(error)
            else:
                module_loaded = True

    return MojoBridgeInfo(
        available=info.available,
        availability_reason=info.availability_reason,
        command=_command_or_none(),
        active_mojo_version=_active_mojo_version(),
        bridge_source=_bridge_source_kind(),
        bundled_project_version=(
            None if metadata is None else metadata.project_version
        ),
        bundled_mojo_version=None if metadata is None else metadata.mojo_version,
        module_loaded=module_loaded,
        load_error=load_error,
    )
