"""Canonical prepared exact-search runtime surface and compatibility aliases."""

from __future__ import annotations

from pathlib import Path

from .prepared_exact_process_runtime import PreparedExactProcessRuntime
from .prepared_exact_types import (
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchRuntimeStats,
    _runtime_config,
)


PreparedExactSearchRuntime = PreparedExactProcessRuntime


def prepare_exact_search_runtime(
    *,
    service_root: str | Path,
    collection_id: str,
    tenant_id: str,
    namespace_id: str,
    snapshot_id: str,
    load_text_corpus: bool = True,
    config: PreparedExactSearchRuntimeConfig | None = None,
) -> PreparedExactSearchRuntime:
    """Prepare one local exact-search runtime pinned to one hosted snapshot."""

    resolved_config = _runtime_config(config)
    if resolved_config.execution_backend != "process":
        raise ValueError(
            "prepared exact search runtime only verifies the 'process' backend "
            f"today; got {resolved_config.execution_backend!r}"
        )
    return PreparedExactSearchRuntime(
        service_root=service_root,
        collection_id=collection_id,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        snapshot_id=snapshot_id,
        load_text_corpus=load_text_corpus,
        config=resolved_config,
    )


# Compatibility aliases preserve the earlier scheduler naming while the runtime
# contract becomes the canonical surface.
PreparedExactSearchSchedulerConfig = PreparedExactSearchRuntimeConfig
PreparedExactSearchSchedulerStats = PreparedExactSearchRuntimeStats
PreparedExactSearchScheduler = PreparedExactSearchRuntime


def prepare_exact_search_scheduler(
    *,
    service_root: str | Path,
    collection_id: str,
    tenant_id: str,
    namespace_id: str,
    snapshot_id: str,
    load_text_corpus: bool = True,
    config: PreparedExactSearchRuntimeConfig | None = None,
) -> PreparedExactSearchRuntime:
    """Compatibility alias for the prepared exact-search runtime constructor."""

    return prepare_exact_search_runtime(
        service_root=service_root,
        collection_id=collection_id,
        tenant_id=tenant_id,
        namespace_id=namespace_id,
        snapshot_id=snapshot_id,
        load_text_corpus=load_text_corpus,
        config=config,
    )
