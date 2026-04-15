"""Owns explicit hosted prepared exact runtimes for one service process.

This module owns runtime lifecycle and registry policy for the HTTP server.
It does not own HTTP parsing or response formatting outside the runtime
summaries it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .prepared_exact_runtime import (
    PreparedExactSearchRuntime,
    PreparedExactSearchRuntimeConfig,
    prepare_exact_search_runtime,
)
from .prepared_exact_types import (
    prepared_exact_runtime_config_json,
    prepared_exact_runtime_stats_json,
)


class HostedPreparedExactRuntimeNotFoundError(KeyError):
    """Raised when a hosted prepared exact runtime handle is unknown."""


@dataclass(frozen=True, slots=True)
class HostedPreparedExactRuntimeKey:
    """Explicit runtime identity and policy key for hosted reuse."""

    service_root: Path
    collection_id: str
    tenant_id: str
    namespace_id: str
    snapshot_id: str
    load_text_corpus: bool
    config: PreparedExactSearchRuntimeConfig


@dataclass(slots=True)
class HostedPreparedExactRuntimeEntry:
    """One active hosted prepared exact runtime plus its registry handle."""

    runtime_id: str
    key: HostedPreparedExactRuntimeKey
    runtime: PreparedExactSearchRuntime


def _runtime_summary_payload(
    entry: HostedPreparedExactRuntimeEntry,
) -> dict[str, object]:
    return {
        "runtime_id": entry.runtime_id,
        "collection_id": entry.key.collection_id,
        "tenant_id": entry.key.tenant_id,
        "namespace_id": entry.key.namespace_id,
        "snapshot_id": entry.key.snapshot_id,
        "load_text_corpus": entry.key.load_text_corpus,
        "config": prepared_exact_runtime_config_json(entry.key.config),
        "stats": prepared_exact_runtime_stats_json(entry.runtime.stats()),
    }


class HostedPreparedExactRuntimeRegistry:
    """Registry of process-backed exact runtimes owned by one hosted server."""

    __slots__ = (
        "_service_root",
        "_entries_by_runtime_id",
        "_runtime_id_by_key",
        "_next_runtime_index",
    )

    def __init__(self, *, service_root: Path) -> None:
        self._service_root = service_root
        self._entries_by_runtime_id: dict[str, HostedPreparedExactRuntimeEntry] = {}
        self._runtime_id_by_key: dict[HostedPreparedExactRuntimeKey, str] = {}
        self._next_runtime_index = 1

    def active_runtime_count(self) -> int:
        return len(self._entries_by_runtime_id)

    def list_runtime_summaries(self) -> list[dict[str, object]]:
        return [
            _runtime_summary_payload(entry)
            for entry in self._entries_by_runtime_id.values()
        ]

    def prepare_runtime(
        self,
        *,
        collection_id: str,
        tenant_id: str,
        namespace_id: str,
        snapshot_id: str,
        load_text_corpus: bool,
        config: PreparedExactSearchRuntimeConfig,
    ) -> tuple[dict[str, object], bool]:
        key = HostedPreparedExactRuntimeKey(
            service_root=self._service_root,
            collection_id=collection_id,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            snapshot_id=snapshot_id,
            load_text_corpus=load_text_corpus,
            config=config,
        )
        existing_runtime_id = self._runtime_id_by_key.get(key)
        if existing_runtime_id is not None:
            return self.runtime_summary(existing_runtime_id), True

        runtime_id = f"prepared-exact-runtime-{self._next_runtime_index:04d}"
        self._next_runtime_index += 1
        runtime = prepare_exact_search_runtime(
            service_root=self._service_root,
            collection_id=collection_id,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            snapshot_id=snapshot_id,
            load_text_corpus=load_text_corpus,
            config=config,
        )
        entry = HostedPreparedExactRuntimeEntry(
            runtime_id=runtime_id,
            key=key,
            runtime=runtime,
        )
        self._entries_by_runtime_id[runtime_id] = entry
        self._runtime_id_by_key[key] = runtime_id
        return _runtime_summary_payload(entry), False

    def runtime_summary(self, runtime_id: str) -> dict[str, object]:
        return _runtime_summary_payload(self._require_entry(runtime_id))

    def search(self, runtime_id: str, payload: dict[str, object]) -> dict[str, object]:
        entry = self._require_entry(runtime_id)
        return entry.runtime.search(payload)

    def search_batch(
        self,
        runtime_id: str,
        requests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        entry = self._require_entry(runtime_id)
        return entry.runtime.search_batch(requests)

    def close_runtime(self, runtime_id: str) -> dict[str, object]:
        entry = self._pop_entry(runtime_id)
        entry.runtime.close()
        return _runtime_summary_payload(entry)

    def close_all(self) -> list[dict[str, object]]:
        closed_entries = list(self._entries_by_runtime_id.values())
        self._entries_by_runtime_id.clear()
        self._runtime_id_by_key.clear()
        for entry in closed_entries:
            entry.runtime.close()
        return [_runtime_summary_payload(entry) for entry in closed_entries]

    def invalidate_reclaimed_snapshots(
        self,
        *,
        collection_id: str,
        tenant_id: str,
        namespace_id: str,
        reclaimed_snapshot_ids: list[str],
    ) -> list[dict[str, object]]:
        reclaimed = set(reclaimed_snapshot_ids)
        invalidated_runtime_ids = [
            runtime_id
            for runtime_id, entry in self._entries_by_runtime_id.items()
            if entry.key.collection_id == collection_id
            and entry.key.tenant_id == tenant_id
            and entry.key.namespace_id == namespace_id
            and entry.key.snapshot_id in reclaimed
        ]
        invalidated: list[dict[str, object]] = []
        for runtime_id in invalidated_runtime_ids:
            invalidated.append(self.close_runtime(runtime_id))
        return invalidated

    def _require_entry(self, runtime_id: str) -> HostedPreparedExactRuntimeEntry:
        entry = self._entries_by_runtime_id.get(runtime_id)
        if entry is None:
            raise HostedPreparedExactRuntimeNotFoundError(
                f"hosted prepared exact runtime does not exist: {runtime_id}"
            )
        return entry

    def _pop_entry(self, runtime_id: str) -> HostedPreparedExactRuntimeEntry:
        entry = self._require_entry(runtime_id)
        del self._entries_by_runtime_id[runtime_id]
        del self._runtime_id_by_key[entry.key]
        return entry
