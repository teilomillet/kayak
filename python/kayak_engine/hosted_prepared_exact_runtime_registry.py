"""Owns explicit hosted prepared exact runtimes for one service process.

This module owns runtime lifecycle and registry policy for the HTTP server.
It does not own HTTP parsing or response formatting outside the runtime
summaries it returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading

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
    active_operation_count: int = 0
    closing: bool = False


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
        "_condition",
        "_entries_by_runtime_id",
        "_runtime_id_by_key",
        "_preparing_keys",
        "_next_runtime_index",
        "_closed",
    )

    def __init__(self, *, service_root: Path) -> None:
        self._service_root = service_root
        self._condition = threading.Condition()
        self._entries_by_runtime_id: dict[str, HostedPreparedExactRuntimeEntry] = {}
        self._runtime_id_by_key: dict[HostedPreparedExactRuntimeKey, str] = {}
        self._preparing_keys: set[HostedPreparedExactRuntimeKey] = set()
        self._next_runtime_index = 1
        self._closed = False

    def active_runtime_count(self) -> int:
        with self._condition:
            return len(self._entries_by_runtime_id)

    def list_runtime_summaries(self) -> list[dict[str, object]]:
        with self._condition:
            entries = list(self._entries_by_runtime_id.values())
        return [_runtime_summary_payload(entry) for entry in entries]

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

        while True:
            with self._condition:
                if self._closed:
                    raise RuntimeError("hosted prepared exact runtime registry is closed")

                existing_runtime_id = self._runtime_id_by_key.get(key)
                if existing_runtime_id is not None:
                    existing_entry = self._require_entry_locked(existing_runtime_id)
                    return _runtime_summary_payload(existing_entry), True

                if key not in self._preparing_keys:
                    self._preparing_keys.add(key)
                    break

                self._condition.wait()

        # Runtime preparation is intentionally outside the registry condition so
        # unrelated searches and summaries on active runtimes do not stall.
        runtime: PreparedExactSearchRuntime | None = None
        try:
            runtime = prepare_exact_search_runtime(
                service_root=self._service_root,
                collection_id=collection_id,
                tenant_id=tenant_id,
                namespace_id=namespace_id,
                snapshot_id=snapshot_id,
                load_text_corpus=load_text_corpus,
                config=config,
            )
            runtime.wait_until_ready(timeout=60.0)
        except Exception:
            with self._condition:
                self._preparing_keys.remove(key)
                self._condition.notify_all()
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:
                    pass
            raise

        with self._condition:
            self._preparing_keys.remove(key)
            if self._closed:
                self._condition.notify_all()
            else:
                runtime_id = f"prepared-exact-runtime-{self._next_runtime_index:04d}"
                self._next_runtime_index += 1
                entry = HostedPreparedExactRuntimeEntry(
                    runtime_id=runtime_id,
                    key=key,
                    runtime=runtime,
                )
                self._entries_by_runtime_id[runtime_id] = entry
                self._runtime_id_by_key[key] = runtime_id
                self._condition.notify_all()
                return _runtime_summary_payload(entry), False

        try:
            runtime.close()
        except Exception:
            pass
        raise RuntimeError("hosted prepared exact runtime registry is closed")

    def runtime_summary(self, runtime_id: str) -> dict[str, object]:
        entry = self._acquire_entry(runtime_id)
        try:
            return _runtime_summary_payload(entry)
        finally:
            self._release_entry(entry)

    def search(self, runtime_id: str, payload: dict[str, object]) -> dict[str, object]:
        entry = self._acquire_entry(runtime_id)
        try:
            return entry.runtime.search(payload)
        finally:
            self._release_entry(entry)

    def search_batch(
        self,
        runtime_id: str,
        requests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        entry = self._acquire_entry(runtime_id)
        try:
            return entry.runtime.search_batch(requests)
        finally:
            self._release_entry(entry)

    def close_runtime(self, runtime_id: str) -> dict[str, object]:
        with self._condition:
            entry = self._pop_entry_locked(runtime_id)
            entry.closing = True
            while entry.active_operation_count > 0:
                self._condition.wait()
        entry.runtime.close()
        return _runtime_summary_payload(entry)

    def close_all(self) -> list[dict[str, object]]:
        with self._condition:
            # Closing the registry prevents in-flight prepares from publishing
            # new runtimes after shutdown has already started.
            self._closed = True
            closed_entries = list(self._entries_by_runtime_id.values())
            self._entries_by_runtime_id.clear()
            self._runtime_id_by_key.clear()
            for entry in closed_entries:
                entry.closing = True
            self._condition.notify_all()
            while any(entry.active_operation_count > 0 for entry in closed_entries):
                self._condition.wait()
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
        with self._condition:
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

    def _acquire_entry(self, runtime_id: str) -> HostedPreparedExactRuntimeEntry:
        with self._condition:
            entry = self._require_entry_locked(runtime_id)
            if entry.closing:
                raise HostedPreparedExactRuntimeNotFoundError(
                    f"hosted prepared exact runtime does not exist: {runtime_id}"
                )
            entry.active_operation_count += 1
            return entry

    def _release_entry(self, entry: HostedPreparedExactRuntimeEntry) -> None:
        with self._condition:
            entry.active_operation_count -= 1
            if entry.active_operation_count < 0:
                raise RuntimeError("hosted prepared exact runtime operation count underflow")
            if entry.closing and entry.active_operation_count == 0:
                self._condition.notify_all()

    def _require_entry_locked(self, runtime_id: str) -> HostedPreparedExactRuntimeEntry:
        entry = self._entries_by_runtime_id.get(runtime_id)
        if entry is None:
            raise HostedPreparedExactRuntimeNotFoundError(
                f"hosted prepared exact runtime does not exist: {runtime_id}"
            )
        return entry

    def _pop_entry_locked(self, runtime_id: str) -> HostedPreparedExactRuntimeEntry:
        entry = self._require_entry_locked(runtime_id)
        del self._entries_by_runtime_id[runtime_id]
        del self._runtime_id_by_key[entry.key]
        return entry
