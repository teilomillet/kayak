"""Caches prepared packed-index objects for the Mojo exact backend.

The Python exact bridge is currently dominated by repeated packed-index
materialization. A small bounded cache is justified because:
- `LateIndex` instances are immutable at the Python layer
- the same index object is commonly reused across many query batches
- evicting old prepared objects keeps the cache measurable and bounded
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from .layouts import INDEX_LAYOUT_PACKED
from .mojo_payloads import index_payload
from .prepared_index_storage_artifact import (
    prepared_packed_index_artifact_root,
    remove_prepared_packed_index_artifact,
    write_prepared_packed_index_artifact,
)


PREPARED_PACKED_INDEX_CACHE_SIZE = 8


@dataclass(slots=True)
class PreparedPackedIndexEntry:
    index: "LateIndex"
    prepared_index: object
    artifact_root: str | None


_prepared_packed_index_cache: OrderedDict[
    tuple[int, int], PreparedPackedIndexEntry
] = OrderedDict()
_prepared_packed_index_cache_lock = RLock()


def clear_prepared_packed_index_cache(*, module: object | None = None) -> None:
    del module
    with _prepared_packed_index_cache_lock:
        for entry in _prepared_packed_index_cache.values():
            remove_prepared_packed_index_artifact(
                _artifact_root_path(entry.artifact_root)
            )
        _prepared_packed_index_cache.clear()


def _artifact_root_path(artifact_root: str | None) -> "Path | None":
    if artifact_root is None:
        return None

    return Path(artifact_root)


def _prepare_packed_index(
    index: "LateIndex", *, module: object
) -> tuple[object, str | None]:
    if hasattr(module, "prepare_packed_index_from_storage"):
        if index.vector_dim != 128:
            payload = index_payload(index)
            prepared_index = module.prepare_packed_index(
                payload.doc_ids,
                payload.doc_offsets,
                payload.packed_vectors,
            )
            return prepared_index, None

        artifact_root = prepared_packed_index_artifact_root(
            module=module,
            index=index,
        )
        write_prepared_packed_index_artifact(index, root=artifact_root)
        try:
            prepared_index = module.prepare_packed_index_from_storage(
                str(artifact_root),
                index.vector_dim,
            )
        except Exception:
            remove_prepared_packed_index_artifact(artifact_root)
            raise
        return prepared_index, str(artifact_root)

    payload = index_payload(index)
    prepared_index = module.prepare_packed_index(
        payload.doc_ids,
        payload.doc_offsets,
        payload.packed_vectors,
    )
    return prepared_index, None


def prepared_packed_index_object(index: "LateIndex", *, module: object) -> object:
    if index.layout != INDEX_LAYOUT_PACKED:
        raise ValueError("prepared packed-index objects require packed layout")

    key = (id(module), id(index))
    with _prepared_packed_index_cache_lock:
        cached = _prepared_packed_index_cache.get(key)
        if cached is not None and cached.index is index:
            _prepared_packed_index_cache.move_to_end(key)
            return cached.prepared_index

    prepared_index, artifact_root = _prepare_packed_index(index, module=module)

    with _prepared_packed_index_cache_lock:
        _prepared_packed_index_cache[key] = PreparedPackedIndexEntry(
            index=index,
            prepared_index=prepared_index,
            artifact_root=artifact_root,
        )
        _prepared_packed_index_cache.move_to_end(key)

        while len(_prepared_packed_index_cache) > PREPARED_PACKED_INDEX_CACHE_SIZE:
            _, evicted = _prepared_packed_index_cache.popitem(last=False)
            remove_prepared_packed_index_artifact(
                _artifact_root_path(evicted.artifact_root)
            )

    return prepared_index
