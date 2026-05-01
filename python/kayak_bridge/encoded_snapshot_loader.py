"""Loads sharded encoded snapshots for benchmark and index-build entrypoints.

This module owns snapshot-to-array adapters. It deliberately keeps full-corpus
streaming separate from the current in-memory TAC reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, TOKEN_ID_DTYPE, VECTOR_DTYPE


SNAPSHOT_VECTOR_DTYPES = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}
SNAPSHOT_TOKEN_ID_DTYPES = {
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
}


@dataclass(frozen=True, slots=True)
class PackedSnapshotDocuments:
    doc_ids: tuple[str, ...]
    doc_offsets: np.ndarray
    token_vectors: np.ndarray
    token_ids: np.ndarray
    source_vector_dtype: str
    source_token_id_dtype: str
    loaded_payload_bytes: int
    loaded_document_count: int
    loaded_vector_count: int
    loaded_shard_count: int


@dataclass(frozen=True, slots=True)
class SnapshotQueries:
    query_ids: tuple[str, ...]
    texts: tuple[str, ...]
    relevant_doc_ids: tuple[tuple[str, ...], ...]
    query_matrices: tuple[np.ndarray, ...]
    query_vector_count: int
    vector_dim: int
    primary_metric: str
    k: int


def load_snapshot_manifest(snapshot_root: Path) -> dict[str, Any]:
    manifest_path = snapshot_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_packed_snapshot_documents(
    snapshot_root: Path,
    *,
    document_limit: int | None = None,
    max_vector_count: int | None = None,
    compute_dtype: np.dtype = VECTOR_DTYPE,
) -> PackedSnapshotDocuments:
    if document_limit is not None and document_limit <= 0:
        raise ValueError("document_limit must be positive when provided")
    if max_vector_count is not None and max_vector_count <= 0:
        raise ValueError("max_vector_count must be positive when provided")

    manifest = load_snapshot_manifest(snapshot_root)
    vector_dim = int(manifest["vector_dim"])
    source_vector_dtype = str(manifest["vector_dtype"])
    source_token_id_dtype = str(manifest["token_id_dtype"])
    vector_dtype = _require_dtype(
        source_vector_dtype,
        SNAPSHOT_VECTOR_DTYPES,
        "snapshot vector_dtype",
    )
    token_id_dtype = _require_dtype(
        source_token_id_dtype,
        SNAPSHOT_TOKEN_ID_DTYPES,
        "snapshot token_id_dtype",
    )
    shard_specs = _selected_shard_specs(
        manifest["shards"],
        document_limit=document_limit,
    )
    selected_document_count = sum(count for _shard, count in shard_specs)
    selected_vector_count = sum(
        _selected_vector_count(snapshot_root, shard, count)
        for shard, count in shard_specs
    )
    if max_vector_count is not None and selected_vector_count > max_vector_count:
        raise ValueError(
            "snapshot selection has "
            f"{selected_vector_count} vectors, above max_vector_count="
            f"{max_vector_count}; use a smaller selection or a streaming builder"
        )

    token_vectors = np.empty(
        (selected_vector_count, vector_dim),
        dtype=compute_dtype,
    )
    token_ids = np.empty(selected_vector_count, dtype=TOKEN_ID_DTYPE)
    doc_offsets = np.empty(selected_document_count + 1, dtype=INDEX_OFFSET_DTYPE)
    doc_ids: list[str] = []
    doc_offsets[0] = 0
    output_vector_offset = 0
    output_doc_offset = 0
    loaded_payload_bytes = 0

    for shard, selected_count in shard_specs:
        shard_dir = snapshot_root / "shards" / f"{int(shard['shard_index']):06d}"
        shard_doc_offsets = np.load(shard_dir / "doc_offsets.u64.npy").astype(
            np.uint64,
            copy=False,
        )
        start_vector = 0
        stop_vector = int(shard_doc_offsets[selected_count])
        vector_rows = _load_shard_vectors(
            shard_dir,
            vector_dtype=vector_dtype,
            vector_dim=vector_dim,
            start_vector=start_vector,
            stop_vector=stop_vector,
        )
        token_row = _load_shard_token_ids(
            shard_dir,
            token_id_dtype=token_id_dtype,
            start_vector=start_vector,
            stop_vector=stop_vector,
        )
        next_vector_offset = output_vector_offset + stop_vector
        token_vectors[output_vector_offset:next_vector_offset] = vector_rows.astype(
            compute_dtype,
            copy=False,
        )
        token_ids[output_vector_offset:next_vector_offset] = token_row.astype(
            TOKEN_ID_DTYPE,
            copy=False,
        )

        selected_offsets = shard_doc_offsets[: selected_count + 1].astype(
            INDEX_OFFSET_DTYPE,
            copy=False,
        )
        doc_offsets[
            output_doc_offset : output_doc_offset + selected_count + 1
        ] = selected_offsets + output_vector_offset
        if output_doc_offset > 0:
            # The first offset for each new shard duplicates the previous global
            # stop offset and is intentionally overwritten by later entries.
            doc_offsets[output_doc_offset] = output_vector_offset
        output_doc_offset += selected_count
        output_vector_offset = next_vector_offset
        doc_ids.extend(_load_doc_ids(shard_dir, selected_count))
        loaded_payload_bytes += _selected_payload_bytes(
            vector_count=stop_vector,
            document_count=selected_count,
            vector_dim=vector_dim,
            vector_dtype=vector_dtype,
            token_id_dtype=token_id_dtype,
        )

    return PackedSnapshotDocuments(
        doc_ids=tuple(doc_ids),
        doc_offsets=np.ascontiguousarray(doc_offsets, dtype=INDEX_OFFSET_DTYPE),
        token_vectors=np.ascontiguousarray(token_vectors, dtype=compute_dtype),
        token_ids=np.ascontiguousarray(token_ids, dtype=TOKEN_ID_DTYPE),
        source_vector_dtype=source_vector_dtype,
        source_token_id_dtype=source_token_id_dtype,
        loaded_payload_bytes=loaded_payload_bytes,
        loaded_document_count=selected_document_count,
        loaded_vector_count=selected_vector_count,
        loaded_shard_count=len(shard_specs),
    )


def iter_snapshot_document_shards(
    snapshot_root: Path,
) -> Iterable[tuple[dict[str, Any], np.memmap, np.memmap, np.ndarray, tuple[str, ...]]]:
    """Yield memory-mapped shard payloads for future streaming builders."""

    manifest = load_snapshot_manifest(snapshot_root)
    vector_dim = int(manifest["vector_dim"])
    vector_dtype = _require_dtype(
        str(manifest["vector_dtype"]),
        SNAPSHOT_VECTOR_DTYPES,
        "snapshot vector_dtype",
    )
    token_id_dtype = _require_dtype(
        str(manifest["token_id_dtype"]),
        SNAPSHOT_TOKEN_ID_DTYPES,
        "snapshot token_id_dtype",
    )
    for shard in manifest["shards"]:
        shard_dir = snapshot_root / "shards" / f"{int(shard['shard_index']):06d}"
        vector_count = int(shard["vector_count"])
        vectors = np.memmap(
            _shard_vector_path(shard_dir, vector_dtype),
            dtype=vector_dtype,
            mode="r",
            shape=(vector_count, vector_dim),
        )
        token_ids = np.memmap(
            _shard_token_id_path(shard_dir, token_id_dtype),
            dtype=token_id_dtype,
            mode="r",
            shape=(vector_count,),
        )
        offsets = np.load(shard_dir / "doc_offsets.u64.npy")
        yield shard, vectors, token_ids, offsets, tuple(_load_doc_ids(shard_dir, None))


def load_snapshot_queries(
    snapshot_root: Path,
    *,
    query_limit: int | None = None,
) -> SnapshotQueries:
    if query_limit is not None and query_limit <= 0:
        raise ValueError("query_limit must be positive when provided")
    query_root = snapshot_root / "queries"
    manifest_path = query_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"query manifest not found: {manifest_path}")
    query_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vector_dim = int(query_manifest["vector_dim"])
    offsets = np.load(query_root / "query_offsets.u64.npy")
    rows = _load_query_rows(query_root / "queries.jsonl")
    selected_count = len(rows) if query_limit is None else min(query_limit, len(rows))
    stop_vector = int(offsets[selected_count])
    raw_vectors = np.fromfile(query_root / "query_vectors.f32", dtype=np.float32)
    expected_values = int(offsets[-1]) * vector_dim
    if raw_vectors.shape[0] != expected_values:
        raise ValueError("query vector file does not match query offsets")
    matrix = raw_vectors[: stop_vector * vector_dim].reshape(stop_vector, vector_dim)
    query_matrices = []
    for query_index in range(selected_count):
        start = int(offsets[query_index])
        stop = int(offsets[query_index + 1])
        query_matrices.append(
            np.ascontiguousarray(matrix[start:stop], dtype=VECTOR_DTYPE)
        )
    return SnapshotQueries(
        query_ids=tuple(str(row["query_id"]) for row in rows[:selected_count]),
        texts=tuple(str(row["text"]) for row in rows[:selected_count]),
        relevant_doc_ids=tuple(
            tuple(str(doc_id) for doc_id in row["relevant_doc_ids"])
            for row in rows[:selected_count]
        ),
        query_matrices=tuple(query_matrices),
        query_vector_count=sum(int(row.shape[0]) for row in query_matrices),
        vector_dim=vector_dim,
        primary_metric=str(query_manifest.get("primary_metric", "mrr")),
        k=int(query_manifest.get("k", 10)),
    )


def snapshot_task_metadata(
    *,
    manifest: dict[str, Any],
    documents: PackedSnapshotDocuments,
    queries: SnapshotQueries,
) -> dict[str, Any]:
    return {
        "dataset_id": str(manifest["dataset_id"]),
        "model_name": str(manifest["model_name"]),
        "family": "msmarco",
        "slice_name": (
            f"snapshot_docs_{documents.loaded_document_count}_"
            f"queries_{len(queries.query_ids)}"
        ),
        "primary_metric": queries.primary_metric,
        "k": queries.k,
        "documents": [{"doc_id": doc_id} for doc_id in documents.doc_ids],
        "queries": [
            {
                "query_id": query_id,
                "text": text,
                "relevant_doc_ids": list(relevant_doc_ids),
            }
            for query_id, text, relevant_doc_ids in zip(
                queries.query_ids,
                queries.texts,
                queries.relevant_doc_ids,
                strict=True,
            )
        ],
    }


def _selected_shard_specs(
    shards: Sequence[dict[str, Any]],
    *,
    document_limit: int | None,
) -> list[tuple[dict[str, Any], int]]:
    remaining = document_limit
    selected: list[tuple[dict[str, Any], int]] = []
    for shard in shards:
        document_count = int(shard["document_count"])
        if remaining is None:
            selected.append((shard, document_count))
            continue
        if remaining <= 0:
            break
        take = min(remaining, document_count)
        if take > 0:
            selected.append((shard, take))
            remaining -= take
    return selected


def _selected_vector_count(
    snapshot_root: Path,
    shard: dict[str, Any],
    selected_document_count: int,
) -> int:
    if selected_document_count == int(shard["document_count"]):
        return int(shard["vector_count"])
    shard_dir = snapshot_root / "shards" / f"{int(shard['shard_index']):06d}"
    offsets = np.load(shard_dir / "doc_offsets.u64.npy")
    return int(offsets[selected_document_count])


def _load_shard_vectors(
    shard_dir: Path,
    *,
    vector_dtype: np.dtype,
    vector_dim: int,
    start_vector: int,
    stop_vector: int,
) -> np.ndarray:
    values = np.fromfile(_shard_vector_path(shard_dir, vector_dtype), dtype=vector_dtype)
    if values.shape[0] % vector_dim != 0:
        raise ValueError("snapshot vector file length is not divisible by vector_dim")
    matrix = values.reshape(values.shape[0] // vector_dim, vector_dim)
    return matrix[start_vector:stop_vector]


def _load_shard_token_ids(
    shard_dir: Path,
    *,
    token_id_dtype: np.dtype,
    start_vector: int,
    stop_vector: int,
) -> np.ndarray:
    token_ids = np.fromfile(
        _shard_token_id_path(shard_dir, token_id_dtype),
        dtype=token_id_dtype,
    )
    return token_ids[start_vector:stop_vector]


def _load_doc_ids(shard_dir: Path, selected_count: int | None) -> list[str]:
    rows = [
        line.rstrip("\n")
        for line in (shard_dir / "doc_ids.txt").read_text(encoding="utf-8").splitlines()
    ]
    if selected_count is None:
        return rows
    return rows[:selected_count]


def _load_query_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _selected_payload_bytes(
    *,
    vector_count: int,
    document_count: int,
    vector_dim: int,
    vector_dtype: np.dtype,
    token_id_dtype: np.dtype,
) -> int:
    return int(
        vector_count * vector_dim * vector_dtype.itemsize
        + vector_count * token_id_dtype.itemsize
        + (document_count + 1) * np.dtype(np.uint64).itemsize
    )


def _shard_vector_path(shard_dir: Path, dtype: np.dtype) -> Path:
    if dtype == np.dtype(np.float16):
        return shard_dir / "vectors.f16"
    if dtype == np.dtype(np.float32):
        return shard_dir / "vectors.f32"
    raise ValueError(f"unsupported vector dtype: {dtype}")


def _shard_token_id_path(shard_dir: Path, dtype: np.dtype) -> Path:
    if dtype == np.dtype(np.uint32):
        return shard_dir / "token_ids.u32"
    if dtype == np.dtype(np.int64):
        return shard_dir / "token_ids.i64"
    raise ValueError(f"unsupported token-id dtype: {dtype}")


def _require_dtype(
    dtype_name: str,
    allowed: dict[str, np.dtype],
    field_name: str,
) -> np.dtype:
    if dtype_name not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")
    return allowed[dtype_name]
