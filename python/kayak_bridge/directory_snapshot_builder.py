"""Owns one-pass packed directory snapshots for large corpus benchmarks.

This module owns:
- streamed writing of `DirectoryLateStore`-compatible packed snapshots
- explicit accounting for document count, vector count, and reserved capacity

This module does not own:
- dataset loading
- text encoding policy
- search or evaluation

Assumptions:
- all documents share one vector dimension
- callers make the maximum per-document vector budget explicit
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
from numpy.lib.format import open_memmap

from kayak.stores.directory import (
    _DOC_IDS_FILENAME,
    _DirectoryManifest,
    _DOC_OFFSETS_FILENAME,
    _TOKEN_VECTORS_FILENAME,
    _directory_byte_size,
    _replace_store_root,
    _write_manifest,
)

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE


@dataclass(frozen=True, slots=True)
class StreamedPackedSnapshotDocument:
    """One aligned document row for a streamed packed snapshot."""

    doc_id: str
    token_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class StreamedPackedSnapshotSummary:
    """Reports the measurable result of one streamed packed snapshot build."""

    document_count: int
    total_vector_count: int
    reserved_vector_capacity: int
    vector_dim: int
    storage_byte_size: int

    def to_json_ready(self) -> dict[str, object]:
        return {
            "document_count": self.document_count,
            "total_vector_count": self.total_vector_count,
            "reserved_vector_capacity": self.reserved_vector_capacity,
            "vector_dim": self.vector_dim,
            "storage_byte_size": self.storage_byte_size,
        }


def _staging_root(root: Path) -> Path:
    return root.with_name(f"{root.name}.tmp")


def _write_doc_ids(root: Path, doc_ids: list[str]) -> None:
    (root / _DOC_IDS_FILENAME).write_text(
        json.dumps(doc_ids, indent=2),
        encoding="utf-8",
    )


def _write_doc_offsets(root: Path, doc_offsets: list[int]) -> None:
    np.save(
        root / _DOC_OFFSETS_FILENAME,
        np.asarray(doc_offsets, dtype=INDEX_OFFSET_DTYPE),
        allow_pickle=False,
    )


def _normalized_token_matrix(
    matrix: np.ndarray,
    *,
    vector_dim: int,
    max_document_vector_count: int,
) -> np.ndarray:
    normalized = np.asarray(matrix, dtype=VECTOR_DTYPE)
    if normalized.ndim != 2:
        raise ValueError("streamed snapshot token matrices must be 2D")
    if int(normalized.shape[1]) != vector_dim:
        raise ValueError("streamed snapshot token matrices must share vector_dim")
    if int(normalized.shape[0]) > max_document_vector_count:
        raise ValueError(
            "streamed snapshot document exceeded the configured vector budget"
        )
    return normalized


def write_streamed_packed_directory_snapshot(
    root: Path,
    documents: Iterable[StreamedPackedSnapshotDocument],
    *,
    document_count: int,
    vector_dim: int,
    max_document_vector_count: int,
) -> StreamedPackedSnapshotSummary:
    if document_count <= 0:
        raise ValueError("streamed snapshot requires at least one document")
    if vector_dim <= 0:
        raise ValueError("streamed snapshot vector_dim must be positive")
    if max_document_vector_count <= 0:
        raise ValueError(
            "streamed snapshot max_document_vector_count must be positive"
        )

    staging_root = _staging_root(root)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    reserved_vector_capacity = document_count * max_document_vector_count
    token_vectors = open_memmap(
        staging_root / _TOKEN_VECTORS_FILENAME,
        mode="w+",
        dtype=VECTOR_DTYPE,
        shape=(reserved_vector_capacity, vector_dim),
    )

    doc_ids: list[str] = []
    doc_offsets = [0]
    written_documents = 0
    written_vectors = 0

    try:
        for document in documents:
            if written_documents >= document_count:
                raise ValueError(
                    "streamed snapshot received more documents than expected"
                )

            matrix = _normalized_token_matrix(
                document.token_matrix,
                vector_dim=vector_dim,
                max_document_vector_count=max_document_vector_count,
            )
            next_offset = written_vectors + int(matrix.shape[0])
            token_vectors[written_vectors:next_offset] = matrix
            written_vectors = next_offset
            written_documents += 1
            doc_ids.append(str(document.doc_id))
            doc_offsets.append(written_vectors)

        if written_documents != document_count:
            raise ValueError(
                "streamed snapshot received fewer documents than expected"
            )

        token_vectors.flush()
        _write_doc_ids(staging_root, doc_ids)
        _write_doc_offsets(staging_root, doc_offsets)
        _write_manifest(
            staging_root / "manifest.json",
            _DirectoryManifest(
                version=1,
                document_count=written_documents,
                total_vector_count=written_vectors,
                vector_dim=vector_dim,
                has_texts=False,
                has_metadata=False,
            ),
        )
        _replace_store_root(staging_root, root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    return StreamedPackedSnapshotSummary(
        document_count=written_documents,
        total_vector_count=written_vectors,
        reserved_vector_capacity=reserved_vector_capacity,
        vector_dim=vector_dim,
        storage_byte_size=_directory_byte_size(root),
    )
