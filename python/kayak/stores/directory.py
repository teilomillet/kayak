"""Owns the default directory-backed late store for the public Python SDK."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids
from kayak_bridge.dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE

from .base import LateStoreStats, StoreCapabilities
from .materialize import packed_index_from_parts
from .memory import MemoryLateStore
from .metadata import (
    matches_metadata_filter,
    normalize_metadata_filter,
)


_STORE_VERSION = 1
_MANIFEST_FILENAME = "manifest.json"
_DOC_IDS_FILENAME = "doc_ids.json"
_DOC_OFFSETS_FILENAME = "doc_offsets.npy"
_TOKEN_VECTORS_FILENAME = "token_vectors.npy"
_DOC_TEXTS_FILENAME = "doc_texts.json"
_METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class _DirectoryManifest:
    version: int
    document_count: int
    total_vector_count: int
    vector_dim: int
    has_texts: bool
    has_metadata: bool

    def to_json_ready(self) -> dict[str, object]:
        return {
            "version": self.version,
            "document_count": self.document_count,
            "total_vector_count": self.total_vector_count,
            "vector_dim": self.vector_dim,
            "has_texts": self.has_texts,
            "has_metadata": self.has_metadata,
        }


class DirectoryLateStore:
    """Persists late-interaction documents as one local packed snapshot."""

    def __init__(self, path: str | Path) -> None:
        self._root = Path(path)
        self._doc_ids: tuple[str, ...] = ()
        self._doc_offsets: np.ndarray | None = None
        self._doc_texts: tuple[str, ...] | None = None
        self._metadata_rows: tuple[dict[str, object], ...] | None = None
        self._doc_positions: dict[str, int] = {}
        self._vector_dim: int | None = None
        self._token_vectors_path = self._root / _TOKEN_VECTORS_FILENAME
        self._load_existing_snapshot()

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="directory",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def stats(self) -> LateStoreStats:
        return LateStoreStats(
            kind="directory",
            document_count=len(self._doc_ids),
            total_vector_count=(
                0 if self._doc_offsets is None else int(self._doc_offsets[-1])
            ),
            vector_dim=self._vector_dim,
            has_texts=self._doc_texts is not None,
            has_metadata=self._metadata_rows is not None,
            storage_byte_size=(
                _directory_byte_size(self._root) if self._root.exists() else 0
            ),
        )

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        memory = self._working_memory_store()
        memory.upsert(documents, metadata=metadata)
        self._write_snapshot(memory)
        self._load_existing_snapshot()

    def delete(self, doc_ids: object) -> None:
        memory = self._working_memory_store()
        memory.delete(doc_ids)
        self._write_snapshot(memory)
        self._load_existing_snapshot()

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        if not self._doc_ids:
            raise ValueError("store selection did not produce any documents")

        selected_positions = self._selected_positions(doc_ids=doc_ids, where=where)
        if not selected_positions:
            raise ValueError("store selection did not produce any documents")

        if self._is_full_store_selection(selected_positions):
            index = self._load_full_packed_index(include_text=include_text)
            return index.to_layout(layout)

        index = self._load_selected_packed_index(
            selected_positions,
            include_text=include_text,
        )
        return index.to_layout(layout)

    def _selected_positions(
        self,
        *,
        doc_ids: object | None,
        where: object | None,
    ) -> tuple[int, ...]:
        where_filter = normalize_metadata_filter(where)
        if doc_ids is None:
            candidate_positions = tuple(range(len(self._doc_ids)))
        else:
            selected_doc_ids = to_doc_ids(doc_ids, "store load doc_ids")
            candidate_positions = tuple(
                self._doc_positions[doc_id]
                for doc_id in selected_doc_ids
                if doc_id in self._doc_positions
            )

        selected: list[int] = []
        for position in candidate_positions:
            metadata = (
                None if self._metadata_rows is None else self._metadata_rows[position]
            )
            if not matches_metadata_filter(metadata, where_filter):
                continue
            selected.append(position)
        return tuple(selected)

    def _is_full_store_selection(self, positions: tuple[int, ...]) -> bool:
        return positions == tuple(range(len(self._doc_ids)))

    def _load_full_packed_index(self, *, include_text: bool) -> LateIndex:
        if self._doc_offsets is None:
            raise ValueError("directory store has no stored offsets")

        token_vectors = _load_token_vectors_memmap(self._token_vectors_path)
        token_vectors.setflags(write=False)

        return packed_index_from_parts(
            self._doc_ids,
            self._doc_offsets,
            token_vectors,
            doc_texts=self._doc_texts if include_text else None,
        )

    def _load_selected_packed_index(
        self,
        positions: tuple[int, ...],
        *,
        include_text: bool,
    ) -> LateIndex:
        token_vectors = _load_token_vectors_memmap(self._token_vectors_path)
        selected_doc_ids = tuple(self._doc_ids[position] for position in positions)
        selected_texts = (
            None
            if not include_text or self._doc_texts is None
            else tuple(self._doc_texts[position] for position in positions)
        )
        selected_offsets, selected_vectors = _materialize_selected_vectors(
            token_vectors,
            self._doc_offsets,
            positions,
        )
        return packed_index_from_parts(
            selected_doc_ids,
            selected_offsets,
            selected_vectors,
            doc_texts=selected_texts,
        )

    def _load_existing_snapshot(self) -> None:
        manifest_path = self._root / _MANIFEST_FILENAME
        if not manifest_path.exists():
            self._doc_ids = ()
            self._doc_offsets = None
            self._doc_texts = None
            self._metadata_rows = None
            self._doc_positions = {}
            self._vector_dim = None
            return

        manifest = _read_manifest(manifest_path)
        self._doc_ids = tuple(
            json.loads((self._root / _DOC_IDS_FILENAME).read_text(encoding="utf-8"))
        )
        self._doc_positions = {
            doc_id: index for index, doc_id in enumerate(self._doc_ids)
        }
        offsets = np.load(self._root / _DOC_OFFSETS_FILENAME)
        offsets = np.asarray(offsets, dtype=INDEX_OFFSET_DTYPE)
        offsets.setflags(write=False)
        self._doc_offsets = offsets
        self._vector_dim = manifest.vector_dim
        self._doc_texts = None
        if manifest.has_texts:
            self._doc_texts = tuple(
                json.loads(
                    (self._root / _DOC_TEXTS_FILENAME).read_text(encoding="utf-8")
                )
            )
        self._metadata_rows = None
        if manifest.has_metadata:
            self._metadata_rows = tuple(
                json.loads(
                    (self._root / _METADATA_FILENAME).read_text(encoding="utf-8")
                )
            )

    def _working_memory_store(self) -> MemoryLateStore:
        memory = MemoryLateStore()
        if not self._doc_ids or self._doc_offsets is None:
            return memory

        token_vectors = np.load(self._token_vectors_path, mmap_mode="r")
        late_documents = LateDocuments.from_inputs(
            self._doc_ids,
            [
                token_vectors[
                    int(self._doc_offsets[position]) : int(
                        self._doc_offsets[position + 1]
                    )
                ]
                for position in range(len(self._doc_ids))
            ],
            texts=self._doc_texts,
        )
        memory.upsert(late_documents, metadata=self._metadata_rows)
        return memory

    def _write_snapshot(self, memory: MemoryLateStore) -> None:
        stats = memory.stats()
        staging_root = self._root.with_name(f"{self._root.name}.tmp")
        if staging_root.exists():
            shutil.rmtree(staging_root)
        staging_root.mkdir(parents=True, exist_ok=True)

        if stats.document_count == 0:
            _replace_store_root(staging_root, self._root)
            return

        records = memory._selected_records(doc_ids=None, where=None)
        late_index = memory.load_index(include_text=True)
        metadata_rows = tuple(record.metadata for record in records)
        has_texts = late_index.doc_texts is not None
        has_metadata = any(row is not None for row in metadata_rows)
        manifest = _DirectoryManifest(
            version=_STORE_VERSION,
            document_count=late_index.document_count,
            total_vector_count=late_index.total_vector_count,
            vector_dim=late_index.vector_dim,
            has_texts=has_texts,
            has_metadata=has_metadata,
        )

        _write_manifest(staging_root / _MANIFEST_FILENAME, manifest)
        (staging_root / _DOC_IDS_FILENAME).write_text(
            json.dumps(list(late_index.doc_ids), indent=2),
            encoding="utf-8",
        )
        np.save(
            staging_root / _DOC_OFFSETS_FILENAME,
            np.asarray(late_index.doc_offsets, dtype=INDEX_OFFSET_DTYPE),
            allow_pickle=False,
        )
        np.save(
            staging_root / _TOKEN_VECTORS_FILENAME,
            np.asarray(late_index.as_packed_token_matrix(), dtype=VECTOR_DTYPE),
            allow_pickle=False,
        )
        if has_texts:
            assert late_index.doc_texts is not None
            (staging_root / _DOC_TEXTS_FILENAME).write_text(
                json.dumps(list(late_index.doc_texts), indent=2),
                encoding="utf-8",
            )
        if has_metadata:
            (staging_root / _METADATA_FILENAME).write_text(
                json.dumps(
                    list(metadata_rows),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

        _replace_store_root(staging_root, self._root)


def _write_manifest(path: Path, manifest: _DirectoryManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_json_ready(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _read_manifest(path: Path) -> _DirectoryManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _DirectoryManifest(
        version=int(payload["version"]),
        document_count=int(payload["document_count"]),
        total_vector_count=int(payload["total_vector_count"]),
        vector_dim=int(payload["vector_dim"]),
        has_texts=bool(payload["has_texts"]),
        has_metadata=bool(payload["has_metadata"]),
    )


def _replace_store_root(staging_root: Path, target_root: Path) -> None:
    backup_root = target_root.with_name(f"{target_root.name}.bak")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    if target_root.exists():
        target_root.replace(backup_root)
    staging_root.replace(target_root)
    shutil.rmtree(backup_root, ignore_errors=True)


def _directory_byte_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        path.stat().st_size for path in root.rglob("*") if path.is_file()
    )


def _load_token_vectors_memmap(path: Path) -> np.ndarray:
    token_vectors = np.load(path, mmap_mode="r")
    if token_vectors.dtype != np.dtype(VECTOR_DTYPE):
        raise ValueError("directory store token vectors must be float32")
    if token_vectors.ndim != 2:
        raise ValueError("directory store token vectors must be a 2D matrix")
    return token_vectors


def _materialize_selected_vectors(
    token_vectors: np.ndarray,
    doc_offsets: np.ndarray,
    positions: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    position_array = np.asarray(positions, dtype=np.int64)
    start_offsets = doc_offsets[position_array]
    stop_offsets = doc_offsets[position_array + 1]
    vector_counts = stop_offsets - start_offsets

    selected_offsets = np.empty(len(positions) + 1, dtype=doc_offsets.dtype)
    selected_offsets[0] = 0
    np.cumsum(
        vector_counts,
        dtype=doc_offsets.dtype,
        out=selected_offsets[1:],
    )

    selected_vectors = np.empty(
        (int(selected_offsets[-1]), int(token_vectors.shape[1])),
        dtype=VECTOR_DTYPE,
    )
    cursor = 0
    for start, stop in zip(start_offsets, stop_offsets, strict=True):
        start_index = int(start)
        stop_index = int(stop)
        next_cursor = cursor + (stop_index - start_index)
        selected_vectors[cursor:next_cursor] = token_vectors[start_index:stop_index]
        cursor = next_cursor

    selected_offsets.setflags(write=False)
    selected_vectors.setflags(write=False)
    return selected_offsets, selected_vectors
