"""Owns the in-memory late-store implementation for tests and small workloads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .materialize import packed_index_from_matrices
from .metadata import (
    matches_metadata_filter,
    normalize_metadata_filter,
    normalize_metadata_rows,
)


@dataclass(frozen=True, slots=True)
class _StoredDocument:
    doc_id: str
    token_matrix: np.ndarray
    text: str | None
    metadata: dict[str, object] | None


class MemoryLateStore:
    """Stores late-interaction documents in process memory."""

    def __init__(self) -> None:
        self._records: dict[str, _StoredDocument] = {}
        self._order: list[str] = []

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="memory",
            persistent=False,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def stats(self) -> LateStoreStats:
        document_count = len(self._order)
        total_vector_count = sum(
            int(self._records[doc_id].token_matrix.shape[0]) for doc_id in self._order
        )
        vector_dim = None
        has_texts = False
        has_metadata = False
        if self._order:
            first = self._records[self._order[0]]
            vector_dim = int(first.token_matrix.shape[1])
            has_texts = any(
                self._records[doc_id].text is not None for doc_id in self._order
            )
            has_metadata = any(
                self._records[doc_id].metadata is not None for doc_id in self._order
            )
        return LateStoreStats(
            kind="memory",
            document_count=document_count,
            total_vector_count=total_vector_count,
            vector_dim=vector_dim,
            has_texts=has_texts,
            has_metadata=has_metadata,
            storage_byte_size=None,
        )

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        metadata_rows = normalize_metadata_rows(
            metadata,
            expected_length=documents.document_count,
        )
        for index, doc_id in enumerate(documents.doc_ids):
            if doc_id not in self._records:
                self._order.append(doc_id)
            self._records[doc_id] = _StoredDocument(
                doc_id=doc_id,
                token_matrix=documents.token_matrices[index],
                text=None if documents.texts is None else documents.texts[index],
                metadata=None if metadata_rows is None else metadata_rows[index],
            )

    def delete(self, doc_ids: object) -> None:
        for doc_id in to_doc_ids(doc_ids, "store delete doc_ids"):
            if doc_id in self._records:
                del self._records[doc_id]
                self._order.remove(doc_id)

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        selected = self._selected_records(doc_ids=doc_ids, where=where)
        if not selected:
            raise ValueError("store selection did not produce any documents")

        index = packed_index_from_matrices(
            tuple(record.doc_id for record in selected),
            tuple(record.token_matrix for record in selected),
            doc_texts=(
                tuple(record.text or "" for record in selected)
                if include_text and any(record.text is not None for record in selected)
                else None
            ),
        )
        return index.to_layout(layout)

    def _selected_records(
        self,
        *,
        doc_ids: object | None,
        where: object | None,
    ) -> tuple[_StoredDocument, ...]:
        where_filter = normalize_metadata_filter(where)
        if doc_ids is None:
            doc_order = tuple(self._order)
        else:
            doc_order = to_doc_ids(doc_ids, "store load doc_ids")

        selected: list[_StoredDocument] = []
        for doc_id in doc_order:
            record = self._records.get(doc_id)
            if record is None:
                continue
            if not matches_metadata_filter(record.metadata, where_filter):
                continue
            selected.append(record)
        return tuple(selected)
