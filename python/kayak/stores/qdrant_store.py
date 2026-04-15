"""Owns the public Qdrant-backed late-store adapter for the Kayak SDK."""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .client_types import QdrantClientLike
from .external_payloads import (
    INTERNAL_DOC_ID_KEY,
    INTERNAL_METADATA_JSON_KEY,
    INTERNAL_TEXT_KEY,
    INTERNAL_VECTOR_COUNT_KEY,
    INTERNAL_VECTOR_DIM_KEY,
    decode_metadata_json,
    filterable_metadata_items,
    is_filter_pushdown_safe,
)
from .external_records import ExternalStoredDocument, packed_index_from_records
from .metadata import (
    matches_metadata_filter,
    normalize_metadata_filter,
    normalize_metadata_rows,
)
from .qdrant_imports import require_qdrant


_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kayak.qdrant.late-store")


class QdrantLateStore:
    """Persist late-interaction documents in a Qdrant collection.

    Parameters
    ----------
    path:
        Optional local Qdrant path for embedded persistence.
    client:
        Optional existing Qdrant client.
    collection_name:
        Collection used to store one multivector payload per document.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        client: QdrantClientLike | None = None,
        collection_name: str = "late_documents",
    ) -> None:
        QdrantClient, _ = require_qdrant()
        self._root = None if path is None else Path(path)
        self._collection_name = str(collection_name)
        self._client = (
            client if client is not None else QdrantClient(path=str(self._root))
        )

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="qdrant",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "QdrantLateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def stats(self) -> LateStoreStats:
        records = self._iterate_records(
            where=None,
            include_vectors=False,
        )
        if not records:
            return LateStoreStats(
                kind="qdrant",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=_directory_byte_size(self._root),
            )

        return LateStoreStats(
            kind="qdrant",
            document_count=len(records),
            total_vector_count=sum(
                int(record.payload[INTERNAL_VECTOR_COUNT_KEY]) for record in records
            ),
            vector_dim=int(records[0].payload[INTERNAL_VECTOR_DIM_KEY]),
            has_texts=any(record.payload.get(INTERNAL_TEXT_KEY) is not None for record in records),
            has_metadata=any(
                record.payload.get(INTERNAL_METADATA_JSON_KEY) is not None
                for record in records
            ),
            storage_byte_size=_directory_byte_size(self._root),
        )

    def upsert(
        self,
        documents: LateDocuments,
        *,
        metadata: object | None = None,
    ) -> None:
        _, models = require_qdrant()
        metadata_rows = normalize_metadata_rows(
            metadata,
            expected_length=documents.document_count,
        )
        self._ensure_collection(documents.vector_dim)

        points = []
        for index, doc_id in enumerate(documents.doc_ids):
            metadata_row = None if metadata_rows is None else metadata_rows[index]
            payload = {
                INTERNAL_DOC_ID_KEY: doc_id,
                INTERNAL_TEXT_KEY: None if documents.texts is None else documents.texts[index],
                INTERNAL_METADATA_JSON_KEY: (
                    None if metadata_row is None else _metadata_json(metadata_row)
                ),
                INTERNAL_VECTOR_COUNT_KEY: int(documents.token_matrices[index].shape[0]),
                INTERNAL_VECTOR_DIM_KEY: int(documents.vector_dim),
                **filterable_metadata_items(metadata_row),
            }
            points.append(
                models.PointStruct(
                    id=_point_id_for_doc_id(doc_id),
                    vector=documents.token_matrices[index].tolist(),
                    payload=payload,
                )
            )

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def delete(self, doc_ids: object) -> None:
        selected_doc_ids = to_doc_ids(doc_ids, "store delete doc_ids")
        if not selected_doc_ids or not self._collection_exists():
            return
        _, models = require_qdrant()
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(
                points=[_point_id_for_doc_id(doc_id) for doc_id in selected_doc_ids]
            ),
            wait=True,
        )

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        records = self._selected_records(doc_ids=doc_ids, where=where)
        if not records:
            raise ValueError("store selection did not produce any documents")
        return packed_index_from_records(
            records,
            include_text=include_text,
            layout=layout,
        )

    def _selected_records(
        self,
        *,
        doc_ids: object | None,
        where: object | None,
    ) -> tuple[ExternalStoredDocument, ...]:
        requested_doc_ids = (
            None if doc_ids is None else to_doc_ids(doc_ids, "store load doc_ids")
        )
        where_filter = normalize_metadata_filter(where)
        retrieved = (
            self._retrieve_by_ids(requested_doc_ids)
            if requested_doc_ids is not None
            else self._iterate_records(
                where=where_filter,
                include_vectors=True,
            )
        )

        decoded: list[ExternalStoredDocument] = []
        for record in retrieved:
            metadata = decode_metadata_json(
                record.payload.get(INTERNAL_METADATA_JSON_KEY)
            )
            if not matches_metadata_filter(metadata, where_filter):
                continue
            matrix = np.asarray(record.vector, dtype=np.float32)
            matrix.setflags(write=False)
            decoded.append(
                ExternalStoredDocument(
                    doc_id=str(record.payload.get(INTERNAL_DOC_ID_KEY, record.id)),
                    token_matrix=matrix,
                    text=_optional_string(record.payload.get(INTERNAL_TEXT_KEY)),
                    metadata=metadata,
                )
            )

        if requested_doc_ids is None:
            return tuple(decoded)
        order = {doc_id: index for index, doc_id in enumerate(requested_doc_ids)}
        decoded.sort(key=lambda record: order[record.doc_id])
        return tuple(decoded)

    def _retrieve_by_ids(self, doc_ids: tuple[str, ...]) -> list[object]:
        if not doc_ids or not self._collection_exists():
            return []
        return list(
            self._client.retrieve(
                collection_name=self._collection_name,
                ids=[_point_id_for_doc_id(doc_id) for doc_id in doc_ids],
                with_payload=True,
                with_vectors=True,
            )
        )

    def _iterate_records(
        self,
        *,
        where: dict[str, object] | None,
        include_vectors: bool,
    ) -> list[object]:
        if not self._collection_exists():
            return []

        _, models = require_qdrant()
        offset = None
        records: list[object] = []
        scroll_filter = None
        if where is not None and is_filter_pushdown_safe(where):
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=str(key),
                        match=models.MatchValue(value=value),
                    )
                    for key, value in where.items()
                ]
            )

        while True:
            page, offset = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=include_vectors,
            )
            records.extend(page)
            if offset is None:
                break
        return records

    def _ensure_collection(self, vector_dim: int) -> None:
        if self._collection_exists():
            return
        _, models = require_qdrant()
        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=models.VectorParams(
                size=int(vector_dim),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        )

    def _collection_exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection_name))


def _metadata_json(metadata: dict[str, object]) -> str:
    from .external_payloads import encode_metadata_json

    payload = encode_metadata_json(metadata)
    assert payload is not None
    return payload


def _point_id_for_doc_id(doc_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, doc_id))


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _directory_byte_size(root: Path | None) -> int | None:
    if root is None or not root.exists():
        return None
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
