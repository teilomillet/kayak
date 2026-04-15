"""Owns the public Chroma-backed late-store adapter for the Kayak SDK."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .client_types import ChromaClientLike
from .chromadb_imports import require_chromadb
from .external_payloads import (
    INTERNAL_METADATA_JSON_KEY,
    INTERNAL_TOKEN_MATRIX_JSON_KEY,
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
from .token_matrix_json import decode_token_matrix_json, encode_token_matrix_json


class ChromaLateStore:
    """Persist late-interaction documents in a Chroma collection.

    Parameters
    ----------
    path:
        Optional Chroma persistence directory.
    client:
        Optional existing Chroma client.
    collection_name:
        Collection used to store one document row plus the serialized token
        matrix payload.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        client: ChromaClientLike | None = None,
        collection_name: str = "late_documents",
    ) -> None:
        chromadb = require_chromadb()
        self._root = None if path is None else Path(path)
        self._collection_name = str(collection_name)
        self._client = (
            client
            if client is not None
            else chromadb.PersistentClient(path=str(self._root))
        )

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="chromadb",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "ChromaLateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def stats(self) -> LateStoreStats:
        collection = self._collection(create=False)
        if collection is None:
            return LateStoreStats(
                kind="chromadb",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=_directory_byte_size(self._root),
            )

        rows = self._get_rows(
            ids=None,
            where=None,
            include_text=True,
        )
        if not rows:
            return LateStoreStats(
                kind="chromadb",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=_directory_byte_size(self._root),
            )

        return LateStoreStats(
            kind="chromadb",
            document_count=len(rows),
            total_vector_count=sum(
                int(row["metadata"][INTERNAL_VECTOR_COUNT_KEY]) for row in rows
            ),
            vector_dim=int(rows[0]["metadata"][INTERNAL_VECTOR_DIM_KEY]),
            has_texts=any(row["text"] is not None for row in rows),
            has_metadata=any(
                row["metadata"].get(INTERNAL_METADATA_JSON_KEY) is not None
                for row in rows
            ),
            storage_byte_size=_directory_byte_size(self._root),
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
        collection = self._collection(create=True)

        ids: list[str] = []
        embeddings: list[list[float]] = []
        texts: list[str] = []
        metadatas: list[dict[str, object]] = []
        for index, doc_id in enumerate(documents.doc_ids):
            matrix = np.asarray(documents.token_matrices[index], dtype=np.float32)
            metadata_row = None if metadata_rows is None else metadata_rows[index]
            ids.append(doc_id)
            embeddings.append(matrix.mean(axis=0).tolist())
            texts.append("" if documents.texts is None else documents.texts[index])
            metadatas.append(
                {
                    INTERNAL_METADATA_JSON_KEY: (
                        None if metadata_row is None else _metadata_json(metadata_row)
                    ),
                    INTERNAL_TOKEN_MATRIX_JSON_KEY: encode_token_matrix_json(matrix),
                    INTERNAL_VECTOR_COUNT_KEY: int(matrix.shape[0]),
                    INTERNAL_VECTOR_DIM_KEY: int(matrix.shape[1]),
                    **filterable_metadata_items(metadata_row),
                }
            )

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def delete(self, doc_ids: object) -> None:
        collection = self._collection(create=False)
        if collection is None:
            return
        collection.delete(ids=list(to_doc_ids(doc_ids, "store delete doc_ids")))

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool = False,
        layout: str = "packed",
    ) -> LateIndex:
        requested_doc_ids = (
            None if doc_ids is None else to_doc_ids(doc_ids, "store load doc_ids")
        )
        where_filter = normalize_metadata_filter(where)
        rows = self._get_rows(
            ids=requested_doc_ids,
            where=where_filter,
            include_text=include_text,
        )

        records: list[ExternalStoredDocument] = []
        for row in rows:
            metadata = decode_metadata_json(row["metadata"].get(INTERNAL_METADATA_JSON_KEY))
            if not matches_metadata_filter(metadata, where_filter):
                continue
            records.append(
                ExternalStoredDocument(
                    doc_id=row["id"],
                    token_matrix=decode_token_matrix_json(
                        row["metadata"][INTERNAL_TOKEN_MATRIX_JSON_KEY]
                    ),
                    text=row["text"] if include_text else None,
                    metadata=metadata,
                )
            )

        if requested_doc_ids is not None:
            order = {doc_id: index for index, doc_id in enumerate(requested_doc_ids)}
            records.sort(key=lambda record: order[record.doc_id])
        if not records:
            raise ValueError("store selection did not produce any documents")

        return packed_index_from_records(
            tuple(records),
            include_text=include_text,
            layout=layout,
        )

    def _get_rows(
        self,
        *,
        ids: tuple[str, ...] | None,
        where: dict[str, object] | None,
        include_text: bool,
    ) -> list[dict[str, object]]:
        collection = self._collection(create=False)
        if collection is None:
            return []

        if ids is not None:
            result = collection.get(
                ids=list(ids),
                where=where if is_filter_pushdown_safe(where) else None,
                include=_chroma_include(include_text),
            )
            return _chroma_rows(result)

        rows: list[dict[str, object]] = []
        offset = 0
        while True:
            result = collection.get(
                where=where if is_filter_pushdown_safe(where) else None,
                include=_chroma_include(include_text),
                limit=256,
                offset=offset,
            )
            batch = _chroma_rows(result)
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
        return rows

    def _collection(self, *, create: bool) -> object | None:
        if create:
            return self._client.get_or_create_collection(name=self._collection_name)
        try:
            return self._client.get_collection(name=self._collection_name)
        except Exception:
            return None


def _metadata_json(metadata: dict[str, object]) -> str:
    from .external_payloads import encode_metadata_json

    payload = encode_metadata_json(metadata)
    assert payload is not None
    return payload


def _chroma_include(include_text: bool) -> list[str]:
    include = ["metadatas"]
    if include_text:
        include.append("documents")
    return include


def _chroma_rows(result: object) -> list[dict[str, object]]:
    ids = list(result["ids"])
    metadatas = list(result.get("metadatas") or [None] * len(ids))
    documents = list(result.get("documents") or [None] * len(ids))

    rows: list[dict[str, object]] = []
    for index, doc_id in enumerate(ids):
        metadata = metadatas[index]
        if metadata is None:
            metadata = {}
        rows.append(
            {
                "id": str(doc_id),
                "metadata": dict(metadata),
                "text": None if documents[index] is None else str(documents[index]),
            }
        )
    return rows


def _directory_byte_size(root: Path | None) -> int | None:
    if root is None or not root.exists():
        return None
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
