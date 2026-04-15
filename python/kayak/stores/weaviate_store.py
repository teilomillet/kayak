"""Owns the public Weaviate-backed late-store adapter for the Kayak SDK."""

from __future__ import annotations

from pathlib import Path
import uuid

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.array_conversions import to_doc_ids

from .base import LateStoreStats, StoreCapabilities
from .client_types import WeaviateClientLike
from .external_payloads import decode_metadata_json
from .external_records import ExternalStoredDocument, packed_index_from_records
from .metadata import (
    matches_metadata_filter,
    normalize_metadata_filter,
    normalize_metadata_rows,
)
from .weaviate_imports import require_weaviate


_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "kayak.weaviate.late-store")


class WeaviateLateStore:
    """Persist late-interaction documents in a Weaviate collection.

    Parameters
    ----------
    persistence_path:
        Optional embedded Weaviate persistence directory.
    client:
        Optional existing Weaviate client.
    collection_name:
        Collection used to store one object per document.
    vector_name:
        Name of the self-provided multivector field.
    environment_variables:
        Optional embedded Weaviate environment variables.
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
        *,
        client: WeaviateClientLike | None = None,
        collection_name: str = "LateDocument",
        vector_name: str = "colbert",
        environment_variables: dict[str, str] | None = None,
    ) -> None:
        weaviate, _ = require_weaviate()
        self._root = None if persistence_path is None else Path(persistence_path)
        self._collection_name = str(collection_name)
        self._vector_name = str(vector_name)
        self._client = (
            client
            if client is not None
            else weaviate.connect_to_embedded(
                persistence_data_path=str(self._root),
                environment_variables=environment_variables or {},
            )
        )

    def capabilities(self) -> StoreCapabilities:
        return StoreCapabilities(
            kind="weaviate",
            persistent=True,
            supports_metadata_filter=True,
            supports_document_subset_load=True,
            supported_layouts=("packed", "hybrid_flat_dim128"),
        )

    def close(self) -> None:
        close_fn = getattr(self._client, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "WeaviateLateStore":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def stats(self) -> LateStoreStats:
        records = self._iterate_records()
        if not records:
            return LateStoreStats(
                kind="weaviate",
                document_count=0,
                total_vector_count=0,
                vector_dim=None,
                has_texts=False,
                has_metadata=False,
                storage_byte_size=_directory_byte_size(self._root),
            )

        first_matrix = records[0].token_matrix
        return LateStoreStats(
            kind="weaviate",
            document_count=len(records),
            total_vector_count=sum(
                int(record.token_matrix.shape[0]) for record in records
            ),
            vector_dim=int(first_matrix.shape[1]),
            has_texts=any(record.text is not None for record in records),
            has_metadata=any(record.metadata is not None for record in records),
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

        for index, doc_id in enumerate(documents.doc_ids):
            object_uuid = _uuid_for_doc_id(doc_id)
            delete_by_id = getattr(collection.data, "delete_by_id", None)
            if callable(delete_by_id):
                try:
                    delete_by_id(object_uuid)
                except Exception:
                    pass

            metadata_row = None if metadata_rows is None else metadata_rows[index]
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "text": None if documents.texts is None else documents.texts[index],
                    "metadata_json": (
                        None if metadata_row is None else _metadata_json(metadata_row)
                    ),
                },
                vector={
                    self._vector_name: documents.token_matrices[index].tolist(),
                },
                uuid=object_uuid,
            )

    def delete(self, doc_ids: object) -> None:
        if not self._collection_exists():
            return
        collection = self._collection(create=False)
        assert collection is not None
        for doc_id in to_doc_ids(doc_ids, "store delete doc_ids"):
            collection.data.delete_by_id(_uuid_for_doc_id(doc_id))

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

        records = self._iterate_records()
        selected: list[ExternalStoredDocument] = []
        allowed_ids = None if requested_doc_ids is None else set(requested_doc_ids)
        for record in records:
            if allowed_ids is not None and record.doc_id not in allowed_ids:
                continue
            if not matches_metadata_filter(record.metadata, where_filter):
                continue
            selected.append(
                ExternalStoredDocument(
                    doc_id=record.doc_id,
                    token_matrix=record.token_matrix,
                    text=record.text if include_text else None,
                    metadata=record.metadata,
                )
            )

        if requested_doc_ids is not None:
            order = {doc_id: index for index, doc_id in enumerate(requested_doc_ids)}
            selected.sort(key=lambda record: order[record.doc_id])
        if not selected:
            raise ValueError("store selection did not produce any documents")

        return packed_index_from_records(
            tuple(selected),
            include_text=include_text,
            layout=layout,
        )

    def _iterate_records(self) -> tuple[ExternalStoredDocument, ...]:
        if not self._collection_exists():
            return ()

        collection = self._collection(create=False)
        assert collection is not None
        records: list[ExternalStoredDocument] = []
        iterator = collection.iterator(
            include_vector=True,
            return_properties=["doc_id", "text", "metadata_json"],
        )
        for obj in iterator:
            vector_group = obj.vector or {}
            matrix = np.asarray(vector_group[self._vector_name], dtype=np.float32)
            matrix.setflags(write=False)
            properties = obj.properties or {}
            records.append(
                ExternalStoredDocument(
                    doc_id=str(properties["doc_id"]),
                    token_matrix=matrix,
                    text=_optional_string(properties.get("text")),
                    metadata=decode_metadata_json(properties.get("metadata_json")),
                )
            )
        return tuple(records)

    def _collection(self, *, create: bool) -> object | None:
        if create and not self._collection_exists():
            self._create_collection()
        if not self._collection_exists():
            return None
        return self._client.collections.get(self._collection_name)

    def _collection_exists(self) -> bool:
        return bool(self._client.collections.exists(self._collection_name))

    def _create_collection(self) -> None:
        _, wvc = require_weaviate()
        self._client.collections.create(
            name=self._collection_name,
            properties=[
                wvc.config.Property(name="doc_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(
                    name="metadata_json",
                    data_type=wvc.config.DataType.TEXT,
                ),
            ],
            vector_config=wvc.config.Configure.MultiVectors.self_provided(
                name=self._vector_name,
            ),
        )


def _uuid_for_doc_id(doc_id: str) -> str:
    return str(uuid.uuid5(_UUID_NAMESPACE, doc_id))


def _metadata_json(metadata: dict[str, object]) -> str:
    from .external_payloads import encode_metadata_json

    payload = encode_metadata_json(metadata)
    assert payload is not None
    return payload


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _directory_byte_size(root: Path | None) -> int | None:
    if root is None or not root.exists():
        return None
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
