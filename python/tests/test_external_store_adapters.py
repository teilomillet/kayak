from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unittest
from unittest.mock import patch

import numpy as np

import kayak


def _matrix(*rows: tuple[float, ...]) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


class _FakePointStruct:
    def __init__(self, *, id: str, vector: object, payload: dict[str, object]) -> None:
        self.id = str(id)
        self.vector = vector
        self.payload = payload


class _FakeMatchValue:
    def __init__(self, *, value: object) -> None:
        self.value = value


class _FakeFieldCondition:
    def __init__(self, *, key: str, match: _FakeMatchValue) -> None:
        self.key = key
        self.match = match


class _FakeFilter:
    def __init__(self, *, must: list[_FakeFieldCondition]) -> None:
        self.must = must


class _FakePointIdsList:
    def __init__(self, *, points: list[str]) -> None:
        self.points = list(points)


class _FakeVectorParams:
    def __init__(self, *, size: int, distance: object, multivector_config: object) -> None:
        self.size = size
        self.distance = distance
        self.multivector_config = multivector_config


class _FakeMultiVectorConfig:
    def __init__(self, *, comparator: object) -> None:
        self.comparator = comparator


class _FakeDistance:
    COSINE = "cosine"


class _FakeMultiVectorComparator:
    MAX_SIM = "max_sim"


class _FakeQdrantModels:
    PointStruct = _FakePointStruct
    MatchValue = _FakeMatchValue
    FieldCondition = _FakeFieldCondition
    Filter = _FakeFilter
    PointIdsList = _FakePointIdsList
    VectorParams = _FakeVectorParams
    MultiVectorConfig = _FakeMultiVectorConfig
    Distance = _FakeDistance
    MultiVectorComparator = _FakeMultiVectorComparator


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.collections: dict[str, dict[str, object]] = {}

    def collection_exists(self, name: str) -> bool:
        return name in self.collections

    def create_collection(self, *, collection_name: str, vectors_config: object) -> None:
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "points": {},
        }

    def upsert(self, *, collection_name: str, points: list[_FakePointStruct], wait: bool) -> None:
        stored = self.collections[collection_name]["points"]
        assert isinstance(stored, dict)
        for point in points:
            stored[point.id] = _FakePointStruct(
                id=point.id,
                vector=point.vector,
                payload=dict(point.payload),
            )

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
        with_vectors: bool,
    ) -> list[_FakePointStruct]:
        stored = self.collections[collection_name]["points"]
        assert isinstance(stored, dict)
        return [stored[doc_id] for doc_id in ids if doc_id in stored]

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: _FakeFilter | None,
        limit: int,
        offset: int | None,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[_FakePointStruct], int | None]:
        stored = self.collections[collection_name]["points"]
        assert isinstance(stored, dict)
        records = list(stored.values())
        if scroll_filter is not None:
            records = [
                record
                for record in records
                if all(
                    record.payload.get(condition.key) == condition.match.value
                    for condition in scroll_filter.must
                )
            ]
        start = 0 if offset is None else offset
        page = records[start : start + limit]
        next_offset = start + limit
        if next_offset >= len(records):
            next_offset = None
        return page, next_offset

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: _FakePointIdsList,
        wait: bool,
    ) -> None:
        stored = self.collections[collection_name]["points"]
        assert isinstance(stored, dict)
        for doc_id in points_selector.points:
            stored.pop(doc_id, None)


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, object]],
    ) -> None:
        for doc_id, embedding, document, metadata in zip(
            ids,
            embeddings,
            documents,
            metadatas,
            strict=True,
        ):
            self.records[doc_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": dict(metadata),
            }

    def delete(self, *, ids: list[str]) -> None:
        for doc_id in ids:
            self.records.pop(doc_id, None)

    def get(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, object]:
        selected = list(self.records.items())
        if ids is not None:
            selected = [(doc_id, self.records[doc_id]) for doc_id in ids if doc_id in self.records]
        if where is not None:
            selected = [
                (doc_id, row)
                for doc_id, row in selected
                if all(row["metadata"].get(key) == value for key, value in where.items())
            ]
        if ids is None:
            start = 0 if offset is None else offset
            stop = None if limit is None else start + limit
            selected = selected[start:stop]
        metadata_rows = [row["metadata"] for _, row in selected]
        documents = [row["document"] for _, row in selected]
        return {
            "ids": [doc_id for doc_id, _ in selected],
            "metadatas": metadata_rows if include and "metadatas" in include else None,
            "documents": documents if include and "documents" in include else None,
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeChromaCollection] = {}

    def get_or_create_collection(self, *, name: str) -> _FakeChromaCollection:
        return self.collections.setdefault(name, _FakeChromaCollection())

    def get_collection(self, *, name: str) -> _FakeChromaCollection:
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]


@dataclass
class _FakeWeaviateObject:
    properties: dict[str, object]
    vector: dict[str, object]


class _FakeWeaviateData:
    def __init__(self, collection: "_FakeWeaviateCollection") -> None:
        self._collection = collection

    def insert(
        self,
        *,
        properties: dict[str, object],
        vector: dict[str, object],
        uuid: str,
    ) -> None:
        self._collection.records[uuid] = _FakeWeaviateObject(
            properties=dict(properties),
            vector=dict(vector),
        )

    def delete_by_id(self, uuid: str) -> None:
        self._collection.records.pop(uuid, None)


class _FakeWeaviateCollection:
    def __init__(self) -> None:
        self.records: dict[str, _FakeWeaviateObject] = {}
        self.data = _FakeWeaviateData(self)

    def iterator(
        self,
        *,
        include_vector: bool,
        return_properties: list[str],
    ):
        del include_vector, return_properties
        return iter(self.records.values())


class _FakeWeaviateCollections:
    def __init__(self) -> None:
        self._collections: dict[str, _FakeWeaviateCollection] = {}

    def exists(self, name: str) -> bool:
        return name in self._collections

    def create(self, *, name: str, properties: list[object], vector_config: object) -> None:
        del properties, vector_config
        self._collections[name] = _FakeWeaviateCollection()

    def get(self, name: str) -> _FakeWeaviateCollection:
        return self._collections[name]


class _FakeWeaviateClient:
    def __init__(self) -> None:
        self.collections = _FakeWeaviateCollections()


class _FakeDataType:
    TEXT = "text"


class _FakeProperty:
    def __init__(self, *, name: str, data_type: object) -> None:
        self.name = name
        self.data_type = data_type


class _FakeMultiVectors:
    @staticmethod
    def self_provided(*, name: str) -> dict[str, str]:
        return {"name": name}


class _FakeConfigure:
    MultiVectors = _FakeMultiVectors


class _FakeConfig:
    DataType = _FakeDataType
    Property = _FakeProperty
    Configure = _FakeConfigure


class _FakeWvc:
    config = _FakeConfig


class _FakePgVectorCursor:
    def __init__(self, connection: "_FakePgVectorConnection") -> None:
        self._connection = connection
        self._results: list[tuple[object, ...]] = []

    def __enter__(self) -> "_FakePgVectorCursor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb

    def execute(
        self,
        query: str,
        params: tuple[object, ...] | None = None,
    ) -> None:
        normalized = " ".join(query.split()).lower()
        arguments = () if params is None else params

        if normalized == "create extension if not exists vector":
            self._connection.extension_enabled = True
            self._results = []
            return

        if normalized.startswith("select to_regclass("):
            self._results = [
                (
                    arguments[0] if self._connection.table_exists else None,
                )
            ]
            return

        if normalized.startswith("create table if not exists"):
            match = re.search(r"vector\((\d+)\)\[\]", normalized)
            assert match is not None
            self._connection.table_exists = True
            self._connection.vector_dim = int(match.group(1))
            self._results = []
            return

        if "select count(*)" in normalized:
            rows = list(self._connection.rows.values())
            self._results = [
                (
                    len(rows),
                    sum(len(row["token_matrix"]) for row in rows),
                    any(row["text"] is not None for row in rows),
                    any(row["metadata"] is not None for row in rows),
                )
            ]
            return

        if normalized.startswith("select token_matrix from") and "limit 1" in normalized:
            if not self._connection.rows:
                self._results = []
                return
            first_row = next(iter(self._connection.rows.values()))
            self._results = [(first_row["token_matrix"],)]
            return

        if normalized.startswith("select doc_id, token_matrix,"):
            requested_doc_ids: set[str] | None = None
            metadata_filter: dict[str, object] | None = None
            argument_index = 0
            if "doc_id = any(%s)" in normalized:
                requested_doc_ids = set(str(doc_id) for doc_id in arguments[argument_index])
                argument_index += 1
            if "metadata_json @> %s::jsonb" in normalized:
                metadata_filter = dict(json.loads(str(arguments[argument_index])))

            include_text = "token_matrix, text, metadata_json" in normalized
            selected: list[tuple[object, ...]] = []
            for doc_id, row in self._connection.rows.items():
                if requested_doc_ids is not None and doc_id not in requested_doc_ids:
                    continue
                metadata = row["metadata"]
                if metadata_filter is not None and not all(
                    metadata is not None and metadata.get(key) == value
                    for key, value in metadata_filter.items()
                ):
                    continue
                selected.append(
                    (
                        doc_id,
                        row["token_matrix"],
                        row["text"] if include_text else None,
                        None if metadata is None else dict(metadata),
                    )
                )
            self._results = selected
            return

        if normalized.startswith("delete from"):
            for doc_id in arguments[0]:
                self._connection.rows.pop(str(doc_id), None)
            self._results = []
            return

        raise AssertionError(f"unexpected pgvector SQL: {query}")

    def executemany(
        self,
        query: str,
        params_seq: list[tuple[object, ...]],
    ) -> None:
        normalized = " ".join(query.split()).lower()
        if not normalized.startswith("insert into"):
            raise AssertionError(f"unexpected pgvector SQL: {query}")

        assert self._connection.vector_dim is not None
        for doc_id, token_matrix, text, metadata_json in params_seq:
            matrix = np.asarray(token_matrix, dtype=np.float32)
            if matrix.shape[1] != self._connection.vector_dim:
                raise ValueError("vector dimension mismatch")
            self._connection.rows[str(doc_id)] = {
                "token_matrix": matrix,
                "text": None if text is None else str(text),
                "metadata": (
                    None
                    if metadata_json is None
                    else dict(json.loads(str(metadata_json)))
                ),
            }
        self._results = []

    def fetchone(self) -> tuple[object, ...] | None:
        if not self._results:
            return None
        return self._results[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._results)


class _FakePgVectorConnection:
    def __init__(self) -> None:
        self.closed = False
        self.extension_enabled = False
        self.table_exists = False
        self.vector_dim: int | None = None
        self.vector_registered = False
        self.commit_count = 0
        self.rollback_count = 0
        self.rows: dict[str, dict[str, object]] = {}

    def cursor(self) -> _FakePgVectorCursor:
        return _FakePgVectorCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


class ExternalStoreAdapterTests(unittest.TestCase):
    def _documents(self) -> kayak.LateDocuments:
        return kayak.documents(
            ["doc-a", "doc-b", "doc-c"],
            [
                _matrix((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                _matrix((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                _matrix((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ],
            texts=["alpha", "beta", "gamma"],
        )

    def _metadata(self) -> list[dict[str, object]]:
        return [
            {"topic": "hardware", "tenant": "a"},
            {"topic": "software", "tenant": "a"},
            {"topic": "hardware", "tenant": "b"},
        ]

    @patch("kayak.stores.qdrant_store.require_qdrant")
    def test_qdrant_store_is_kayak_compliant(self, require_qdrant_mock: object) -> None:
        require_qdrant_mock.return_value = (object, _FakeQdrantModels)
        client = _FakeQdrantClient()
        store = kayak.open_store("qdrant", client=client, collection_name="docs")

        store.upsert(self._documents(), metadata=self._metadata())
        stats = store.stats()
        self.assertEqual(stats.kind, "qdrant")
        self.assertEqual(stats.document_count, 3)
        self.assertEqual(stats.total_vector_count, 6)
        self.assertEqual(stats.vector_dim, 3)

        filtered = store.load_index(where={"tenant": "a"}, include_text=True)
        self.assertEqual(filtered.doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(filtered.doc_texts, ("alpha", "beta"))

        query = kayak.query([[0.0, 0.0, 1.0]])
        hits = kayak.search(
            query,
            filtered,
            k=1,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        self.assertEqual(hits[0].doc_id, "doc-b")

        store.delete(["doc-b"])
        after_delete = store.load_index(include_text=True)
        self.assertEqual(after_delete.doc_ids, ("doc-a", "doc-c"))

    @patch("kayak.stores.chromadb_store.require_chromadb")
    def test_chromadb_store_is_kayak_compliant(
        self,
        require_chromadb_mock: object,
    ) -> None:
        require_chromadb_mock.return_value = object()
        client = _FakeChromaClient()
        store = kayak.open_store("chromadb", client=client, collection_name="docs")

        store.upsert(self._documents(), metadata=self._metadata())
        stats = store.stats()
        self.assertEqual(stats.kind, "chromadb")
        self.assertEqual(stats.document_count, 3)
        self.assertEqual(stats.total_vector_count, 6)
        self.assertEqual(stats.vector_dim, 3)

        subset = store.load_index(
            doc_ids=["doc-c", "doc-a"],
            where={"topic": "hardware"},
            include_text=True,
        )
        self.assertEqual(subset.doc_ids, ("doc-c", "doc-a"))
        self.assertEqual(subset.doc_texts, ("gamma", "alpha"))

        query = kayak.query([[1.0, 0.0, 0.0]])
        hits = kayak.search(
            query,
            subset,
            k=1,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        self.assertEqual(hits[0].doc_id, "doc-c")

        store.delete(["doc-a"])
        after_delete = store.load_index(include_text=True)
        self.assertEqual(after_delete.doc_ids, ("doc-b", "doc-c"))

    @patch("kayak.stores.weaviate_store.require_weaviate")
    def test_weaviate_store_is_kayak_compliant(
        self,
        require_weaviate_mock: object,
    ) -> None:
        require_weaviate_mock.return_value = (object(), _FakeWvc)
        client = _FakeWeaviateClient()
        store = kayak.open_store(
            "weaviate",
            client=client,
            collection_name="Doc",
            vector_name="colbert",
        )

        store.upsert(self._documents(), metadata=self._metadata())
        stats = store.stats()
        self.assertEqual(stats.kind, "weaviate")
        self.assertEqual(stats.document_count, 3)
        self.assertEqual(stats.total_vector_count, 6)
        self.assertEqual(stats.vector_dim, 3)

        filtered = store.load_index(where={"topic": "hardware"}, include_text=True)
        self.assertEqual(filtered.doc_ids, ("doc-a", "doc-c"))
        self.assertEqual(filtered.doc_texts, ("alpha", "gamma"))

        query = kayak.query([[0.0, 1.0, 0.0]])
        hits = kayak.search(
            query,
            filtered,
            k=1,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        self.assertEqual(hits[0].doc_id, "doc-a")

        store.delete(["doc-a"])
        after_delete = store.load_index(include_text=True)
        self.assertEqual(after_delete.doc_ids, ("doc-b", "doc-c"))

    @patch("kayak.stores.pgvector_store.require_pgvector")
    def test_pgvector_store_is_kayak_compliant(
        self,
        require_pgvector_mock: object,
    ) -> None:
        def _register_vector(connection: _FakePgVectorConnection) -> None:
            connection.vector_registered = True

        require_pgvector_mock.return_value = (object(), _register_vector)
        connection = _FakePgVectorConnection()
        store = kayak.open_store(
            "pgvector",
            connection=connection,
            table_name="docs",
        )

        store.upsert(self._documents(), metadata=self._metadata())
        stats = store.stats()
        self.assertEqual(stats.kind, "pgvector")
        self.assertEqual(stats.document_count, 3)
        self.assertEqual(stats.total_vector_count, 6)
        self.assertEqual(stats.vector_dim, 3)
        self.assertTrue(connection.extension_enabled)
        self.assertTrue(connection.vector_registered)

        filtered = store.load_index(
            doc_ids=["doc-c", "doc-a"],
            where={"topic": "hardware"},
            include_text=True,
        )
        self.assertEqual(filtered.doc_ids, ("doc-c", "doc-a"))
        self.assertEqual(filtered.doc_texts, ("gamma", "alpha"))

        query = kayak.query([[1.0, 0.0, 0.0]])
        hits = kayak.search(
            query,
            filtered,
            k=1,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        self.assertEqual(hits[0].doc_id, "doc-c")

        store.delete(["doc-a"])
        after_delete = store.load_index(include_text=True)
        self.assertEqual(after_delete.doc_ids, ("doc-b", "doc-c"))

        store.close()
        self.assertFalse(connection.closed)


if __name__ == "__main__":
    unittest.main()
