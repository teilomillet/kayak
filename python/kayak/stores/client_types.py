"""Public-facing protocol types for optional external store clients.

These protocols intentionally describe only the methods that Kayak's public
store adapters rely on. They improve editor and type-checker help for
constructor parameters like ``client=...`` and ``connection=...`` without
coupling the public SDK to one concrete client implementation class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol


class ChromaCollectionLike(Protocol):
    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, object]],
    ) -> object: ...

    def delete(self, *, ids: Sequence[str]) -> object: ...

    def get(
        self,
        *,
        ids: Sequence[str] | None = None,
        where: dict[str, object] | None = None,
        include: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, object]: ...


class ChromaClientLike(Protocol):
    def get_or_create_collection(self, *, name: str) -> ChromaCollectionLike: ...

    def get_collection(self, *, name: str) -> ChromaCollectionLike: ...

    def close(self) -> object: ...


class QdrantClientLike(Protocol):
    def close(self) -> object: ...

    def upsert(
        self,
        *,
        collection_name: str,
        points: Sequence[object],
        wait: bool = True,
    ) -> object: ...

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: object,
        wait: bool = True,
    ) -> object: ...

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: Sequence[object],
        with_payload: bool = True,
        with_vectors: bool = True,
    ) -> Iterable[object]: ...

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: object | None = None,
        limit: int = 256,
        offset: object | None = None,
        with_payload: bool = True,
        with_vectors: bool = True,
    ) -> tuple[Sequence[object], object | None]: ...

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: object,
    ) -> object: ...

    def collection_exists(self, collection_name: str) -> bool: ...


class WeaviateCollectionDataLike(Protocol):
    def insert(
        self,
        *,
        properties: dict[str, object],
        vector: dict[str, object],
        uuid: str,
    ) -> object: ...

    def delete_by_id(self, uuid: str) -> object: ...


class WeaviateCollectionLike(Protocol):
    data: WeaviateCollectionDataLike

    def iterator(
        self,
        *,
        include_vector: bool,
        return_properties: Sequence[str],
    ) -> Iterable[object]: ...


class WeaviateCollectionsLike(Protocol):
    def exists(self, name: str) -> bool: ...

    def get(self, name: str) -> WeaviateCollectionLike: ...

    def create(
        self,
        *,
        name: str,
        properties: Sequence[object],
        vector_config: object,
    ) -> object: ...


class WeaviateClientLike(Protocol):
    collections: WeaviateCollectionsLike

    def close(self) -> object: ...


class PsycopgCursorLike(Protocol):
    def execute(
        self, query: str, params: Sequence[object] | None = None
    ) -> object: ...

    def executemany(
        self, query: str, params_seq: Sequence[Sequence[object]]
    ) -> object: ...

    def fetchone(self) -> object: ...

    def fetchall(self) -> list[object]: ...


class PsycopgConnectionLike(Protocol):
    def cursor(self) -> AbstractContextManager[PsycopgCursorLike]: ...

    def commit(self) -> object: ...

    def rollback(self) -> object: ...

    def close(self) -> object: ...
