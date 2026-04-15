"""Owns the high-level text retrieval workflow surface for the public SDK."""

from __future__ import annotations

from dataclasses import dataclass

from kayak_bridge import (
    LateDocuments,
    LateIndex,
    LateQuery,
    SearchPlan,
    search,
    search_with_plan,
)

from ..encoders import LateTextEncoder
from ..stores import LateStore, LateStoreStats, StoreCapabilities
from .backend_policy import default_text_retriever_backend


@dataclass(slots=True)
class LateTextRetriever:
    """Composes one text encoder and one late store into one workflow object."""

    encoder: LateTextEncoder
    store: LateStore
    default_backend: str | None = None
    default_layout: str = "packed"
    default_include_text: bool = False

    def __post_init__(self) -> None:
        if self.default_backend is None:
            self.default_backend = default_text_retriever_backend()

    def capabilities(self) -> StoreCapabilities:
        return self.store.capabilities()

    def stats(self) -> LateStoreStats:
        return self.store.stats()

    def upsert_texts(
        self,
        doc_ids: object,
        texts: object,
        *,
        metadata: object | None = None,
    ) -> LateDocuments:
        documents = self.encoder.encode_documents(doc_ids, texts)
        self.store.upsert(documents, metadata=metadata)
        return documents

    def delete(self, doc_ids: object) -> None:
        self.store.delete(doc_ids)

    def load_index(
        self,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool | None = None,
        layout: str | None = None,
    ) -> LateIndex:
        return self.store.load_index(
            doc_ids=doc_ids,
            where=where,
            include_text=self.default_include_text
            if include_text is None
            else include_text,
            layout=self.default_layout if layout is None else layout,
        )

    def search_text(
        self,
        text: str,
        *,
        k: int,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> object:
        query = self.encoder.encode_query(text)
        return self.search_query(
            query,
            k=k,
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
            backend=backend,
        )

    def search_query(
        self,
        query: LateQuery,
        *,
        k: int,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> object:
        index = self.load_index(
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
        )
        return search(
            query,
            index,
            k=k,
            backend=self.default_backend if backend is None else backend,
        )

    def search_text_with_plan(
        self,
        text: str,
        plan: SearchPlan,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> object:
        query = self.encoder.encode_query(text)
        return self.search_query_with_plan(
            query,
            plan,
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
            backend=backend,
        )

    def search_query_with_plan(
        self,
        query: LateQuery,
        plan: SearchPlan,
        *,
        doc_ids: object | None = None,
        where: object | None = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> object:
        index = self.load_index(
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
        )
        return search_with_plan(
            query,
            index,
            plan,
            backend=self.default_backend if backend is None else backend,
        )
