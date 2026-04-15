"""Owns the high-level text retrieval workflow surface for the public SDK."""

from __future__ import annotations

from dataclasses import dataclass

from kayak_bridge.api_types import (
    DocIdsInput,
    DocTextsInput,
    MetadataFilterInput,
    MetadataRowsInput,
    QueryTextsInput,
    TokenMatrixInput,
)
from kayak_bridge import (
    LateDocuments,
    LateIndex,
    LateQuery,
    LateQueryBatch,
    SearchHit,
    SearchPlan,
    SearchPlanResult,
    query_batch,
    search,
    search_batch,
    search_with_plan,
)

from ..encoders import LateTextEncoder
from ..stores import LateStore, LateStoreStats, StoreCapabilities
from .backend_policy import default_text_retriever_backend
from .session import LateTextSearchSession


@dataclass(slots=True)
class LateTextRetriever:
    """Compose one text encoder and one late store into one coding workflow.

    This is the main high-level Python interface when your application starts
    from text instead of already-materialized ``LateQuery`` and ``LateIndex``
    objects.

    The retriever owns:
    - encoding query strings
    - encoding and upserting document texts
    - loading exact index slices from the configured store
    - running exact or staged search over those slices
    - opening reusable search sessions over one loaded slice
    """

    encoder: LateTextEncoder
    store: LateStore
    default_backend: str | None = None
    default_layout: str = "packed"
    default_include_text: bool = False

    def __post_init__(self) -> None:
        if self.default_backend is None:
            self.default_backend = default_text_retriever_backend()

    def close(self) -> None:
        """Release any store resources owned by this retriever."""
        self.store.close()

    def __enter__(self) -> "LateTextRetriever":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def capabilities(self) -> StoreCapabilities:
        """Return the capability flags of the underlying store."""
        return self.store.capabilities()

    def stats(self) -> LateStoreStats:
        """Return measurable state for the underlying store."""
        return self.store.stats()

    def encode_query(self, text: str) -> LateQuery:
        """Encode one query string without touching the configured store.

        Use this when you want the retriever to own model access but your
        calling code still wants to keep late-interaction queries explicit.
        """
        return self.encoder.encode_query(text)

    def encode_document_vectors(self, text: str) -> TokenMatrixInput:
        """Encode one document string into token vectors without storing it."""
        return self.encoder.encode_document_vectors(text)

    def encode_documents(
        self,
        doc_ids: DocIdsInput,
        texts: DocTextsInput,
    ) -> LateDocuments:
        """Encode aligned document ids and texts without upserting them.

        This is the side-effect-free counterpart to ``upsert_texts(...)`` for
        cases where the caller wants to keep persistence or materialization
        explicit.
        """
        return self.encoder.encode_documents(doc_ids, texts)

    def upsert_texts(
        self,
        doc_ids: DocIdsInput,
        texts: DocTextsInput,
        *,
        metadata: MetadataRowsInput = None,
    ) -> LateDocuments:
        """Encode aligned texts and upsert them into the configured store.

        Parameters
        ----------
        doc_ids:
            Document ids aligned with ``texts``.
        texts:
            Document texts to encode with the configured encoder.
        metadata:
            Optional metadata rows aligned with ``doc_ids`` and ``texts``.

        Returns
        -------
        LateDocuments
            The encoded late-interaction documents that were written to the
            store.
        """
        documents = self.encoder.encode_documents(doc_ids, texts)
        self.store.upsert(documents, metadata=metadata)
        return documents

    def delete(self, doc_ids: DocIdsInput) -> None:
        """Delete the requested document ids from the underlying store."""
        self.store.delete(doc_ids)

    def load_index(
        self,
        *,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
    ) -> LateIndex:
        """Materialize one reusable exact slice from the configured store.

        Parameters
        ----------
        doc_ids:
            Optional explicit document id subset.
        where:
            Optional metadata filter applied by the store.
        include_text:
            Override whether loaded documents should include document text.
        layout:
            Override the materialized index layout such as ``"packed"`` or
            ``"hybrid_flat_dim128"``.

        Returns
        -------
        LateIndex
            One exact searchable index slice ready for repeated queries.
        """
        return self.store.load_index(
            doc_ids=doc_ids,
            where=where,
            include_text=self.default_include_text
            if include_text is None
            else include_text,
            layout=self.default_layout if layout is None else layout,
        )

    def session(
        self,
        *,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> LateTextSearchSession:
        """Load one exact slice once and return a reusable search session.

        Use this when many queries should hit the same filtered subset or the
        same materialized index layout without reloading that slice on each
        call.
        """
        return LateTextSearchSession(
            encoder=self.encoder,
            index=self.load_index(
                doc_ids=doc_ids,
                where=where,
                include_text=include_text,
                layout=layout,
            ),
            default_backend=self.default_backend if backend is None else backend,
        )

    def search_text(
        self,
        text: str,
        *,
        k: int,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Encode one query string and run top-k search against a loaded slice.

        Use this as the shortest high-level search call when you start from
        raw query text.
        """
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
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Run top-k search for one already-encoded late-interaction query.

        Parameters
        ----------
        query:
            Pre-encoded late-interaction query vectors.
        k:
            Number of hits to return.
        doc_ids, where:
            Optional subset selectors applied before exact search.
        include_text:
            Override whether the loaded slice should include document text.
        layout:
            Override the exact index layout used for this call.
        backend:
            Override the exact scoring backend for this call.

        Returns
        -------
        tuple[SearchHit, ...]
            Exact top-k hits for the selected slice.
        """
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

    def search_text_batch(
        self,
        texts: QueryTextsInput,
        *,
        k: int,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        """Encode many query strings and run batched top-k search.

        Use this when the index slice stays fixed and you want one exact batch
        call over many queries.
        """
        query_texts = tuple(str(text) for text in texts)
        queries = tuple(self.encoder.encode_query(text) for text in query_texts)
        batch = query_batch(
            tuple(query.as_vector_matrix() for query in queries)
        )
        return self.search_query_batch(
            batch,
            k=k,
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
            backend=backend,
        )

    def search_query_batch(
        self,
        batch: LateQueryBatch,
        *,
        k: int,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        """Run batched top-k search for one already-encoded query batch.

        Returns one top-k hit list per query in the batch.
        """
        index = self.load_index(
            doc_ids=doc_ids,
            where=where,
            include_text=include_text,
            layout=layout,
        )
        return search_batch(
            batch,
            index,
            k=k,
            backend=self.default_backend if backend is None else backend,
        )

    def search_text_with_plan(
        self,
        text: str,
        plan: SearchPlan,
        *,
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> SearchPlanResult:
        """Encode one query string and run an explicit search plan.

        Use this when you want candidate generation, reranking, or verifier
        behavior to stay explicit instead of calling plain exact top-k search.
        """
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
        doc_ids: DocIdsInput | None = None,
        where: MetadataFilterInput = None,
        include_text: bool | None = None,
        layout: str | None = None,
        backend: str | None = None,
    ) -> SearchPlanResult:
        """Run an explicit search plan for one already-encoded query.

        Returns a ``SearchPlanResult`` with per-stage profiles, materialized
        artifacts, and the final hit list.
        """
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
