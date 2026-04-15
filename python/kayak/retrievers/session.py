"""Owns reusable search sessions over one fixed late-interaction index.

This module owns:
- binding one text encoder to one already-materialized ``LateIndex``
- repeated exact and staged search against that fixed index
- keeping backend choice explicit at the session boundary

This module does not own:
- text encoding model construction
- store persistence or filtering
- low-level scoring kernels

Assumptions:
- the caller intentionally chose and materialized the exact index slice
- repeated search against the same slice should not require reloading it
"""

from __future__ import annotations

from dataclasses import dataclass

from kayak_bridge.api_types import QueryTextsInput
from kayak_bridge import (
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
from .backend_policy import default_text_retriever_backend


@dataclass(slots=True)
class LateTextSearchSession:
    """Reuse one loaded ``LateIndex`` across repeated text or query searches.

    Use this when you already know the exact slice you want to search and you
    want one object that keeps the encoder, index, and default backend bound
    together for repeated calls.
    """

    encoder: LateTextEncoder
    index: LateIndex
    default_backend: str | None = None

    def __post_init__(self) -> None:
        if self.default_backend is None:
            self.default_backend = default_text_retriever_backend()

    def search_text(
        self,
        text: str,
        *,
        k: int,
        backend: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Encode one query string and search the bound index."""
        query = self.encoder.encode_query(text)
        return self.search_query(query, k=k, backend=backend)

    def search_query(
        self,
        query: LateQuery,
        *,
        k: int,
        backend: str | None = None,
    ) -> tuple[SearchHit, ...]:
        """Run top-k search for one already-encoded query on the bound index."""
        return search(
            query,
            self.index,
            k=k,
            backend=self.default_backend if backend is None else backend,
        )

    def search_text_batch(
        self,
        texts: QueryTextsInput,
        *,
        k: int,
        backend: str | None = None,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        """Encode many query strings and search the same bound index once."""
        query_texts = tuple(str(text) for text in texts)
        queries = tuple(self.encoder.encode_query(text) for text in query_texts)
        batch = query_batch(
            tuple(query.as_vector_matrix() for query in queries)
        )
        return self.search_query_batch(batch, k=k, backend=backend)

    def search_query_batch(
        self,
        batch: LateQueryBatch,
        *,
        k: int,
        backend: str | None = None,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        """Run batched top-k search against the bound index."""
        return search_batch(
            batch,
            self.index,
            k=k,
            backend=self.default_backend if backend is None else backend,
        )

    def search_text_with_plan(
        self,
        text: str,
        plan: SearchPlan,
        *,
        backend: str | None = None,
    ) -> SearchPlanResult:
        """Encode one query string and run a plan on the bound index."""
        query = self.encoder.encode_query(text)
        return self.search_query_with_plan(
            query,
            plan,
            backend=backend,
        )

    def search_query_with_plan(
        self,
        query: LateQuery,
        plan: SearchPlan,
        *,
        backend: str | None = None,
    ) -> SearchPlanResult:
        """Run one explicit search plan against the bound index."""
        return search_with_plan(
            query,
            self.index,
            plan,
            backend=self.default_backend if backend is None else backend,
        )
