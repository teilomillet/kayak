"""Mojo-backed PLAID-style centroid-posting search for benchmarks.

This module owns Python orchestration for Kayak's optional approximation lane.
It does not implement candidate generation or reranking itself: centroid
sampling, token assignment, document candidate scoring, and exact MaxSim rerank
are executed by the Mojo bridge so speed-track measurements exercise the same
systems layer as the exact backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .layouts import INDEX_LAYOUT_HYBRID_FLAT_DIM128, QUERY_LAYOUT_FLAT_DIM128
from .late_scores import SearchHit
from .mojo_exact_cpu import load_module

if TYPE_CHECKING:
    from .late_index import LateIndex
    from .late_query import LateQuery
    from .late_query_batch import LateQueryBatch


VECTOR_DIM = 128
INT_BYTES = 8


@dataclass(frozen=True, slots=True)
class KayakPlaidApproxConfig:
    """Explicit sampled-centroid approximation knobs for Mojo PLAID search."""

    centroid_count: int = 128
    centroids_per_query_vector: int = 4
    candidate_k: int = 10

    def validate(self, *, final_k: int) -> None:
        _require_positive("centroid_count", self.centroid_count)
        _require_positive(
            "centroids_per_query_vector",
            self.centroids_per_query_vector,
        )
        _require_positive("candidate_k", self.candidate_k)
        if self.candidate_k < final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")


@dataclass(frozen=True, slots=True)
class KayakPlaidApproxIndex:
    """Prepared Mojo PLAID approximation index with exact rerank search."""

    doc_ids: tuple[str, ...]
    document_count: int
    document_vector_count: int | None
    document_vector_counts: tuple[int, ...]
    vector_dim: int
    centroid_count: int
    config: KayakPlaidApproxConfig
    _prepared_index: Any
    _index_bytes: int

    @classmethod
    def build(
        cls,
        *,
        doc_ids: tuple[str, ...],
        documents: np.ndarray,
        config: KayakPlaidApproxConfig,
        final_k: int,
    ) -> "KayakPlaidApproxIndex":
        config.validate(final_k=final_k)
        normalized_documents = _as_document_tensor(documents)
        if len(doc_ids) != int(normalized_documents.shape[0]):
            raise ValueError("doc_ids must match document tensor length")

        document_count = int(normalized_documents.shape[0])
        document_vector_count = int(normalized_documents.shape[1])
        total_vector_count = document_count * document_vector_count
        centroid_count = min(config.centroid_count, total_vector_count)
        doc_offsets = _regular_doc_offsets(
            document_count=document_count,
            document_vector_count=document_vector_count,
        )
        token_values = _flat_values_list(normalized_documents)
        return cls._build_from_flat_fields(
            doc_ids=doc_ids,
            doc_offsets=doc_offsets,
            token_values=token_values,
            document_vector_count=document_vector_count,
            document_vector_counts=(document_vector_count,) * document_count,
            centroid_count=centroid_count,
            config=config,
        )

    @classmethod
    def from_late_index(
        cls,
        late_index: "LateIndex",
        *,
        config: KayakPlaidApproxConfig,
        final_k: int,
    ) -> "KayakPlaidApproxIndex":
        """Prepare a public ``LateIndex`` for Mojo PLAID approximation search."""
        config.validate(final_k=final_k)
        if late_index.vector_dim != VECTOR_DIM:
            raise ValueError("Mojo PLAID approximation currently requires vector_dim=128")

        hybrid_index = late_index.to_layout(INDEX_LAYOUT_HYBRID_FLAT_DIM128)
        assert hybrid_index.token_values is not None
        centroid_count = min(config.centroid_count, hybrid_index.total_vector_count)
        vector_counts = tuple(
            int(hybrid_index.doc_offsets[index + 1] - hybrid_index.doc_offsets[index])
            for index in range(hybrid_index.document_count)
        )
        regular_vector_count = (
            vector_counts[0]
            if len(set(vector_counts)) == 1
            else None
        )
        return cls._build_from_flat_fields(
            doc_ids=hybrid_index.doc_ids,
            doc_offsets=hybrid_index.doc_offsets,
            token_values=_flat_values_list(hybrid_index.token_values),
            document_vector_count=regular_vector_count,
            document_vector_counts=vector_counts,
            centroid_count=centroid_count,
            config=config,
        )

    @classmethod
    def _build_from_flat_fields(
        cls,
        *,
        doc_ids: tuple[str, ...],
        doc_offsets: np.ndarray,
        token_values: list[float],
        document_vector_count: int | None,
        document_vector_counts: tuple[int, ...],
        centroid_count: int,
        config: KayakPlaidApproxConfig,
    ) -> "KayakPlaidApproxIndex":
        module = load_module()
        prepared_index = module.prepare_plaid_approx_hybrid_flat_dim128(
            list(doc_ids),
            doc_offsets.tolist(),
            token_values,
            centroid_count,
        )
        posting_count = int(
            module.plaid_approx_prepared_posting_count(prepared_index)
        )

        index_bytes = _prepared_index_bytes(
            doc_offsets=doc_offsets,
            token_value_count=len(token_values),
            centroid_count=centroid_count,
            posting_count=posting_count,
        )
        return cls(
            doc_ids=doc_ids,
            document_count=len(doc_ids),
            document_vector_count=document_vector_count,
            document_vector_counts=document_vector_counts,
            vector_dim=VECTOR_DIM,
            centroid_count=centroid_count,
            config=config,
            _prepared_index=prepared_index,
            _index_bytes=index_bytes,
        )

    @property
    def index_bytes(self) -> int:
        return self._index_bytes

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        query_values = _flat_query_values_by_query(normalized_queries)
        module = load_module()
        rows = module.search_plaid_approx_prepared_batch(
            query_values,
            final_k,
            self.config.centroids_per_query_vector,
            self.config.candidate_k,
            self._prepared_index,
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def search(
        self,
        query: "LateQuery",
        *,
        final_k: int,
    ) -> tuple[SearchHit, ...]:
        """Return exact-reranked approximate hits for one public query."""
        return self.search_batch(
            _single_query_batch(query),
            final_k=final_k,
        )[0]

    def search_batch(
        self,
        query_batch: "LateQueryBatch",
        *,
        final_k: int,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        """Return exact-reranked approximate hits for a public query batch."""
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        query_values = [
            _flat_query_values_from_query(query, vector_dim=self.vector_dim)
            for query in query_batch.queries
        ]
        module = load_module()
        rows = module.search_plaid_approx_prepared_hits_batch(
            query_values,
            final_k,
            self.config.centroids_per_query_vector,
            self.config.candidate_k,
            self._prepared_index,
        )
        return tuple(
            tuple(
                SearchHit(doc_id=str(raw_hit[0]), score=float(raw_hit[1]))
                for raw_hit in row
            )
            for row in rows
        )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _as_document_tensor(documents: np.ndarray) -> np.ndarray:
    array = np.asarray(documents, dtype=VECTOR_DTYPE)
    if array.ndim != 3:
        raise ValueError(
            "documents must have shape "
            "(document_count, document_vector_count, vector_dim)"
        )
    if 0 in array.shape:
        raise ValueError("documents must not contain empty dimensions")
    if int(array.shape[2]) != VECTOR_DIM:
        raise ValueError("Mojo PLAID approximation currently requires vector_dim=128")
    return np.ascontiguousarray(array)


def _as_query_tensor(queries: np.ndarray, *, vector_dim: int) -> np.ndarray:
    array = np.asarray(queries, dtype=VECTOR_DTYPE)
    if array.ndim != 3:
        raise ValueError(
            "queries must have shape "
            "(query_count, query_vector_count, vector_dim)"
        )
    if 0 in array.shape:
        raise ValueError("queries must not contain empty dimensions")
    if int(array.shape[2]) != vector_dim:
        raise ValueError("query vector_dim must match index vector_dim")
    return np.ascontiguousarray(array)


def _regular_doc_offsets(
    *,
    document_count: int,
    document_vector_count: int,
) -> np.ndarray:
    return (
        np.arange(document_count + 1, dtype=INDEX_OFFSET_DTYPE)
        * document_vector_count
    )


def _flat_values_list(values: np.ndarray) -> list[float]:
    return values.reshape(-1).tolist()


def _flat_query_values_by_query(queries: np.ndarray) -> list[list[float]]:
    return [query.reshape(-1).tolist() for query in queries]


def _flat_query_values_from_query(query: "LateQuery", *, vector_dim: int) -> list[float]:
    flat_query = query.to_layout(QUERY_LAYOUT_FLAT_DIM128)
    if flat_query.vector_dim != vector_dim:
        raise ValueError("query vector_dim must match index vector_dim")
    return flat_query.as_flat_values().tolist()


def _single_query_batch(query: "LateQuery") -> "LateQueryBatch":
    from .late_query_batch import LateQueryBatch

    return LateQueryBatch.from_queries([query])


def _prepared_index_bytes(
    *,
    doc_offsets: np.ndarray,
    token_value_count: int,
    centroid_count: int,
    posting_count: int,
) -> int:
    token_value_bytes = token_value_count * np.dtype(VECTOR_DTYPE).itemsize
    centroid_token_index_bytes = centroid_count * INT_BYTES
    centroid_doc_offset_bytes = (centroid_count + 1) * INT_BYTES
    centroid_doc_index_bytes = posting_count * INT_BYTES
    return int(
        doc_offsets.nbytes
        + token_value_bytes
        + centroid_token_index_bytes
        + centroid_doc_offset_bytes
        + centroid_doc_index_bytes
    )


PlaidApproxConfig = KayakPlaidApproxConfig
PlaidApproxIndex = KayakPlaidApproxIndex
