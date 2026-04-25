"""Mojo-backed PLAID-style centroid-posting search for benchmarks.

This module owns Python orchestration for Kayak's optional approximation lane.
It does not implement candidate generation or reranking itself: centroid
sampling, token assignment, document candidate scoring, and exact MaxSim rerank
are executed by the Mojo bridge so speed-track measurements exercise the same
systems layer as the exact backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .mojo_exact_cpu import load_module


VECTOR_DIM = 128
INT_BYTES = 8


@dataclass(frozen=True, slots=True)
class KayakPlaidApproxConfig:
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
    doc_ids: tuple[str, ...]
    document_count: int
    document_vector_count: int
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
            document_count=document_count,
            document_vector_count=document_vector_count,
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
