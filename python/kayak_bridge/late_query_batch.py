"""Owns explicit batches of late queries without collapsing ragged query structure."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .array_conversions import to_query_matrices
from .layouts import NUMPY_REFERENCE_BACKEND
from .late_query import LateQuery

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the managed env.
    torch = None


@dataclass(frozen=True, slots=True)
class LateQueryBatch:
    queries: tuple[LateQuery, ...]
    batch_size: int
    vector_dim: int

    @classmethod
    def from_inputs(cls, token_vectors: object) -> "LateQueryBatch":
        if not isinstance(token_vectors, np.ndarray) and not (
            torch is not None and isinstance(token_vectors, torch.Tensor)
        ):
            if isinstance(token_vectors, Sequence):
                normalized = tuple(token_vectors)
                if normalized and all(
                    isinstance(query, LateQuery) for query in normalized
                ):
                    return cls.from_queries(normalized)

        return cls.from_queries(
            [
                LateQuery.from_vectors(matrix)
                for matrix in to_query_matrices(token_vectors, "query batch")
            ]
        )

    @classmethod
    def from_queries(cls, queries: Sequence[LateQuery]) -> "LateQueryBatch":
        normalized = tuple(queries)
        if not normalized:
            raise ValueError("query batch must contain at least one query")

        vector_dim = normalized[0].vector_dim
        for query in normalized:
            if query.vector_dim != vector_dim:
                raise ValueError(
                    "all queries in a batch must share the same vector dimension"
                )

        return cls(
            queries=normalized,
            batch_size=len(normalized),
            vector_dim=vector_dim,
        )

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("query batch must contain at least one query")
        if len(self.queries) != self.batch_size:
            raise ValueError("query batch size must match the number of queries")
        if self.vector_dim <= 0:
            raise ValueError("query batch vector_dim must be positive")

    @property
    def vector_counts(self) -> tuple[int, ...]:
        return tuple(query.vector_count for query in self.queries)

    @property
    def layouts(self) -> tuple[str, ...]:
        return tuple(query.layout for query in self.queries)

    def to_layout(self, layout: str) -> "LateQueryBatch":
        return LateQueryBatch.from_queries(
            [query.to_layout(layout) for query in self.queries]
        )

    def maxsim(
        self, index: "LateIndex", *, backend: str = NUMPY_REFERENCE_BACKEND
    ) -> tuple["LateScores", ...]:
        from .batch_dispatch import maxsim_scores_batch

        return maxsim_scores_batch(self, index, backend=backend)

    def search(
        self,
        index: "LateIndex",
        *,
        k: int,
        backend: str = NUMPY_REFERENCE_BACKEND,
    ) -> tuple[tuple["SearchHit", ...], ...]:
        from .late_ops import search_batch

        return search_batch(self, index, k=k, backend=backend)
