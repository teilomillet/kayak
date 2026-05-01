"""Prepared TAC index orchestration and exact candidate-window rerank."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .dtypes import TOKEN_ID_DTYPE, VECTOR_DTYPE
from .late_scores import SearchHit
from .tachiom_allocation import allocate_tac_centroid_counts, _allocation_summary
from .tachiom_arrays import (
    _as_doc_offsets,
    _as_document_tensor,
    _as_flat_token_ids,
    _as_query_tensor,
    _as_token_id_matrix,
    _as_token_matrix,
    _doc_positions_from_offsets,
    _regular_doc_offsets,
    _regular_vector_count_or_none,
    _single_query_batch,
    _top_positions,
)
from .tachiom_candidates import ranked_candidate_positions_from_scores
from .tachiom_clustering import _build_centroids
from .tachiom_types import TachiomTacConfig, TacAllocationSummary

if TYPE_CHECKING:
    from .late_index import LateIndex
    from .late_query import LateQuery
    from .late_query_batch import LateQueryBatch


@dataclass(frozen=True, slots=True)
class TachiomTacIndex:
    """Prepared token-aware centroid index with exact candidate-window rerank."""

    doc_ids: tuple[str, ...]
    token_ids: np.ndarray
    token_values: np.ndarray
    token_doc_positions: np.ndarray
    doc_offsets: np.ndarray
    regular_document_vector_count: int | None
    centroids: np.ndarray
    centroid_token_ids: np.ndarray
    centroid_doc_postings: tuple[np.ndarray, ...]
    config: TachiomTacConfig
    allocation_summary: TacAllocationSummary
    build_seconds: float
    documents: np.ndarray | None

    @classmethod
    def build(
        cls,
        *,
        doc_ids: Sequence[object],
        documents: np.ndarray,
        token_ids: np.ndarray,
        config: TachiomTacConfig,
        final_k: int,
    ) -> "TachiomTacIndex":
        config.validate(final_k=final_k)
        started_at = time.perf_counter()
        normalized_doc_ids = tuple(str(doc_id) for doc_id in doc_ids)
        document_tensor = _as_document_tensor(documents)
        token_id_matrix = _as_token_id_matrix(
            token_ids,
            expected_shape=document_tensor.shape[:2],
        )
        if len(normalized_doc_ids) != int(document_tensor.shape[0]):
            raise ValueError("doc_ids must match document tensor length")

        document_count = int(document_tensor.shape[0])
        document_vector_count = int(document_tensor.shape[1])
        total_vector_count = document_count * document_vector_count
        doc_offsets = _regular_doc_offsets(
            document_count=document_count,
            document_vector_count=document_vector_count,
        )
        token_values = np.ascontiguousarray(
            document_tensor.reshape(total_vector_count, document_tensor.shape[2]),
            dtype=VECTOR_DTYPE,
        )
        flat_token_ids = np.ascontiguousarray(
            token_id_matrix.reshape(total_vector_count),
            dtype=TOKEN_ID_DTYPE,
        )
        return cls._build_from_packed_fields(
            doc_ids=normalized_doc_ids,
            token_values=token_values,
            token_ids=flat_token_ids,
            doc_offsets=doc_offsets,
            regular_document_vector_count=document_vector_count,
            documents=document_tensor,
            config=config,
            final_k=final_k,
            started_at=started_at,
        )

    @classmethod
    def build_packed(
        cls,
        *,
        doc_ids: Sequence[object],
        doc_offsets: np.ndarray,
        token_vectors: np.ndarray,
        token_ids: np.ndarray,
        config: TachiomTacConfig,
        final_k: int,
    ) -> "TachiomTacIndex":
        """Build TAC from packed vectors and one aligned token-id buffer."""

        config.validate(final_k=final_k)
        started_at = time.perf_counter()
        normalized_doc_ids = tuple(str(doc_id) for doc_id in doc_ids)
        token_values = _as_token_matrix(token_vectors)
        offsets = _as_doc_offsets(
            doc_offsets,
            document_count=len(normalized_doc_ids),
            total_vector_count=int(token_values.shape[0]),
        )
        flat_token_ids = _as_flat_token_ids(
            token_ids,
            expected_length=int(token_values.shape[0]),
        )
        return cls._build_from_packed_fields(
            doc_ids=normalized_doc_ids,
            token_values=token_values,
            token_ids=flat_token_ids,
            doc_offsets=offsets,
            regular_document_vector_count=_regular_vector_count_or_none(offsets),
            documents=None,
            config=config,
            final_k=final_k,
            started_at=started_at,
        )

    @classmethod
    def from_late_index(
        cls,
        late_index: "LateIndex",
        *,
        config: TachiomTacConfig,
        final_k: int,
    ) -> "TachiomTacIndex":
        """Prepare TAC from a ``LateIndex`` that carries aligned token ids."""

        if late_index.token_ids is None:
            raise ValueError("Tachiom TAC requires index token_ids")
        return cls.build_packed(
            doc_ids=late_index.doc_ids,
            doc_offsets=late_index.doc_offsets,
            token_vectors=late_index.as_packed_token_matrix(),
            token_ids=late_index.token_ids,
            config=config,
            final_k=final_k,
        )

    @classmethod
    def _build_from_packed_fields(
        cls,
        *,
        doc_ids: tuple[str, ...],
        token_values: np.ndarray,
        token_ids: np.ndarray,
        doc_offsets: np.ndarray,
        regular_document_vector_count: int | None,
        documents: np.ndarray | None,
        config: TachiomTacConfig,
        final_k: int,
        started_at: float,
    ) -> "TachiomTacIndex":
        config.validate(final_k=final_k)
        token_doc_positions = _doc_positions_from_offsets(doc_offsets)
        allocation = allocate_tac_centroid_counts(
            token_values=token_values,
            token_ids=token_ids,
            config=config,
        )
        centroids, centroid_token_ids, centroid_doc_postings = _build_centroids(
            token_values=token_values,
            token_ids=token_ids,
            token_doc_positions=token_doc_positions,
            allocation=allocation,
            config=config,
        )
        summary = _allocation_summary(
            allocation=allocation,
            token_ids=token_ids,
            requested_centroid_count=config.centroid_count,
            config=config,
        )
        return cls(
            doc_ids=doc_ids,
            token_ids=token_ids,
            token_values=token_values,
            token_doc_positions=token_doc_positions,
            doc_offsets=doc_offsets,
            regular_document_vector_count=regular_document_vector_count,
            centroids=centroids,
            centroid_token_ids=centroid_token_ids,
            centroid_doc_postings=centroid_doc_postings,
            config=config,
            allocation_summary=summary,
            build_seconds=time.perf_counter() - started_at,
            documents=documents,
        )

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    @property
    def document_vector_count(self) -> int:
        if self.regular_document_vector_count is None:
            raise ValueError("document_vector_count is not regular for this index")
        return self.regular_document_vector_count

    @property
    def vector_dim(self) -> int:
        return int(self.token_values.shape[1])

    @property
    def total_vector_count(self) -> int:
        return int(self.token_values.shape[0])

    @property
    def centroid_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def posting_count(self) -> int:
        return int(sum(len(row) for row in self.centroid_doc_postings))

    @property
    def sidecar_index_bytes(self) -> int:
        return (
            int(self.centroids.nbytes)
            + int(self.centroid_token_ids.nbytes)
            + int(self.token_ids.nbytes)
            + int(self.doc_offsets.nbytes)
            + sum(int(row.nbytes) for row in self.centroid_doc_postings)
        )

    @property
    def exact_rerank_vector_bytes(self) -> int:
        return int(self.token_values.nbytes)

    @property
    def index_bytes(self) -> int:
        return self.sidecar_index_bytes + self.exact_rerank_vector_bytes

    @property
    def index_kind(self) -> str:
        return "token_aware_centroid_postings_exact_rerank"

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        return tuple(
            self._candidate_positions_for_query(query, final_k=final_k)
            for query in query_tensor
        )

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        rows: list[tuple[int, ...]] = []
        for query in query_tensor:
            candidates = self._candidate_positions_for_query(query, final_k=final_k)
            rows.append(self._exact_rerank_positions(query, candidates, final_k=final_k))
        return tuple(rows)

    def search_batch_hits(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        rows: list[tuple[SearchHit, ...]] = []
        for query in query_tensor:
            candidates = self._candidate_positions_for_query(query, final_k=final_k)
            scored = self._exact_rerank_position_scores(query, candidates, final_k=final_k)
            rows.append(
                tuple(
                    SearchHit(doc_id=self.doc_ids[position], score=score)
                    for position, score in scored
                )
            )
        return tuple(rows)

    def search_query(
        self,
        query: "LateQuery",
        *,
        final_k: int,
    ) -> tuple[SearchHit, ...]:
        return self.search_query_batch(_single_query_batch(query), final_k=final_k)[0]

    def search_query_batch(
        self,
        query_batch: "LateQueryBatch",
        *,
        final_k: int,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        rows: list[tuple[SearchHit, ...]] = []
        for query in query_batch.queries:
            rows.append(self.search_batch_hits(query.as_vector_matrix(), final_k=final_k)[0])
        return tuple(rows)

    def _candidate_positions_for_query(
        self,
        query: np.ndarray,
        *,
        final_k: int | None,
    ) -> tuple[int, ...]:
        if self.config.candidate_k >= self.document_count:
            return tuple(range(self.document_count))

        centroid_scores = np.matmul(query, self.centroids.T)
        selected_per_query_vector = min(
            self.config.centroids_per_query_vector,
            self.centroid_count,
        )
        doc_scores = np.zeros(self.document_count, dtype=VECTOR_DTYPE)
        touched = np.zeros(self.document_count, dtype=bool)

        for query_token_index in range(query.shape[0]):
            scores = centroid_scores[query_token_index]
            centroid_positions = _top_positions(scores, selected_per_query_vector)
            best_for_query_token = np.full(
                self.document_count,
                -np.inf,
                dtype=VECTOR_DTYPE,
            )
            for centroid_position in centroid_positions:
                posting = self.centroid_doc_postings[int(centroid_position)]
                if posting.size == 0:
                    continue
                score = VECTOR_DTYPE(scores[int(centroid_position)])
                best_for_query_token[posting] = np.maximum(
                    best_for_query_token[posting],
                    score,
                )
                touched[posting] = True

            matched = np.isfinite(best_for_query_token)
            doc_scores[matched] += best_for_query_token[matched]

        candidate_count = min(self.config.candidate_k, self.document_count)
        if not np.any(touched):
            return tuple(range(candidate_count))
        ranked_scores = doc_scores.copy()
        ranked_scores[~touched] = np.float32(-3.4e38)
        return ranked_candidate_positions_from_scores(
            ranked_scores,
            candidate_k=candidate_count,
            final_k=final_k,
            candidate_pruning_alpha=self.config.candidate_pruning_alpha,
        )

    def _exact_rerank_positions(
        self,
        query: np.ndarray,
        candidate_positions: Sequence[int],
        *,
        final_k: int,
    ) -> tuple[int, ...]:
        return tuple(
            position
            for position, _ in self._exact_rerank_position_scores(
                query,
                candidate_positions,
                final_k=final_k,
            )
        )

    def _exact_rerank_position_scores(
        self,
        query: np.ndarray,
        candidate_positions: Sequence[int],
        *,
        final_k: int,
    ) -> tuple[tuple[int, float], ...]:
        if len(candidate_positions) == 0:
            return ()
        scores = np.empty(len(candidate_positions), dtype=VECTOR_DTYPE)
        for output_index, document_position in enumerate(candidate_positions):
            start = int(self.doc_offsets[int(document_position)])
            stop = int(self.doc_offsets[int(document_position) + 1])
            document = self.token_values[start:stop]
            interactions = np.matmul(query, document.T)
            scores[output_index] = np.max(interactions, axis=1).sum()
        top_local = _top_positions(scores, min(final_k, len(candidate_positions)))
        return tuple(
            (int(candidate_positions[int(index)]), float(scores[int(index)]))
            for index in top_local
        )
