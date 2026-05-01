"""Residual-PQ refine path for TAC candidate windows.

This module owns a benchmarkable Python reference for the paper's refine-stage
idea: store each token as a TAC centroid id plus PQ-compressed normalized
residual and a residual norm. It does not own HNSW candidate generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .tachiom_arrays import _as_query_tensor, _top_positions
from .tachiom_candidates import ranked_candidate_positions_from_scores
from .tachiom_clustering import _kmeans_deterministic, _squared_distances
from .tachiom_index import TachiomTacIndex


@dataclass(frozen=True, slots=True)
class TachiomResidualPqConfig:
    """Explicit knobs for normalized residual product quantization."""

    subspace_count: int = 32
    codebook_size: int = 256
    kmeans_iterations: int = 8
    residual_norm_floor: float = 1.0e-12
    training_sample_count: int | None = None

    def validate(self, *, vector_dim: int) -> None:
        if self.subspace_count <= 0:
            raise ValueError("subspace_count must be positive")
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.codebook_size > 256:
            raise ValueError("codebook_size must fit in uint8 codes")
        if self.kmeans_iterations <= 0:
            raise ValueError("kmeans_iterations must be positive")
        if self.residual_norm_floor <= 0.0:
            raise ValueError("residual_norm_floor must be positive")
        if self.training_sample_count is not None and self.training_sample_count <= 0:
            raise ValueError("training_sample_count must be positive when provided")
        if vector_dim % self.subspace_count != 0:
            raise ValueError("subspace_count must divide vector_dim")


@dataclass(frozen=True, slots=True)
class TachiomResidualPqIndex:
    """TAC centroid postings plus centroid-id and residual-PQ document layout."""

    doc_ids: tuple[str, ...]
    doc_offsets: np.ndarray
    centroids: np.ndarray
    centroid_doc_postings: tuple[np.ndarray, ...]
    token_centroid_positions: np.ndarray
    residual_norms: np.ndarray
    pq_codes: np.ndarray
    pq_codebooks: np.ndarray
    tac_config_candidate_k: int
    tac_config_centroids_per_query_vector: int
    tac_config_candidate_pruning_alpha: float | None
    config: TachiomResidualPqConfig
    build_seconds: float

    @classmethod
    def from_tac_index(
        cls,
        index: TachiomTacIndex,
        *,
        config: TachiomResidualPqConfig,
    ) -> "TachiomResidualPqIndex":
        config.validate(vector_dim=index.vector_dim)
        started_at = time.perf_counter()
        token_centroid_positions = _nearest_token_centroid_positions(index)
        residuals = np.ascontiguousarray(
            index.token_values - index.centroids[token_centroid_positions],
            dtype=VECTOR_DTYPE,
        )
        residual_norms = np.linalg.norm(residuals, axis=1).astype(VECTOR_DTYPE)
        safe_norms = np.maximum(
            residual_norms,
            np.asarray(config.residual_norm_floor, dtype=VECTOR_DTYPE),
        )
        normalized_residuals = np.ascontiguousarray(
            residuals / safe_norms[:, None],
            dtype=VECTOR_DTYPE,
        )
        zero_residuals = residual_norms <= np.asarray(
            config.residual_norm_floor,
            dtype=VECTOR_DTYPE,
        )
        if np.any(zero_residuals):
            normalized_residuals[zero_residuals] = np.asarray(0.0, dtype=VECTOR_DTYPE)

        pq_codebooks, pq_codes = _train_residual_pq(
            normalized_residuals,
            config=config,
        )
        return cls(
            doc_ids=index.doc_ids,
            doc_offsets=np.ascontiguousarray(index.doc_offsets, dtype=INDEX_OFFSET_DTYPE),
            centroids=np.ascontiguousarray(index.centroids, dtype=VECTOR_DTYPE),
            centroid_doc_postings=tuple(
                np.ascontiguousarray(row, dtype=INDEX_OFFSET_DTYPE)
                for row in index.centroid_doc_postings
            ),
            token_centroid_positions=np.ascontiguousarray(
                token_centroid_positions,
                dtype=INDEX_OFFSET_DTYPE,
            ),
            residual_norms=np.ascontiguousarray(residual_norms, dtype=VECTOR_DTYPE),
            pq_codes=np.ascontiguousarray(pq_codes, dtype=np.uint8),
            pq_codebooks=np.ascontiguousarray(pq_codebooks, dtype=VECTOR_DTYPE),
            tac_config_candidate_k=index.config.candidate_k,
            tac_config_centroids_per_query_vector=(
                index.config.centroids_per_query_vector
            ),
            tac_config_candidate_pruning_alpha=index.config.candidate_pruning_alpha,
            config=config,
            build_seconds=time.perf_counter() - started_at,
        )

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    @property
    def vector_dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def total_vector_count(self) -> int:
        return int(self.token_centroid_positions.shape[0])

    @property
    def centroid_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def subspace_count(self) -> int:
        return int(self.pq_codebooks.shape[0])

    @property
    def effective_codebook_size(self) -> int:
        return int(self.pq_codebooks.shape[1])

    @property
    def posting_count(self) -> int:
        return int(sum(len(row) for row in self.centroid_doc_postings))

    @property
    def index_kind(self) -> str:
        return "token_aware_centroid_postings_residual_pq_rerank"

    @property
    def rerank_kind(self) -> str:
        return "centroid_plus_normalized_residual_pq_maxsim"

    @property
    def index_bytes(self) -> int:
        centroid_posting_bytes = sum(int(row.nbytes) for row in self.centroid_doc_postings)
        return int(
            self.doc_offsets.nbytes
            + self.centroids.nbytes
            + centroid_posting_bytes
            + self.token_centroid_positions.nbytes
            + self.residual_norms.nbytes
            + self.pq_codes.nbytes
            + self.pq_codebooks.nbytes
        )

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
            rows.append(self._pq_rerank_positions(query, candidates, final_k=final_k))
        return tuple(rows)

    def rerank_candidate_positions_batch(
        self,
        queries: np.ndarray,
        candidate_positions_by_query: Sequence[Sequence[int]],
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Apply residual-PQ rerank to externally generated candidate windows."""

        if final_k <= 0:
            raise ValueError("final_k must be positive")
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if len(candidate_positions_by_query) != int(query_tensor.shape[0]):
            raise ValueError("candidate windows must align with query batch")
        rows: list[tuple[int, ...]] = []
        for query, candidate_positions in zip(
            query_tensor,
            candidate_positions_by_query,
            strict=True,
        ):
            rows.append(
                self._pq_rerank_positions(
                    query,
                    candidate_positions,
                    final_k=final_k,
                )
            )
        return tuple(rows)

    def _candidate_positions_for_query(
        self,
        query: np.ndarray,
        *,
        final_k: int | None,
    ) -> tuple[int, ...]:
        if self.tac_config_candidate_k >= self.document_count:
            return tuple(range(self.document_count))

        centroid_scores = np.matmul(query, self.centroids.T)
        selected_per_query_vector = min(
            self.tac_config_centroids_per_query_vector,
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

        candidate_count = min(self.tac_config_candidate_k, self.document_count)
        if not np.any(touched):
            return tuple(range(candidate_count))
        ranked_scores = doc_scores.copy()
        ranked_scores[~touched] = np.float32(-3.4e38)
        return ranked_candidate_positions_from_scores(
            ranked_scores,
            candidate_k=candidate_count,
            final_k=final_k,
            candidate_pruning_alpha=self.tac_config_candidate_pruning_alpha,
        )

    def _pq_rerank_positions(
        self,
        query: np.ndarray,
        candidate_positions: Sequence[int],
        *,
        final_k: int,
    ) -> tuple[int, ...]:
        if len(candidate_positions) == 0:
            return ()
        scores = np.empty(len(candidate_positions), dtype=VECTOR_DTYPE)
        centroid_scores = np.matmul(query, self.centroids.T)
        residual_tables = _residual_distance_tables(query, self.pq_codebooks)
        for output_index, document_position in enumerate(candidate_positions):
            scores[output_index] = self._score_document(
                centroid_scores,
                residual_tables,
                int(document_position),
            )
        top_local = _top_positions(scores, min(final_k, len(candidate_positions)))
        return tuple(int(candidate_positions[int(index)]) for index in top_local)

    def _score_document(
        self,
        centroid_scores: np.ndarray,
        residual_tables: np.ndarray,
        document_position: int,
    ) -> np.float32:
        start = int(self.doc_offsets[document_position])
        stop = int(self.doc_offsets[document_position + 1])
        token_centroids = self.token_centroid_positions[start:stop]
        token_codes = self.pq_codes[start:stop]
        token_norms = self.residual_norms[start:stop]
        token_scores = centroid_scores[:, token_centroids].astype(
            VECTOR_DTYPE,
            copy=True,
        )
        residual_scores = np.zeros_like(token_scores)
        for subspace_index in range(self.subspace_count):
            subspace_codes = token_codes[:, subspace_index]
            residual_scores += residual_tables[subspace_index, subspace_codes, :].T
        token_scores += token_norms[None, :] * residual_scores
        return np.asarray(np.max(token_scores, axis=1).sum(), dtype=VECTOR_DTYPE)


def _nearest_token_centroid_positions(index: TachiomTacIndex) -> np.ndarray:
    assignments = np.empty(index.total_vector_count, dtype=INDEX_OFFSET_DTYPE)
    for token_id in np.unique(index.token_ids):
        token_positions = np.flatnonzero(index.token_ids == token_id)
        centroid_positions = np.flatnonzero(index.centroid_token_ids == token_id)
        if centroid_positions.size == 0:
            # A fixed centroid budget can trim rare token ids; PQ still needs a
            # deterministic residual anchor for every stored token.
            centroid_positions = np.arange(index.centroid_count, dtype=np.int64)
        distances = _squared_distances(
            index.token_values[token_positions],
            index.centroids[centroid_positions],
        )
        nearest = np.argmin(distances, axis=1)
        assignments[token_positions] = centroid_positions[nearest]
    return np.ascontiguousarray(assignments, dtype=INDEX_OFFSET_DTYPE)


def _train_residual_pq(
    normalized_residuals: np.ndarray,
    *,
    config: TachiomResidualPqConfig,
) -> tuple[np.ndarray, np.ndarray]:
    vector_dim = int(normalized_residuals.shape[1])
    subspace_count = config.subspace_count
    subspace_dim = vector_dim // subspace_count
    training_positions = _training_sample_positions(
        total_count=int(normalized_residuals.shape[0]),
        sample_count=config.training_sample_count,
    )
    training_residuals = normalized_residuals[training_positions]
    effective_codebook_size = min(config.codebook_size, int(training_residuals.shape[0]))
    codebooks = np.empty(
        (subspace_count, effective_codebook_size, subspace_dim),
        dtype=VECTOR_DTYPE,
    )
    codes = np.empty(
        (int(normalized_residuals.shape[0]), subspace_count),
        dtype=np.uint8,
    )
    for subspace_index in range(subspace_count):
        start = subspace_index * subspace_dim
        stop = start + subspace_dim
        centroids, _ = _kmeans_deterministic(
            training_residuals[:, start:stop],
            k=effective_codebook_size,
            iterations=config.kmeans_iterations,
        )
        assignments = np.argmin(
            _squared_distances(normalized_residuals[:, start:stop], centroids),
            axis=1,
        )
        codebooks[subspace_index] = centroids
        codes[:, subspace_index] = assignments.astype(np.uint8, copy=False)
    return codebooks, codes


def _training_sample_positions(
    *,
    total_count: int,
    sample_count: int | None,
) -> np.ndarray:
    if sample_count is None or sample_count >= total_count:
        return np.arange(total_count, dtype=np.int64)
    # Deterministic coverage avoids hidden benchmark randomness and keeps build
    # cost reproducible while sampling across the whole packed token stream.
    return np.linspace(0, total_count - 1, sample_count, dtype=np.int64)


def _residual_distance_tables(
    query: np.ndarray,
    codebooks: np.ndarray,
) -> np.ndarray:
    subspace_count = int(codebooks.shape[0])
    subspace_dim = int(codebooks.shape[2])
    query_count = int(query.shape[0])
    tables = np.empty(
        (subspace_count, int(codebooks.shape[1]), query_count),
        dtype=VECTOR_DTYPE,
    )
    for subspace_index in range(subspace_count):
        start = subspace_index * subspace_dim
        stop = start + subspace_dim
        tables[subspace_index] = np.matmul(
            codebooks[subspace_index],
            query[:, start:stop].T,
        )
    return tables
