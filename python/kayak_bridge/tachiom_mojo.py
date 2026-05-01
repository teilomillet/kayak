"""Mojo execution wrapper for a prepared TAC probe index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .late_scores import SearchHit
from .mojo_exact_cpu import load_module
from .tachiom_arrays import _as_query_tensor, _single_query_batch

if TYPE_CHECKING:
    from .tachiom_hnsw import TachiomTacHnswIndex
    from .late_query import LateQuery
    from .late_query_batch import LateQueryBatch
    from .tachiom_index import TachiomTacIndex
    from .tachiom_pq import TachiomResidualPqIndex


@dataclass(frozen=True, slots=True)
class TachiomTacMojoIndex:
    """Mojo prepared TAC index using Python-built token-aware centroids."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    document_count: int
    centroid_count: int
    centroids_per_query_vector: int
    candidate_k: int
    candidate_pruning_alpha: float | None
    _prepared_index: Any

    @classmethod
    def from_tac_index(cls, index: "TachiomTacIndex") -> "TachiomTacMojoIndex":
        if index.vector_dim != 128:
            raise ValueError("Mojo TAC currently requires vector_dim=128")
        centroid_offsets, centroid_indices = _flatten_centroid_postings(
            index.centroid_doc_postings
        )
        module = load_module()
        prepared_index = module.prepare_tachiom_tac_hybrid_flat_dim128(
            list(index.doc_ids),
            index.doc_offsets.astype(INDEX_OFFSET_DTYPE, copy=False).tolist(),
            index.token_values.astype(VECTOR_DTYPE, copy=False).reshape(-1).tolist(),
            index.centroids.astype(VECTOR_DTYPE, copy=False).reshape(-1).tolist(),
            centroid_offsets.tolist(),
            centroid_indices.tolist(),
        )
        posting_count = int(module.tachiom_tac_prepared_posting_count(prepared_index))
        if posting_count != int(centroid_indices.shape[0]):
            raise RuntimeError("Mojo TAC prepared posting count mismatch")
        return cls(
            doc_ids=index.doc_ids,
            vector_dim=index.vector_dim,
            document_count=index.document_count,
            centroid_count=index.centroid_count,
            centroids_per_query_vector=index.config.centroids_per_query_vector,
            candidate_k=index.config.candidate_k,
            candidate_pruning_alpha=index.config.candidate_pruning_alpha,
            _prepared_index=prepared_index,
        )

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(normalized_queries.shape[0])))
        module = load_module()
        if self.candidate_pruning_alpha is not None:
            if final_k is None:
                raise ValueError("final_k is required when candidate pruning is enabled")
            rows = module.tachiom_tac_candidate_positions_prepared_batch_address_with_pruning(
                [
                    int(normalized_queries.ctypes.data),
                    int(normalized_queries.shape[0]),
                    int(normalized_queries.shape[1]),
                    self.centroids_per_query_vector,
                    self.candidate_k,
                    final_k,
                    float(self.candidate_pruning_alpha),
                    self._prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.tachiom_tac_candidate_positions_prepared_batch_address(
            int(normalized_queries.ctypes.data),
            int(normalized_queries.shape[0]),
            int(normalized_queries.shape[1]),
            self.centroids_per_query_vector,
            self.candidate_k,
            self._prepared_index,
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        if self.candidate_pruning_alpha is not None:
            rows = module.search_tachiom_tac_prepared_batch_address_with_pruning(
                [
                    int(normalized_queries.ctypes.data),
                    int(normalized_queries.shape[0]),
                    int(normalized_queries.shape[1]),
                    final_k,
                    self.centroids_per_query_vector,
                    self.candidate_k,
                    float(self.candidate_pruning_alpha),
                    self._prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.search_tachiom_tac_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                final_k,
                self.centroids_per_query_vector,
                self.candidate_k,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def profile_candidate_generation_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
        measurement_iterations: int,
    ) -> tuple[dict[str, Any], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        if measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        rows = module.tachiom_tac_candidate_generation_profile_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                self.centroids_per_query_vector,
                self.candidate_k,
                final_k,
                measurement_iterations,
                self._prepared_index,
            ]
        )
        return tuple(_profile_pairs_to_dict(row) for row in rows)

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
        query_values = [
            query.to_layout("flat_dim128").as_flat_values().tolist()
            for query in query_batch.queries
        ]
        module = load_module()
        rows = module.search_tachiom_tac_prepared_hits_batch(
            query_values,
            final_k,
            self.centroids_per_query_vector,
            self.candidate_k,
            self._prepared_index,
        )
        return tuple(
            tuple(
                SearchHit(doc_id=str(raw_hit[0]), score=float(raw_hit[1]))
                for raw_hit in row
            )
            for row in rows
        )


@dataclass(frozen=True, slots=True)
class TachiomTacI8MojoIndex:
    """Mojo TAC index with int8 token payload reranking."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    document_count: int
    centroid_count: int
    centroids_per_query_vector: int
    candidate_k: int
    posting_count: int
    index_bytes: int
    _prepared_index: Any

    @classmethod
    def from_tac_index(cls, index: "TachiomTacIndex") -> "TachiomTacI8MojoIndex":
        if index.vector_dim != 128:
            raise ValueError("Mojo TAC i8 currently requires vector_dim=128")
        if index.config.candidate_pruning_alpha is not None:
            raise ValueError("Mojo TAC+i8 candidate pruning is not implemented yet")
        centroid_offsets, centroid_indices = _flatten_centroid_postings(
            index.centroid_doc_postings
        )
        module = load_module()
        prepared_index = module.prepare_tachiom_tac_i8_hybrid_flat_dim128(
            list(index.doc_ids),
            index.doc_offsets.astype(INDEX_OFFSET_DTYPE, copy=False).tolist(),
            index.token_values.astype(VECTOR_DTYPE, copy=False).reshape(-1).tolist(),
            index.centroids.astype(VECTOR_DTYPE, copy=False).reshape(-1).tolist(),
            centroid_offsets.tolist(),
            centroid_indices.tolist(),
        )
        posting_count = int(
            module.tachiom_tac_i8_prepared_posting_count(prepared_index)
        )
        if posting_count != int(centroid_indices.shape[0]):
            raise RuntimeError("Mojo TAC i8 prepared posting count mismatch")
        return cls(
            doc_ids=index.doc_ids,
            vector_dim=index.vector_dim,
            document_count=index.document_count,
            centroid_count=index.centroid_count,
            centroids_per_query_vector=index.config.centroids_per_query_vector,
            candidate_k=index.config.candidate_k,
            posting_count=posting_count,
            index_bytes=_tachiom_i8_index_bytes(
                doc_offsets=index.doc_offsets,
                total_vector_count=index.total_vector_count,
                vector_dim=index.vector_dim,
                centroids=index.centroids,
                centroid_offsets=centroid_offsets,
                centroid_indices=centroid_indices,
            ),
            _prepared_index=prepared_index,
        )

    @property
    def index_kind(self) -> str:
        return "token_aware_centroid_postings_i8_rerank"

    @property
    def rerank_kind(self) -> str:
        return "i8_maxsim_candidate_window"

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(normalized_queries.shape[0])))
        module = load_module()
        rows = module.tachiom_tac_i8_candidate_positions_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                self.centroids_per_query_vector,
                self.candidate_k,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        rows = module.search_tachiom_tac_i8_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                final_k,
                self.centroids_per_query_vector,
                self.candidate_k,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)


@dataclass(frozen=True, slots=True)
class TachiomTacHnswMojoIndex:
    """Mojo TAC index with a Python-built HNSW centroid graph."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    document_count: int
    centroid_count: int
    centroids_per_query_vector: int
    candidate_k: int
    candidate_pruning_alpha: float | None
    ef_search: int
    posting_count: int
    graph_edge_count: int
    index_bytes: int
    _prepared_index: Any

    @classmethod
    def from_hnsw_index(
        cls,
        index: "TachiomTacHnswIndex",
    ) -> "TachiomTacHnswMojoIndex":
        if index.vector_dim != 128:
            raise ValueError("Mojo TAC HNSW currently requires vector_dim=128")
        centroid_offsets, centroid_indices = _flatten_centroid_postings(
            index.tac_index.centroid_doc_postings
        )
        graph_layer_offsets, graph_node_offsets, graph_neighbor_indices = (
            _flatten_hnsw_layers(index.graph.layers)
        )
        module = load_module()
        prepared_index = module.prepare_tachiom_tac_hnsw_hybrid_flat_dim128(
            [
                list(index.tac_index.doc_ids),
                index.tac_index.doc_offsets.astype(
                    INDEX_OFFSET_DTYPE,
                    copy=False,
                ).tolist(),
                index.tac_index.token_values.astype(
                    VECTOR_DTYPE,
                    copy=False,
                ).reshape(-1).tolist(),
                index.tac_index.centroids.astype(
                    VECTOR_DTYPE,
                    copy=False,
                ).reshape(-1).tolist(),
                centroid_offsets.tolist(),
                centroid_indices.tolist(),
                graph_layer_offsets.tolist(),
                graph_node_offsets.tolist(),
                graph_neighbor_indices.tolist(),
                index.graph.entry_point,
            ]
        )
        posting_count = int(
            module.tachiom_tac_hnsw_prepared_posting_count(prepared_index)
        )
        if posting_count != int(centroid_indices.shape[0]):
            raise RuntimeError("Mojo TAC HNSW prepared posting count mismatch")
        graph_edge_count = int(
            module.tachiom_tac_hnsw_prepared_graph_edge_count(prepared_index)
        )
        if graph_edge_count != int(graph_neighbor_indices.shape[0]):
            raise RuntimeError("Mojo TAC HNSW prepared graph edge count mismatch")
        return cls(
            doc_ids=index.tac_index.doc_ids,
            vector_dim=index.vector_dim,
            document_count=index.document_count,
            centroid_count=index.centroid_count,
            centroids_per_query_vector=(
                index.tac_index.config.centroids_per_query_vector
            ),
            candidate_k=index.tac_index.config.candidate_k,
            candidate_pruning_alpha=index.tac_index.config.candidate_pruning_alpha,
            ef_search=index.graph.config.ef_search,
            posting_count=posting_count,
            graph_edge_count=graph_edge_count,
            index_bytes=index.index_bytes,
            _prepared_index=prepared_index,
        )

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(normalized_queries.shape[0])))
        module = load_module()
        if self.candidate_pruning_alpha is not None:
            if final_k is None:
                raise ValueError("final_k is required when candidate pruning is enabled")
            rows = module.tachiom_tac_hnsw_candidate_positions_prepared_batch_address_with_pruning(
                [
                    int(normalized_queries.ctypes.data),
                    int(normalized_queries.shape[0]),
                    int(normalized_queries.shape[1]),
                    self.centroids_per_query_vector,
                    self.candidate_k,
                    final_k,
                    self.ef_search,
                    float(self.candidate_pruning_alpha),
                    self._prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.tachiom_tac_hnsw_candidate_positions_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                self.centroids_per_query_vector,
                self.candidate_k,
                self.ef_search,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        rows = module.search_tachiom_tac_hnsw_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                final_k,
                self.centroids_per_query_vector,
                self.candidate_k,
                self.ef_search,
                float(self.candidate_pruning_alpha or 0.0),
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)


@dataclass(frozen=True, slots=True)
class TachiomResidualPqMojoIndex:
    """Mojo TAC index with centroid-id plus residual-PQ token payload."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    document_count: int
    centroid_count: int
    centroids_per_query_vector: int
    candidate_k: int
    posting_count: int
    index_bytes: int
    subspace_count: int
    effective_codebook_size: int
    candidate_pruning_alpha: float | None
    _prepared_index: Any

    @classmethod
    def from_pq_index(
        cls,
        index: "TachiomResidualPqIndex",
    ) -> "TachiomResidualPqMojoIndex":
        if index.vector_dim != 128:
            raise ValueError("Mojo TAC residual-PQ currently requires vector_dim=128")
        centroid_offsets, centroid_indices = _flatten_centroid_postings(
            index.centroid_doc_postings
        )
        module = load_module()
        prepared_index = module.prepare_tachiom_tac_pq_dim128(
            [
                list(index.doc_ids),
                index.doc_offsets.astype(INDEX_OFFSET_DTYPE, copy=False).tolist(),
                index.centroids.astype(VECTOR_DTYPE, copy=False).reshape(-1).tolist(),
                centroid_offsets.tolist(),
                centroid_indices.tolist(),
                index.token_centroid_positions.astype(
                    INDEX_OFFSET_DTYPE,
                    copy=False,
                ).tolist(),
                index.residual_norms.astype(VECTOR_DTYPE, copy=False).tolist(),
                index.pq_codes.astype(np.int64, copy=False).reshape(-1).tolist(),
                index.pq_codebooks.astype(
                    VECTOR_DTYPE,
                    copy=False,
                ).reshape(-1).tolist(),
                index.subspace_count,
                index.effective_codebook_size,
            ]
        )
        posting_count = int(
            module.tachiom_tac_pq_prepared_posting_count(prepared_index)
        )
        if posting_count != int(centroid_indices.shape[0]):
            raise RuntimeError("Mojo TAC residual-PQ prepared posting count mismatch")
        return cls(
            doc_ids=index.doc_ids,
            vector_dim=index.vector_dim,
            document_count=index.document_count,
            centroid_count=index.centroid_count,
            centroids_per_query_vector=index.tac_config_centroids_per_query_vector,
            candidate_k=index.tac_config_candidate_k,
            posting_count=posting_count,
            index_bytes=index.index_bytes,
            subspace_count=index.subspace_count,
            effective_codebook_size=index.effective_codebook_size,
            candidate_pruning_alpha=index.tac_config_candidate_pruning_alpha,
            _prepared_index=prepared_index,
        )

    @property
    def index_kind(self) -> str:
        return "token_aware_centroid_postings_residual_pq_rerank"

    @property
    def rerank_kind(self) -> str:
        return "centroid_plus_normalized_residual_pq_maxsim"

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(normalized_queries.shape[0])))
        module = load_module()
        if self.candidate_pruning_alpha is not None:
            if final_k is None:
                raise ValueError("final_k is required when candidate pruning is enabled")
            rows = module.tachiom_tac_pq_candidate_positions_prepared_batch_address_with_pruning(
                [
                    int(normalized_queries.ctypes.data),
                    int(normalized_queries.shape[0]),
                    int(normalized_queries.shape[1]),
                    self.centroids_per_query_vector,
                    self.candidate_k,
                    final_k,
                    float(self.candidate_pruning_alpha),
                    self._prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.tachiom_tac_pq_candidate_positions_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                self.centroids_per_query_vector,
                self.candidate_k,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        if self.candidate_pruning_alpha is not None:
            rows = module.search_tachiom_tac_pq_prepared_batch_address_with_pruning(
                [
                    int(normalized_queries.ctypes.data),
                    int(normalized_queries.shape[0]),
                    int(normalized_queries.shape[1]),
                    final_k,
                    self.centroids_per_query_vector,
                    self.candidate_k,
                    float(self.candidate_pruning_alpha),
                    self._prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.search_tachiom_tac_pq_prepared_batch_address(
            [
                int(normalized_queries.ctypes.data),
                int(normalized_queries.shape[0]),
                int(normalized_queries.shape[1]),
                final_k,
                self.centroids_per_query_vector,
                self.candidate_k,
                self._prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)


def _flatten_centroid_postings(
    postings: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.empty(len(postings) + 1, dtype=INDEX_OFFSET_DTYPE)
    offsets[0] = 0
    running = 0
    rows: list[np.ndarray] = []
    for index, posting in enumerate(postings, start=1):
        normalized = np.asarray(posting, dtype=INDEX_OFFSET_DTYPE)
        rows.append(normalized)
        running += int(normalized.shape[0])
        offsets[index] = running
    indices = (
        np.concatenate(rows).astype(INDEX_OFFSET_DTYPE, copy=False)
        if rows
        else np.asarray([], dtype=INDEX_OFFSET_DTYPE)
    )
    return offsets, np.ascontiguousarray(indices, dtype=INDEX_OFFSET_DTYPE)


def _flatten_hnsw_layers(
    layers: tuple[tuple[tuple[int, ...], ...], ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layer_offsets = np.empty(len(layers), dtype=INDEX_OFFSET_DTYPE)
    node_offsets: list[int] = []
    neighbor_indices: list[int] = []
    for layer_index, layer in enumerate(layers):
        layer_offsets[layer_index] = len(node_offsets)
        node_offsets.append(len(neighbor_indices))
        for neighbors in layer:
            neighbor_indices.extend(int(neighbor) for neighbor in neighbors)
            node_offsets.append(len(neighbor_indices))
    return (
        np.ascontiguousarray(layer_offsets, dtype=INDEX_OFFSET_DTYPE),
        np.ascontiguousarray(node_offsets, dtype=INDEX_OFFSET_DTYPE),
        np.ascontiguousarray(neighbor_indices, dtype=INDEX_OFFSET_DTYPE),
    )


def _profile_pairs_to_dict(row: Any) -> dict[str, Any]:
    return {str(key): value for key, value in row}


def _tachiom_i8_index_bytes(
    *,
    doc_offsets: np.ndarray,
    total_vector_count: int,
    vector_dim: int,
    centroids: np.ndarray,
    centroid_offsets: np.ndarray,
    centroid_indices: np.ndarray,
) -> int:
    token_code_bytes = total_vector_count * vector_dim * np.dtype(np.int8).itemsize
    token_scale_bytes = total_vector_count * np.dtype(VECTOR_DTYPE).itemsize
    return int(
        doc_offsets.nbytes
        + token_code_bytes
        + token_scale_bytes
        + centroids.nbytes
        + centroid_offsets.nbytes
        + centroid_indices.nbytes
    )
