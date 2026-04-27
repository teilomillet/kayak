"""Mojo-backed PLAID-style centroid-posting search.

This module owns Python orchestration for Kayak's optional approximation lane.
It does not implement candidate generation or reranking itself: centroid
sampling, token assignment, document candidate scoring, exact MaxSim rerank,
and i8 score-proxy rerank are executed by the Mojo bridge so speed-track
measurements exercise the same systems layer as the exact backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

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
I8_BYTES = 1
PLAID_PAYLOAD_EXACT = "exact"
PLAID_PAYLOAD_I8 = "i8"
PLAID_PAYLOADS = (PLAID_PAYLOAD_EXACT, PLAID_PAYLOAD_I8)


@dataclass(frozen=True, slots=True)
class KayakPlaidApproxConfig:
    """Explicit sampled-centroid approximation knobs for Mojo PLAID search."""

    centroid_count: int = 128
    centroids_per_query_vector: int = 4
    candidate_k: int = 10
    payload: str = PLAID_PAYLOAD_EXACT

    def validate(self, *, final_k: int) -> None:
        _require_positive("centroid_count", self.centroid_count)
        _require_positive(
            "centroids_per_query_vector",
            self.centroids_per_query_vector,
        )
        _require_positive("candidate_k", self.candidate_k)
        if self.candidate_k < final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")
        if self.payload not in PLAID_PAYLOADS:
            raise ValueError("payload must be one of: exact, i8")


@dataclass(frozen=True, slots=True)
class KayakPlaidI8PayloadSnapshot:
    """Flat dim128 i8 payload exported for benchmark-only GPU probes."""

    doc_offsets: np.ndarray
    token_codes: np.ndarray
    token_scales: np.ndarray
    centroid_token_indices: np.ndarray
    centroid_doc_offsets: np.ndarray
    centroid_doc_indices: np.ndarray
    document_count: int
    total_vector_count: int
    vector_dim: int

    def validate(self) -> None:
        if self.vector_dim != VECTOR_DIM:
            raise ValueError("i8 payload snapshot requires vector_dim=128")
        if self.document_count <= 0:
            raise ValueError("document_count must be positive")
        if self.total_vector_count <= 0:
            raise ValueError("total_vector_count must be positive")
        if self.doc_offsets.shape != (self.document_count + 1,):
            raise ValueError("doc_offsets shape must be document_count + 1")
        if self.token_codes.shape != (self.total_vector_count * VECTOR_DIM,):
            raise ValueError("token_codes shape must be total_vector_count * 128")
        if self.token_scales.shape != (self.total_vector_count,):
            raise ValueError("token_scales shape must be total_vector_count")
        if self.centroid_token_indices.ndim != 1:
            raise ValueError("centroid_token_indices must be a flat array")
        if self.centroid_doc_offsets.shape != (
            int(self.centroid_token_indices.shape[0]) + 1,
        ):
            raise ValueError("centroid_doc_offsets shape must be centroid_count + 1")
        if self.centroid_doc_indices.ndim != 1:
            raise ValueError("centroid_doc_indices must be a flat array")

    @property
    def centroid_count(self) -> int:
        return int(self.centroid_token_indices.shape[0])

    @property
    def posting_count(self) -> int:
        return int(self.centroid_doc_indices.shape[0])

    def candidate_generation_byte_counts(self) -> dict[str, int]:
        return {
            "centroid_token_indices": int(self.centroid_token_indices.nbytes),
            "centroid_doc_offsets": int(self.centroid_doc_offsets.nbytes),
            "centroid_doc_indices": int(self.centroid_doc_indices.nbytes),
        }


@dataclass(frozen=True, slots=True)
class KayakPlaidI8SelectedCentroids:
    """Selected centroid ids and scores for benchmark-only GPU probes."""

    positions_by_query: tuple[tuple[int, ...], ...]
    scores_by_query: tuple[tuple[float, ...], ...]
    query_count: int
    query_vector_count: int
    centroids_per_query_vector: int

    def validate(self) -> None:
        if self.query_count <= 0:
            raise ValueError("query_count must be positive")
        if self.query_vector_count <= 0:
            raise ValueError("query_vector_count must be positive")
        if self.centroids_per_query_vector <= 0:
            raise ValueError("centroids_per_query_vector must be positive")
        if len(self.positions_by_query) != self.query_count:
            raise ValueError("selected centroid position rows must match query_count")
        if len(self.scores_by_query) != self.query_count:
            raise ValueError("selected centroid score rows must match query_count")
        expected = self.selected_centroid_count_per_query
        for positions, scores in zip(self.positions_by_query, self.scores_by_query):
            if len(positions) != expected:
                raise ValueError(
                    "selected centroid positions must match query vectors * budget"
                )
            if len(scores) != expected:
                raise ValueError(
                    "selected centroid scores must match query vectors * budget"
                )

    @property
    def selected_centroid_count_per_query(self) -> int:
        return self.query_vector_count * self.centroids_per_query_vector

    @property
    def selected_centroid_count_total(self) -> int:
        return self.query_count * self.selected_centroid_count_per_query

    def positions_array(self) -> np.ndarray:
        return np.ascontiguousarray(self.positions_by_query, dtype=INDEX_OFFSET_DTYPE)

    def scores_array(self) -> np.ndarray:
        return np.ascontiguousarray(self.scores_by_query, dtype=VECTOR_DTYPE)


@dataclass(frozen=True, slots=True)
class KayakPlaidApproxIndex:
    """Prepared Mojo PLAID approximation index with explicit payload semantics."""

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
        if config.payload == PLAID_PAYLOAD_I8:
            prepared_index = module.prepare_plaid_approx_i8_hybrid_flat_dim128(
                list(doc_ids),
                doc_offsets.tolist(),
                token_values,
                centroid_count,
            )
            posting_count = int(
                module.plaid_approx_i8_prepared_posting_count(prepared_index)
            )
            index_bytes = _prepared_i8_index_bytes(
                doc_offsets=doc_offsets,
                token_value_count=len(token_values),
                centroid_count=centroid_count,
                posting_count=posting_count,
            )
        else:
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

    @property
    def index_kind(self) -> str:
        if self.config.payload == PLAID_PAYLOAD_I8:
            return "sampled_centroid_postings_i8_proxy"
        return "sampled_centroid_postings_exact_rerank"

    @property
    def rerank_kind(self) -> str:
        if self.config.payload == PLAID_PAYLOAD_I8:
            return "i8_maxsim_candidate_window"
        return "exact_maxsim_candidate_window"

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
        search_function = (
            module.search_plaid_approx_i8_prepared_batch
            if self.config.payload == PLAID_PAYLOAD_I8
            else module.search_plaid_approx_prepared_batch
        )
        rows = search_function(
            query_values,
            final_k,
            self.config.centroids_per_query_vector,
            self.config.candidate_k,
            self._prepared_index,
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def i8_candidate_positions_batch(
        self,
        queries: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        return self._i8_candidate_positions_batch(
            queries,
            unordered=False,
            positive_centroids=False,
        )

    def i8_candidate_positions_batch_unordered(
        self,
        queries: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the same candidate set without approximate-score ordering."""
        return self._i8_candidate_positions_batch(
            queries,
            unordered=True,
            positive_centroids=False,
        )

    def i8_candidate_positions_batch_positive_centroids_unordered(
        self,
        queries: np.ndarray,
    ) -> tuple[tuple[int, ...], ...]:
        """Return unordered candidates using positive selected centroids only."""
        return self._i8_candidate_positions_batch(
            queries,
            unordered=True,
            positive_centroids=True,
        )

    def _i8_candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        unordered: bool,
        positive_centroids: bool,
    ) -> tuple[tuple[int, ...], ...]:
        if self.config.payload != PLAID_PAYLOAD_I8:
            raise ValueError("i8 candidate positions require payload='i8'")
        if positive_centroids and not unordered:
            raise ValueError("positive-centroid candidates are unordered only")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.config.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(
                full_window for _ in range(int(normalized_queries.shape[0]))
            )
        module = load_module()
        if positive_centroids:
            candidate_function = getattr(
                module,
                "plaid_i8_positive_centroid_candidate_positions_prepared_batch_address_unordered",
            )
        elif unordered:
            candidate_function = (
                module.plaid_i8_candidate_positions_prepared_batch_address_unordered
            )
        else:
            candidate_function = (
                module.plaid_i8_candidate_positions_prepared_batch_address
            )
        rows = candidate_function(
            int(normalized_queries.ctypes.data),
            int(normalized_queries.shape[0]),
            int(normalized_queries.shape[1]),
            self.config.centroids_per_query_vector,
            self.config.candidate_k,
            self._prepared_index,
        )
        return tuple(tuple(int(position) for position in row) for row in rows)

    def i8_score_candidate_positions_batch(
        self,
        queries: np.ndarray,
        candidate_positions_by_query: Sequence[Sequence[int]],
    ) -> tuple[tuple[float, ...], ...]:
        if self.config.payload != PLAID_PAYLOAD_I8:
            raise ValueError("i8 candidate scores require payload='i8'")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if len(candidate_positions_by_query) != int(normalized_queries.shape[0]):
            raise ValueError("query count must match candidate position row count")
        query_values = _flat_query_values_by_query(normalized_queries)
        candidate_rows = [
            [int(position) for position in positions]
            for positions in candidate_positions_by_query
        ]
        module = load_module()
        rows = module.plaid_i8_candidate_scores_prepared_batch(
            query_values,
            candidate_rows,
            self._prepared_index,
        )
        return tuple(tuple(float(score) for score in row) for row in rows)

    def i8_selected_centroids_batch(
        self,
        queries: np.ndarray,
        *,
        centroids_per_query_vector: int | None = None,
    ) -> KayakPlaidI8SelectedCentroids:
        if self.config.payload != PLAID_PAYLOAD_I8:
            raise ValueError("i8 selected centroids require payload='i8'")
        normalized_queries = _as_query_tensor(queries, vector_dim=self.vector_dim)
        centroid_budget = (
            self.config.centroids_per_query_vector
            if centroids_per_query_vector is None
            else int(centroids_per_query_vector)
        )
        _require_positive("centroids_per_query_vector", centroid_budget)
        module = load_module()
        rows = module.plaid_i8_selected_centroids_prepared_batch_address(
            int(normalized_queries.ctypes.data),
            int(normalized_queries.shape[0]),
            int(normalized_queries.shape[1]),
            centroid_budget,
            self._prepared_index,
        )
        if len(rows) != 2:
            raise RuntimeError("selected centroid export returned invalid shape")
        selected = KayakPlaidI8SelectedCentroids(
            positions_by_query=tuple(
                tuple(int(position) for position in row) for row in rows[0]
            ),
            scores_by_query=tuple(
                tuple(float(score) for score in row) for row in rows[1]
            ),
            query_count=int(normalized_queries.shape[0]),
            query_vector_count=int(normalized_queries.shape[1]),
            centroids_per_query_vector=centroid_budget,
        )
        selected.validate()
        return selected

    def i8_payload_snapshot(self) -> KayakPlaidI8PayloadSnapshot:
        if self.config.payload != PLAID_PAYLOAD_I8:
            raise ValueError("i8 payload snapshot requires payload='i8'")
        module = load_module()
        snapshot = KayakPlaidI8PayloadSnapshot(
            doc_offsets=np.asarray(
                module.plaid_i8_prepared_doc_offsets(self._prepared_index),
                dtype=INDEX_OFFSET_DTYPE,
            ),
            token_codes=np.asarray(
                module.plaid_i8_prepared_token_codes(self._prepared_index),
                dtype=np.int8,
            ),
            token_scales=np.asarray(
                module.plaid_i8_prepared_token_scales(self._prepared_index),
                dtype=VECTOR_DTYPE,
            ),
            centroid_token_indices=np.asarray(
                module.plaid_i8_prepared_centroid_token_indices(
                    self._prepared_index
                ),
                dtype=INDEX_OFFSET_DTYPE,
            ),
            centroid_doc_offsets=np.asarray(
                module.plaid_i8_prepared_centroid_doc_offsets(
                    self._prepared_index
                ),
                dtype=INDEX_OFFSET_DTYPE,
            ),
            centroid_doc_indices=np.asarray(
                module.plaid_i8_prepared_centroid_doc_indices(
                    self._prepared_index
                ),
                dtype=INDEX_OFFSET_DTYPE,
            ),
            document_count=self.document_count,
            total_vector_count=sum(self.document_vector_counts),
            vector_dim=self.vector_dim,
        )
        snapshot.validate()
        return snapshot

    def search(
        self,
        query: "LateQuery",
        *,
        final_k: int,
    ) -> tuple[SearchHit, ...]:
        """Return approximate hits for one public query."""
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
        """Return approximate hits for a public query batch."""
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        query_values = [
            _flat_query_values_from_query(query, vector_dim=self.vector_dim)
            for query in query_batch.queries
        ]
        module = load_module()
        search_function = (
            module.search_plaid_approx_i8_prepared_hits_batch
            if self.config.payload == PLAID_PAYLOAD_I8
            else module.search_plaid_approx_prepared_hits_batch
        )
        rows = search_function(
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


def _prepared_i8_index_bytes(
    *,
    doc_offsets: np.ndarray,
    token_value_count: int,
    centroid_count: int,
    posting_count: int,
) -> int:
    token_count = token_value_count // VECTOR_DIM
    token_code_bytes = token_value_count * I8_BYTES
    token_scale_bytes = token_count * np.dtype(VECTOR_DTYPE).itemsize
    centroid_token_index_bytes = centroid_count * INT_BYTES
    centroid_doc_offset_bytes = (centroid_count + 1) * INT_BYTES
    centroid_doc_index_bytes = posting_count * INT_BYTES
    return int(
        doc_offsets.nbytes
        + token_code_bytes
        + token_scale_bytes
        + centroid_token_index_bytes
        + centroid_doc_offset_bytes
        + centroid_doc_index_bytes
    )


PlaidApproxConfig = KayakPlaidApproxConfig
PlaidApproxIndex = KayakPlaidApproxIndex
