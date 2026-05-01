"""Query-time reader for on-disk streaming TAC/PQ artifacts.

This module owns exact-centroid candidate generation plus residual-PQ rerank
over the files written by ``tachiom_streaming_index``. It does not own HNSW
centroid traversal; that remains a separate paper-speed gap.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .mojo_exact_cpu import load_module
from .tachiom_arrays import _as_query_tensor, _top_positions
from .tachiom_candidates import ranked_candidate_positions_from_scores
from .tachiom_pq import _residual_distance_tables
from .tachiom_streaming_hnsw import (
    StreamingTachiomHnswGraph,
    load_streaming_tachiom_hnsw_graph,
)


@dataclass(frozen=True, slots=True)
class StreamingTachiomPqIndex:
    """Memory-mapped TAC centroid postings with residual-PQ document payload."""

    doc_ids: tuple[str, ...]
    doc_offsets: np.ndarray
    centroids: np.memmap
    centroid_token_ids: np.memmap
    centroid_doc_posting_offsets: np.ndarray
    centroid_doc_postings: np.memmap
    token_centroid_positions: np.memmap
    residual_norms: np.memmap
    pq_codes: np.memmap
    pq_codebooks: np.memmap
    candidate_k: int
    centroids_per_query_vector: int
    candidate_pruning_alpha: float | None
    index_bytes: int
    manifest: Mapping[str, Any]

    @property
    def document_count(self) -> int:
        return len(self.doc_ids)

    @property
    def total_vector_count(self) -> int:
        return int(self.token_centroid_positions.shape[0])

    @property
    def vector_dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def centroid_count(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def posting_count(self) -> int:
        return int(self.centroid_doc_postings.shape[0])

    @property
    def subspace_count(self) -> int:
        return int(self.pq_codebooks.shape[0])

    @property
    def index_kind(self) -> str:
        return "streaming_token_aware_centroid_postings_residual_pq_rerank"

    @property
    def rerank_kind(self) -> str:
        return "centroid_plus_normalized_residual_pq_maxsim"

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

    def _candidate_positions_for_query(
        self,
        query: np.ndarray,
        *,
        final_k: int | None,
    ) -> tuple[int, ...]:
        if self.candidate_k >= self.document_count:
            return tuple(range(self.document_count))

        centroid_scores = np.matmul(query, self.centroids.T)
        selected_per_query_vector = min(
            self.centroids_per_query_vector,
            self.centroid_count,
        )
        doc_scores = np.zeros(self.document_count, dtype=VECTOR_DTYPE)
        touched = np.zeros(self.document_count, dtype=bool)

        for query_token_index in range(query.shape[0]):
            scores = centroid_scores[query_token_index]
            centroid_positions = _top_positions(scores, selected_per_query_vector)
            self._accumulate_selected_postings(
                centroid_positions=centroid_positions,
                selected_scores=scores[centroid_positions],
                doc_scores=doc_scores,
                touched=touched,
            )

        candidate_count = min(self.candidate_k, self.document_count)
        if not np.any(touched):
            return tuple(range(candidate_count))
        ranked_scores = doc_scores.copy()
        ranked_scores[~touched] = np.float32(-3.4e38)
        return ranked_candidate_positions_from_scores(
            ranked_scores,
            candidate_k=candidate_count,
            final_k=final_k,
            candidate_pruning_alpha=self.candidate_pruning_alpha,
        )

    def _posting_for_centroid(self, centroid_position: int) -> np.ndarray:
        start = int(self.centroid_doc_posting_offsets[centroid_position])
        stop = int(self.centroid_doc_posting_offsets[centroid_position + 1])
        return self.centroid_doc_postings[start:stop]

    def _accumulate_selected_postings(
        self,
        *,
        centroid_positions: Sequence[int] | np.ndarray,
        selected_scores: np.ndarray,
        doc_scores: np.ndarray,
        touched: np.ndarray,
    ) -> None:
        positions = np.asarray(centroid_positions, dtype=np.int64)
        if positions.size == 0:
            return
        scores = np.asarray(selected_scores, dtype=VECTOR_DTYPE)
        starts = self.centroid_doc_posting_offsets[positions]
        stops = self.centroid_doc_posting_offsets[positions + 1]
        lengths = stops - starts
        expanded_count = int(lengths.sum())
        if expanded_count == 0:
            return

        posting_docs = np.empty(expanded_count, dtype=self.centroid_doc_postings.dtype)
        posting_scores = np.empty(expanded_count, dtype=VECTOR_DTYPE)
        cursor = 0
        for score, start, length in zip(scores, starts, lengths, strict=True):
            length_int = int(length)
            if length_int == 0:
                continue
            stop = cursor + length_int
            posting_docs[cursor:stop] = self.centroid_doc_postings[
                int(start) : int(start) + length_int
            ]
            posting_scores[cursor:stop] = score
            cursor = stop

        best_for_query_token = np.full(
            self.document_count,
            -np.inf,
            dtype=VECTOR_DTYPE,
        )
        np.maximum.at(best_for_query_token, posting_docs, posting_scores)
        touched[posting_docs] = True
        matched = np.isfinite(best_for_query_token)
        doc_scores[matched] += best_for_query_token[matched]

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


def load_streaming_tachiom_pq_index(index_root: Path) -> StreamingTachiomPqIndex:
    manifest_path = index_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"streaming index manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("pq_enabled", False):
        raise ValueError("streaming search currently requires a PQ-enabled index")

    document_count = int(manifest["document_count"])
    vector_count = int(manifest["document_vector_count"])
    vector_dim = int(manifest["vector_dim"])
    centroid_count = int(manifest["centroid_count"])
    posting_count = int(manifest["posting_count"])
    tac_config = manifest["config"]["tac"]
    pq_config = manifest["config"]["pq"]
    subspace_count = int(pq_config["subspace_count"])
    subspace_dim = vector_dim // subspace_count
    codebook_size = _effective_codebook_size(
        index_root / "pq_codebooks.f32",
        subspace_count=subspace_count,
        subspace_dim=subspace_dim,
    )

    doc_ids = tuple(
        (index_root / "doc_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    if len(doc_ids) != document_count:
        raise ValueError("doc_ids.txt does not match streaming index manifest")

    doc_offsets = np.load(index_root / "doc_offsets.u64.npy").astype(
        INDEX_OFFSET_DTYPE,
        copy=False,
    )
    centroid_doc_posting_offsets = np.load(
        index_root / "centroid_doc_posting_offsets.u64.npy"
    ).astype(INDEX_OFFSET_DTYPE, copy=False)
    if doc_offsets.shape != (document_count + 1,):
        raise ValueError("doc_offsets shape does not match streaming index manifest")
    if int(doc_offsets[-1]) != vector_count:
        raise ValueError("doc_offsets final value does not match vector count")
    if centroid_doc_posting_offsets.shape != (centroid_count + 1,):
        raise ValueError("posting offsets shape does not match centroid count")
    if int(centroid_doc_posting_offsets[-1]) != posting_count:
        raise ValueError("posting offsets final value does not match posting count")

    return StreamingTachiomPqIndex(
        doc_ids=doc_ids,
        doc_offsets=doc_offsets,
        centroids=np.memmap(
            index_root / "centroids.f32",
            dtype=VECTOR_DTYPE,
            mode="r",
            shape=(centroid_count, vector_dim),
        ),
        centroid_token_ids=np.memmap(
            index_root / "centroid_token_ids.i64",
            dtype=np.int64,
            mode="r",
            shape=(centroid_count,),
        ),
        centroid_doc_posting_offsets=centroid_doc_posting_offsets,
        centroid_doc_postings=np.memmap(
            index_root / "centroid_doc_postings.u32",
            dtype=np.uint32,
            mode="r",
            shape=(posting_count,),
        ),
        token_centroid_positions=np.memmap(
            index_root / "token_centroid_positions.u32",
            dtype=np.uint32,
            mode="r",
            shape=(vector_count,),
        ),
        residual_norms=np.memmap(
            index_root / "residual_norms.f32",
            dtype=VECTOR_DTYPE,
            mode="r",
            shape=(vector_count,),
        ),
        pq_codes=np.memmap(
            index_root / "pq_codes.u8",
            dtype=np.uint8,
            mode="r",
            shape=(vector_count, subspace_count),
        ),
        pq_codebooks=np.memmap(
            index_root / "pq_codebooks.f32",
            dtype=VECTOR_DTYPE,
            mode="r",
            shape=(subspace_count, codebook_size, subspace_dim),
        ),
        candidate_k=int(tac_config["candidate_k"]),
        centroids_per_query_vector=int(tac_config["centroids_per_query_vector"]),
        candidate_pruning_alpha=tac_config.get("candidate_pruning_alpha"),
        index_bytes=int(manifest["index_payload_bytes"]),
        manifest=manifest,
    )


@dataclass(frozen=True, slots=True)
class StreamingTachiomPqMojoIndex:
    """Prepared Mojo search over a streaming TAC/PQ artifact."""

    base_index: StreamingTachiomPqIndex
    prepared_index: Any

    @classmethod
    def from_base_index(
        cls,
        base_index: StreamingTachiomPqIndex,
    ) -> "StreamingTachiomPqMojoIndex":
        if base_index.vector_dim != 128:
            raise ValueError("Mojo streaming TAC/PQ currently requires vector_dim=128")
        module = load_module()
        prepared_index = module.prepare_tachiom_tac_pq_dim128(
            [
                list(base_index.doc_ids),
                base_index.doc_offsets.astype(INDEX_OFFSET_DTYPE, copy=False).tolist(),
                base_index.centroids.astype(VECTOR_DTYPE, copy=False)
                .reshape(-1)
                .tolist(),
                base_index.centroid_doc_posting_offsets.astype(
                    INDEX_OFFSET_DTYPE,
                    copy=False,
                ).tolist(),
                base_index.centroid_doc_postings.astype(
                    INDEX_OFFSET_DTYPE,
                    copy=False,
                ).tolist(),
                base_index.token_centroid_positions.astype(
                    INDEX_OFFSET_DTYPE,
                    copy=False,
                ).tolist(),
                base_index.residual_norms.astype(VECTOR_DTYPE, copy=False).tolist(),
                base_index.pq_codes.astype(np.int64, copy=False).reshape(-1).tolist(),
                base_index.pq_codebooks.astype(VECTOR_DTYPE, copy=False)
                .reshape(-1)
                .tolist(),
                base_index.subspace_count,
                int(base_index.pq_codebooks.shape[1]),
            ]
        )
        posting_count = int(module.tachiom_tac_pq_prepared_posting_count(prepared_index))
        if posting_count != base_index.posting_count:
            raise RuntimeError("Mojo streaming TAC/PQ posting count mismatch")
        return cls(base_index=base_index, prepared_index=prepared_index)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.base_index.doc_ids

    @property
    def document_count(self) -> int:
        return self.base_index.document_count

    @property
    def total_vector_count(self) -> int:
        return self.base_index.total_vector_count

    @property
    def vector_dim(self) -> int:
        return self.base_index.vector_dim

    @property
    def centroid_count(self) -> int:
        return self.base_index.centroid_count

    @property
    def posting_count(self) -> int:
        return self.base_index.posting_count

    @property
    def index_bytes(self) -> int:
        return self.base_index.index_bytes

    @property
    def index_kind(self) -> str:
        return "streaming_token_aware_centroid_postings_residual_pq_rerank_mojo"

    @property
    def rerank_kind(self) -> str:
        return self.base_index.rerank_kind

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.base_index.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(query_tensor.shape[0])))
        module = load_module()
        if self.base_index.candidate_pruning_alpha is not None:
            if final_k is None:
                raise ValueError("final_k is required when candidate pruning is enabled")
            rows = module.tachiom_tac_pq_candidate_positions_prepared_batch_address_with_pruning(
                [
                    int(query_tensor.ctypes.data),
                    int(query_tensor.shape[0]),
                    int(query_tensor.shape[1]),
                    self.base_index.centroids_per_query_vector,
                    self.base_index.candidate_k,
                    final_k,
                    float(self.base_index.candidate_pruning_alpha),
                    self.prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.tachiom_tac_pq_candidate_positions_prepared_batch_address(
            [
                int(query_tensor.ctypes.data),
                int(query_tensor.shape[0]),
                int(query_tensor.shape[1]),
                self.base_index.centroids_per_query_vector,
                self.base_index.candidate_k,
                self.prepared_index,
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
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        if self.base_index.candidate_pruning_alpha is not None:
            rows = module.search_tachiom_tac_pq_prepared_batch_address_with_pruning(
                [
                    int(query_tensor.ctypes.data),
                    int(query_tensor.shape[0]),
                    int(query_tensor.shape[1]),
                    final_k,
                    self.base_index.centroids_per_query_vector,
                    self.base_index.candidate_k,
                    float(self.base_index.candidate_pruning_alpha),
                    self.prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        rows = module.search_tachiom_tac_pq_prepared_batch_address(
            [
                int(query_tensor.ctypes.data),
                int(query_tensor.shape[0]),
                int(query_tensor.shape[1]),
                final_k,
                self.base_index.centroids_per_query_vector,
                self.base_index.candidate_k,
                self.prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)


def load_streaming_tachiom_pq_mojo_index(index_root: Path) -> StreamingTachiomPqMojoIndex:
    return StreamingTachiomPqMojoIndex.from_base_index(
        load_streaming_tachiom_pq_index(index_root)
    )


@dataclass(frozen=True, slots=True)
class StreamingTachiomHnswPqMojoIndex:
    """Prepared Mojo search over streaming TAC/PQ plus persisted HNSW graph."""

    base_index: StreamingTachiomPqIndex
    graph: StreamingTachiomHnswGraph
    prepared_index: Any
    address_backed: bool = False

    @classmethod
    def from_components(
        cls,
        *,
        base_index: StreamingTachiomPqIndex,
        graph: StreamingTachiomHnswGraph,
        address_backed: bool = False,
    ) -> "StreamingTachiomHnswPqMojoIndex":
        if base_index.vector_dim != 128:
            raise ValueError("Mojo streaming TAC/HNSW/PQ currently requires vector_dim=128")
        if graph.centroid_count != base_index.centroid_count:
            raise ValueError("HNSW graph centroid count does not match streaming index")
        module = load_module()
        if address_backed:
            prepared_index = module.prepare_tachiom_tac_hnsw_pq_dim128_address(
                [
                    int(base_index.doc_offsets.ctypes.data),
                    base_index.document_count,
                    base_index.total_vector_count,
                    int(base_index.centroids.ctypes.data),
                    base_index.centroid_count,
                    int(base_index.centroid_doc_posting_offsets.ctypes.data),
                    int(base_index.centroid_doc_postings.ctypes.data),
                    base_index.posting_count,
                    int(base_index.token_centroid_positions.ctypes.data),
                    int(base_index.residual_norms.ctypes.data),
                    int(base_index.pq_codes.ctypes.data),
                    int(base_index.pq_codebooks.ctypes.data),
                    base_index.subspace_count,
                    int(base_index.pq_codebooks.shape[1]),
                    int(graph.layer_node_offset_offsets.ctypes.data),
                    int(graph.graph_node_offsets.ctypes.data),
                    int(graph.graph_neighbor_indices.ctypes.data),
                    graph.level_count,
                    int(graph.graph_neighbor_indices.shape[0]),
                    graph.entry_point,
                ]
            )
            posting_count = int(
                module.tachiom_tac_hnsw_pq_address_prepared_posting_count(
                    prepared_index
                )
            )
            graph_edge_count = int(
                module.tachiom_tac_hnsw_pq_address_prepared_graph_edge_count(
                    prepared_index
                )
            )
        else:
            prepared_index = module.prepare_tachiom_tac_hnsw_pq_dim128(
                [
                    list(base_index.doc_ids),
                    base_index.doc_offsets.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    base_index.centroids.astype(VECTOR_DTYPE, copy=False)
                    .reshape(-1)
                    .tolist(),
                    base_index.centroid_doc_posting_offsets.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    base_index.centroid_doc_postings.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    base_index.token_centroid_positions.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    base_index.residual_norms.astype(
                        VECTOR_DTYPE,
                        copy=False,
                    ).tolist(),
                    base_index.pq_codes.astype(
                        np.int64,
                        copy=False,
                    ).reshape(-1).tolist(),
                    base_index.pq_codebooks.astype(VECTOR_DTYPE, copy=False)
                    .reshape(-1)
                    .tolist(),
                    base_index.subspace_count,
                    int(base_index.pq_codebooks.shape[1]),
                    graph.layer_node_offset_offsets.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    graph.graph_node_offsets.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    graph.graph_neighbor_indices.astype(
                        INDEX_OFFSET_DTYPE,
                        copy=False,
                    ).tolist(),
                    graph.entry_point,
                ]
            )
            posting_count = int(
                module.tachiom_tac_hnsw_pq_prepared_posting_count(prepared_index)
            )
            graph_edge_count = int(
                module.tachiom_tac_hnsw_pq_prepared_graph_edge_count(prepared_index)
            )
        if posting_count != base_index.posting_count:
            raise RuntimeError("Mojo streaming TAC/HNSW/PQ posting count mismatch")
        if graph_edge_count != int(graph.graph_neighbor_indices.shape[0]):
            raise RuntimeError("Mojo streaming TAC/HNSW/PQ graph edge count mismatch")
        return cls(
            base_index=base_index,
            graph=graph,
            prepared_index=prepared_index,
            address_backed=address_backed,
        )

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.base_index.doc_ids

    @property
    def document_count(self) -> int:
        return self.base_index.document_count

    @property
    def total_vector_count(self) -> int:
        return self.base_index.total_vector_count

    @property
    def vector_dim(self) -> int:
        return self.base_index.vector_dim

    @property
    def centroid_count(self) -> int:
        return self.base_index.centroid_count

    @property
    def posting_count(self) -> int:
        return self.base_index.posting_count

    @property
    def index_bytes(self) -> int:
        return self.base_index.index_bytes + self.graph.graph_bytes

    @property
    def index_kind(self) -> str:
        if self.address_backed:
            return "streaming_token_aware_hnsw_centroid_postings_residual_pq_sparse_rerank_mojo_address"
        return "streaming_token_aware_hnsw_centroid_postings_residual_pq_sparse_rerank_mojo"

    @property
    def rerank_kind(self) -> str:
        return "centroid_plus_normalized_residual_pq_sparse_candidate_window_maxsim"

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        if self.base_index.candidate_k >= self.document_count:
            full_window = tuple(range(self.document_count))
            return tuple(full_window for _ in range(int(query_tensor.shape[0])))
        module = load_module()
        if self.base_index.candidate_pruning_alpha is not None:
            if final_k is None:
                raise ValueError("final_k is required when candidate pruning is enabled")
            function = (
                module.tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address_with_pruning
                if self.address_backed
                else module.tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address_with_pruning
            )
            rows = function(
                [
                    int(query_tensor.ctypes.data),
                    int(query_tensor.shape[0]),
                    int(query_tensor.shape[1]),
                    self.base_index.centroids_per_query_vector,
                    self.base_index.candidate_k,
                    final_k,
                    self.graph.ef_search,
                    float(self.base_index.candidate_pruning_alpha),
                    self.prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        function = (
            module.tachiom_tac_hnsw_pq_address_candidate_positions_prepared_batch_address
            if self.address_backed
            else module.tachiom_tac_hnsw_pq_candidate_positions_prepared_batch_address
        )
        rows = function(
            [
                int(query_tensor.ctypes.data),
                int(query_tensor.shape[0]),
                int(query_tensor.shape[1]),
                self.base_index.centroids_per_query_vector,
                self.base_index.candidate_k,
                self.graph.ef_search,
                self.prepared_index,
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
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        module = load_module()
        if self.base_index.candidate_pruning_alpha is not None:
            function = (
                module.search_tachiom_tac_hnsw_pq_address_prepared_batch_address_with_pruning
                if self.address_backed
                else module.search_tachiom_tac_hnsw_pq_prepared_batch_address_with_pruning
            )
            rows = function(
                [
                    int(query_tensor.ctypes.data),
                    int(query_tensor.shape[0]),
                    int(query_tensor.shape[1]),
                    final_k,
                    self.base_index.centroids_per_query_vector,
                    self.base_index.candidate_k,
                    self.graph.ef_search,
                    float(self.base_index.candidate_pruning_alpha),
                    self.prepared_index,
                ]
            )
            return tuple(tuple(int(position) for position in row) for row in rows)
        function = (
            module.search_tachiom_tac_hnsw_pq_address_prepared_batch_address
            if self.address_backed
            else module.search_tachiom_tac_hnsw_pq_prepared_batch_address
        )
        rows = function(
            [
                int(query_tensor.ctypes.data),
                int(query_tensor.shape[0]),
                int(query_tensor.shape[1]),
                final_k,
                self.base_index.centroids_per_query_vector,
                self.base_index.candidate_k,
                self.graph.ef_search,
                self.prepared_index,
            ]
        )
        return tuple(tuple(int(position) for position in row) for row in rows)


def load_streaming_tachiom_hnsw_pq_mojo_index(
    index_root: Path,
    *,
    graph_root: Path | None = None,
) -> StreamingTachiomHnswPqMojoIndex:
    return StreamingTachiomHnswPqMojoIndex.from_components(
        base_index=load_streaming_tachiom_pq_index(index_root),
        graph=load_streaming_tachiom_hnsw_graph(
            index_root=index_root,
            graph_root=graph_root,
        ),
    )


def load_streaming_tachiom_hnsw_pq_mojo_address_index(
    index_root: Path,
    *,
    graph_root: Path | None = None,
) -> StreamingTachiomHnswPqMojoIndex:
    return StreamingTachiomHnswPqMojoIndex.from_components(
        base_index=load_streaming_tachiom_pq_index(index_root),
        graph=load_streaming_tachiom_hnsw_graph(
            index_root=index_root,
            graph_root=graph_root,
        ),
        address_backed=True,
    )


@dataclass(frozen=True, slots=True)
class StreamingTachiomHnswPqIndex:
    """Streaming TAC/PQ search using a persisted HNSW centroid graph."""

    base_index: StreamingTachiomPqIndex
    graph: StreamingTachiomHnswGraph

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.base_index.doc_ids

    @property
    def document_count(self) -> int:
        return self.base_index.document_count

    @property
    def total_vector_count(self) -> int:
        return self.base_index.total_vector_count

    @property
    def vector_dim(self) -> int:
        return self.base_index.vector_dim

    @property
    def centroid_count(self) -> int:
        return self.base_index.centroid_count

    @property
    def posting_count(self) -> int:
        return self.base_index.posting_count

    @property
    def index_bytes(self) -> int:
        return self.base_index.index_bytes + self.graph.graph_bytes

    @property
    def index_kind(self) -> str:
        return "streaming_token_aware_hnsw_centroid_postings_residual_pq_rerank"

    @property
    def rerank_kind(self) -> str:
        return self.base_index.rerank_kind

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
            rows.append(
                self.base_index._pq_rerank_positions(
                    query,
                    candidates,
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
        if self.base_index.candidate_k >= self.document_count:
            return tuple(range(self.document_count))
        selected_per_query_vector = min(
            self.base_index.centroids_per_query_vector,
            self.centroid_count,
        )
        doc_scores = np.zeros(self.document_count, dtype=VECTOR_DTYPE)
        touched = np.zeros(self.document_count, dtype=bool)
        for query_token_index in range(query.shape[0]):
            centroid_positions = self._hnsw_centroids_for_query_vector(
                query[query_token_index],
                k=selected_per_query_vector,
            )
            scores = np.matmul(
                self.base_index.centroids[list(centroid_positions)],
                query[query_token_index],
            )
            self.base_index._accumulate_selected_postings(
                centroid_positions=centroid_positions,
                selected_scores=scores,
                doc_scores=doc_scores,
                touched=touched,
            )
        candidate_count = min(self.base_index.candidate_k, self.document_count)
        if not np.any(touched):
            return tuple(range(candidate_count))
        ranked_scores = doc_scores.copy()
        ranked_scores[~touched] = np.float32(-3.4e38)
        return ranked_candidate_positions_from_scores(
            ranked_scores,
            candidate_k=candidate_count,
            final_k=final_k,
            candidate_pruning_alpha=self.base_index.candidate_pruning_alpha,
        )

    def _hnsw_centroids_for_query_vector(
        self,
        query_vector: np.ndarray,
        *,
        k: int,
    ) -> tuple[int, ...]:
        entry = self.graph.entry_point
        for layer_index in range(self.graph.level_count - 1, 0, -1):
            entry = self._greedy_layer_entry(query_vector, layer_index, entry)
        candidates = self._search_layer(
            query_vector,
            layer_index=0,
            entry_points=(entry,),
            ef=max(k, self.graph.ef_search),
        )
        return tuple(position for position, _score in candidates[:k])

    def _greedy_layer_entry(
        self,
        query_vector: np.ndarray,
        layer_index: int,
        entry: int,
    ) -> int:
        current = entry
        current_score = _dot(query_vector, self.base_index.centroids[current])
        improved = True
        while improved:
            improved = False
            neighbors = self.graph.neighbors(
                layer_index=layer_index,
                node_index=current,
            )
            if neighbors.size == 0:
                continue
            scores = np.matmul(self.base_index.centroids[neighbors], query_vector)
            best_local = int(np.argmax(scores))
            best_score = float(scores[best_local])
            if best_score > current_score:
                current = int(neighbors[best_local])
                current_score = best_score
                improved = True
        return current

    def _search_layer(
        self,
        query_vector: np.ndarray,
        *,
        layer_index: int,
        entry_points: Sequence[int],
        ef: int,
    ) -> list[tuple[int, float]]:
        visited = np.zeros(self.centroid_count, dtype=bool)
        initial_entries = tuple(int(entry) for entry in entry_points)
        visited[list(initial_entries)] = True
        initial_scores = np.matmul(
            self.base_index.centroids[list(initial_entries)],
            query_vector,
        )
        candidates = [
            (-float(score), int(entry))
            for entry, score in zip(initial_entries, initial_scores, strict=True)
        ]
        heapq.heapify(candidates)
        best = [
            (float(score), int(entry))
            for entry, score in zip(initial_entries, initial_scores, strict=True)
        ]
        heapq.heapify(best)
        while candidates:
            neg_score, node = heapq.heappop(candidates)
            score = -neg_score
            worst_best = best[0][0] if best else -np.inf
            if len(best) >= ef and score < worst_best:
                break
            neighbors = self.graph.neighbors(
                layer_index=layer_index,
                node_index=int(node),
            )
            if neighbors.size == 0:
                continue
            unseen = neighbors[~visited[neighbors]]
            if unseen.size == 0:
                continue
            visited[unseen] = True
            neighbor_scores = np.matmul(self.base_index.centroids[unseen], query_vector)
            for neighbor_int, neighbor_score_raw in zip(
                unseen,
                neighbor_scores,
                strict=True,
            ):
                neighbor_score = float(neighbor_score_raw)
                if len(best) < ef or neighbor_score > best[0][0]:
                    neighbor_position = int(neighbor_int)
                    heapq.heappush(candidates, (-neighbor_score, neighbor_position))
                    heapq.heappush(best, (neighbor_score, neighbor_position))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted(
            ((position, score) for score, position in best),
            key=lambda row: (-row[1], row[0]),
        )


def load_streaming_tachiom_hnsw_pq_index(
    index_root: Path,
    *,
    graph_root: Path | None = None,
) -> StreamingTachiomHnswPqIndex:
    base_index = load_streaming_tachiom_pq_index(index_root)
    graph = load_streaming_tachiom_hnsw_graph(
        index_root=index_root,
        graph_root=graph_root,
    )
    if graph.centroid_count != base_index.centroid_count:
        raise ValueError("HNSW graph centroid count does not match streaming index")
    return StreamingTachiomHnswPqIndex(base_index=base_index, graph=graph)


def _effective_codebook_size(
    path: Path,
    *,
    subspace_count: int,
    subspace_dim: int,
) -> int:
    itemsize = np.dtype(VECTOR_DTYPE).itemsize
    denominator = subspace_count * subspace_dim * itemsize
    size = path.stat().st_size
    if size % denominator != 0:
        raise ValueError("PQ codebook file size does not match config shape")
    return size // denominator


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))
