"""HNSW-style centroid search layer for the TAC probe.

This module owns a deterministic Python reference for the paper's centroid
hierarchy gate. It does not own residual-PQ scoring or native Mojo kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import time
from typing import Sequence

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from .late_scores import SearchHit
from .tachiom_arrays import _as_query_tensor, _single_query_batch
from .tachiom_candidates import ranked_candidate_positions_from_scores
from .tachiom_index import TachiomTacIndex


@dataclass(frozen=True, slots=True)
class TachiomHnswConfig:
    """Explicit knobs for the centroid HNSW approximation."""

    max_neighbors: int = 16
    ef_construction: int = 64
    ef_search: int = 64
    level_probability: float = 0.0625
    seed: int = 7

    def validate(self) -> None:
        if self.max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if self.ef_construction < self.max_neighbors:
            raise ValueError("ef_construction must be >= max_neighbors")
        if self.ef_search <= 0:
            raise ValueError("ef_search must be positive")
        if not (0.0 < self.level_probability < 1.0):
            raise ValueError("level_probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class TachiomCentroidHnswGraph:
    """Layered navigable graph over TAC centroids."""

    centroids: np.ndarray
    levels: np.ndarray
    layers: tuple[tuple[tuple[int, ...], ...], ...]
    entry_point: int
    config: TachiomHnswConfig
    build_seconds: float

    @classmethod
    def build(
        cls,
        centroids: np.ndarray,
        *,
        config: TachiomHnswConfig,
    ) -> "TachiomCentroidHnswGraph":
        config.validate()
        values = np.ascontiguousarray(centroids, dtype=VECTOR_DTYPE)
        if values.ndim != 2 or values.shape[0] <= 0:
            raise ValueError("centroids must have shape C x dim")
        started_at = time.perf_counter()
        levels = _sample_levels(
            count=int(values.shape[0]),
            probability=config.level_probability,
            seed=config.seed,
        )
        mutable_layers = _build_hnsw_layers(values, levels, config=config)
        layers = tuple(tuple(tuple(row) for row in layer) for layer in mutable_layers)
        entry_point = int(np.argmax(levels))
        return cls(
            centroids=values,
            levels=np.ascontiguousarray(levels, dtype=INDEX_OFFSET_DTYPE),
            layers=layers,
            entry_point=entry_point,
            config=config,
            build_seconds=time.perf_counter() - started_at,
        )

    @property
    def graph_bytes(self) -> int:
        edge_count = sum(len(row) for layer in self.layers for row in layer)
        return int(self.levels.nbytes + edge_count * np.dtype(INDEX_OFFSET_DTYPE).itemsize)

    def search(self, query_vector: np.ndarray, *, k: int) -> tuple[int, ...]:
        if k <= 0:
            raise ValueError("k must be positive")
        vector = np.asarray(query_vector, dtype=VECTOR_DTYPE)
        entry = self.entry_point
        for layer_index in range(len(self.layers) - 1, 0, -1):
            entry = _greedy_layer_entry(
                vector,
                self.centroids,
                self.layers[layer_index],
                entry,
            )
        candidates = _search_layer(
            vector,
            self.centroids,
            self.layers[0],
            entry_points=(entry,),
            ef=max(k, self.config.ef_search),
        )
        return tuple(position for position, _ in candidates[:k])


@dataclass(frozen=True, slots=True)
class TachiomTacHnswIndex:
    """TAC candidate generation using HNSW centroid selection plus exact rerank."""

    tac_index: TachiomTacIndex
    graph: TachiomCentroidHnswGraph

    @classmethod
    def from_tac_index(
        cls,
        index: TachiomTacIndex,
        *,
        config: TachiomHnswConfig,
    ) -> "TachiomTacHnswIndex":
        graph = TachiomCentroidHnswGraph.build(index.centroids, config=config)
        return cls(tac_index=index, graph=graph)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.tac_index.doc_ids

    @property
    def document_count(self) -> int:
        return self.tac_index.document_count

    @property
    def vector_dim(self) -> int:
        return self.tac_index.vector_dim

    @property
    def centroid_count(self) -> int:
        return self.tac_index.centroid_count

    @property
    def posting_count(self) -> int:
        return self.tac_index.posting_count

    @property
    def index_bytes(self) -> int:
        return self.tac_index.index_bytes + self.graph.graph_bytes

    @property
    def build_seconds(self) -> float:
        return self.tac_index.build_seconds + self.graph.build_seconds

    @property
    def index_kind(self) -> str:
        return "token_aware_hnsw_centroid_postings_exact_rerank"

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
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        rows: list[tuple[int, ...]] = []
        for query in query_tensor:
            candidates = self._candidate_positions_for_query(query, final_k=final_k)
            rows.append(
                self.tac_index._exact_rerank_positions(
                    query,
                    candidates,
                    final_k=final_k,
                )
            )
        return tuple(rows)

    def search_query_batch(
        self,
        query_batch: object,
        *,
        final_k: int,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        rows: list[tuple[SearchHit, ...]] = []
        for query in query_batch.queries:
            rows.append(self.search_batch_hits(query.as_vector_matrix(), final_k=final_k)[0])
        return tuple(rows)

    def search_query(self, query: object, *, final_k: int) -> tuple[SearchHit, ...]:
        return self.search_query_batch(_single_query_batch(query), final_k=final_k)[0]

    def search_batch_hits(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[SearchHit, ...], ...]:
        query_tensor = _as_query_tensor(queries, vector_dim=self.vector_dim)
        rows: list[tuple[SearchHit, ...]] = []
        for query in query_tensor:
            candidates = self._candidate_positions_for_query(query, final_k=final_k)
            scored = self.tac_index._exact_rerank_position_scores(
                query,
                candidates,
                final_k=final_k,
            )
            rows.append(
                tuple(
                    SearchHit(doc_id=self.tac_index.doc_ids[position], score=score)
                    for position, score in scored
                )
            )
        return tuple(rows)

    def _candidate_positions_for_query(
        self,
        query: np.ndarray,
        *,
        final_k: int | None,
    ) -> tuple[int, ...]:
        if self.tac_index.config.candidate_k >= self.document_count:
            return tuple(range(self.document_count))
        selected_per_query_vector = min(
            self.tac_index.config.centroids_per_query_vector,
            self.centroid_count,
        )
        doc_scores = np.zeros(self.document_count, dtype=VECTOR_DTYPE)
        touched = np.zeros(self.document_count, dtype=bool)
        for query_token_index in range(query.shape[0]):
            centroid_positions = self.graph.search(
                query[query_token_index],
                k=selected_per_query_vector,
            )
            best_for_query_token = np.full(
                self.document_count,
                -np.inf,
                dtype=VECTOR_DTYPE,
            )
            scores = np.matmul(
                self.tac_index.centroids[list(centroid_positions)],
                query[query_token_index],
            )
            for local_index, centroid_position in enumerate(centroid_positions):
                posting = self.tac_index.centroid_doc_postings[int(centroid_position)]
                if posting.size == 0:
                    continue
                score = VECTOR_DTYPE(scores[local_index])
                best_for_query_token[posting] = np.maximum(
                    best_for_query_token[posting],
                    score,
                )
                touched[posting] = True
            matched = np.isfinite(best_for_query_token)
            doc_scores[matched] += best_for_query_token[matched]
        candidate_count = min(self.tac_index.config.candidate_k, self.document_count)
        if not np.any(touched):
            return tuple(range(candidate_count))
        ranked_scores = doc_scores.copy()
        ranked_scores[~touched] = np.float32(-3.4e38)
        return ranked_candidate_positions_from_scores(
            ranked_scores,
            candidate_k=candidate_count,
            final_k=final_k,
            candidate_pruning_alpha=self.tac_index.config.candidate_pruning_alpha,
        )


def _sample_levels(*, count: int, probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    levels = np.zeros(count, dtype=np.int64)
    for index in range(count):
        level = 0
        while rng.random() < probability:
            level += 1
        levels[index] = level
    return levels


def _build_hnsw_layers(
    centroids: np.ndarray,
    levels: np.ndarray,
    *,
    config: TachiomHnswConfig,
) -> list[list[list[int]]]:
    max_level = int(levels.max(initial=0))
    layers = [[[] for _ in range(int(centroids.shape[0]))] for _ in range(max_level + 1)]
    entry_point = 0
    for node in range(1, int(centroids.shape[0])):
        entry = entry_point
        for layer_index in range(int(levels[entry_point]), int(levels[node]), -1):
            entry = _greedy_layer_entry(
                centroids[node],
                centroids,
                layers[layer_index],
                entry,
            )
        for layer_index in range(min(int(levels[node]), len(layers) - 1), -1, -1):
            neighbors = _search_layer(
                centroids[node],
                centroids,
                layers[layer_index],
                entry_points=(entry,),
                ef=config.ef_construction,
            )
            selected = [position for position, _ in neighbors[: config.max_neighbors]]
            for neighbor in selected:
                _connect_bidirectional(
                    layers[layer_index],
                    centroids,
                    left=node,
                    right=neighbor,
                    max_neighbors=config.max_neighbors,
                )
            if selected:
                entry = selected[0]
        if levels[node] > levels[entry_point]:
            entry_point = node
    return layers


def _greedy_layer_entry(
    query: np.ndarray,
    centroids: np.ndarray,
    layer: Sequence[Sequence[int]],
    entry: int,
) -> int:
    current = entry
    current_score = _dot(query, centroids[current])
    improved = True
    while improved:
        improved = False
        if not layer[current]:
            continue
        neighbors = np.asarray(layer[current], dtype=np.int64)
        scores = np.matmul(centroids[neighbors], query)
        best_local = int(np.argmax(scores))
        best_score = float(scores[best_local])
        if best_score > current_score:
            current = int(neighbors[best_local])
            current_score = best_score
            improved = True
    return current


def _search_layer(
    query: np.ndarray,
    centroids: np.ndarray,
    layer: Sequence[Sequence[int]],
    *,
    entry_points: Sequence[int],
    ef: int,
) -> list[tuple[int, float]]:
    visited = set(int(entry) for entry in entry_points)
    initial_entries = tuple(int(entry) for entry in entry_points)
    initial_scores = np.matmul(centroids[list(initial_entries)], query)
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
        unseen = [
            int(neighbor)
            for neighbor in layer[node]
            if int(neighbor) not in visited
        ]
        if not unseen:
            continue
        visited.update(unseen)
        unseen_array = np.asarray(unseen, dtype=np.int64)
        neighbor_scores = np.matmul(centroids[unseen_array], query)
        for neighbor, neighbor_score_raw in zip(unseen, neighbor_scores, strict=True):
            neighbor_score = float(neighbor_score_raw)
            if len(best) < ef or neighbor_score > best[0][0]:
                heapq.heappush(candidates, (-neighbor_score, neighbor))
                heapq.heappush(best, (neighbor_score, neighbor))
                if len(best) > ef:
                    heapq.heappop(best)
    return sorted(
        ((position, score) for score, position in best),
        key=lambda row: (-row[1], row[0]),
    )


def _connect_bidirectional(
    layer: list[list[int]],
    centroids: np.ndarray,
    *,
    left: int,
    right: int,
    max_neighbors: int,
) -> None:
    if right not in layer[left]:
        layer[left].append(right)
    if left not in layer[right]:
        layer[right].append(left)
    layer[left] = _prune_neighbors(centroids, left, layer[left], max_neighbors)
    layer[right] = _prune_neighbors(centroids, right, layer[right], max_neighbors)


def _prune_neighbors(
    centroids: np.ndarray,
    node: int,
    neighbors: Sequence[int],
    max_neighbors: int,
) -> list[int]:
    unique = np.asarray(sorted(set(int(neighbor) for neighbor in neighbors)), dtype=np.int64)
    if unique.size == 0:
        return []
    scores = np.matmul(centroids[unique], centroids[node])
    order = np.lexsort((unique, -scores))
    scored = [(float(scores[index]), int(unique[index])) for index in order]
    selected: list[int] = []
    deferred: list[int] = []
    for score, neighbor in scored:
        diverse = True
        for chosen in selected:
            if _dot(centroids[neighbor], centroids[chosen]) > score:
                diverse = False
                break
        if diverse:
            selected.append(neighbor)
            if len(selected) == max_neighbors:
                return selected
        else:
            deferred.append(neighbor)
    for neighbor in deferred:
        if neighbor not in selected:
            selected.append(neighbor)
            if len(selected) == max_neighbors:
                break
    return selected


def _dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))
