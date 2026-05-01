"""Deterministic per-token centroid construction for the TAC probe."""

from __future__ import annotations

import numpy as np

from .dtypes import TOKEN_ID_DTYPE, VECTOR_DTYPE
from .tachiom_types import TachiomTacConfig


def _build_centroids(
    *,
    token_values: np.ndarray,
    token_ids: np.ndarray,
    token_doc_positions: np.ndarray,
    allocation: dict[int, int],
    config: TachiomTacConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    centroids: list[np.ndarray] = []
    centroid_token_ids: list[int] = []
    centroid_doc_postings: list[np.ndarray] = []
    for token_id in sorted(allocation):
        group_positions = np.flatnonzero(token_ids == token_id)
        group_values = token_values[group_positions]
        group_centroids, assignments = _kmeans_deterministic(
            group_values,
            k=allocation[token_id],
            iterations=config.kmeans_iterations,
        )
        for local_centroid_index in range(group_centroids.shape[0]):
            local_positions = group_positions[assignments == local_centroid_index]
            posting = np.unique(token_doc_positions[local_positions]).astype(np.int64)
            centroids.append(group_centroids[local_centroid_index])
            centroid_token_ids.append(token_id)
            centroid_doc_postings.append(np.ascontiguousarray(posting, dtype=np.int64))

    if not centroids:
        raise ValueError("token-aware clustering produced no centroids")
    return (
        np.ascontiguousarray(np.stack(centroids), dtype=VECTOR_DTYPE),
        np.ascontiguousarray(centroid_token_ids, dtype=TOKEN_ID_DTYPE),
        tuple(centroid_doc_postings),
    )


def _kmeans_deterministic(
    values: np.ndarray,
    *,
    k: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= int(values.shape[0]):
        return values.copy(), np.arange(int(values.shape[0]), dtype=np.int64)

    centroids = _farthest_first_initialization(values, k)
    assignments = np.zeros(int(values.shape[0]), dtype=np.int64)
    for _ in range(iterations):
        assignments = _nearest_centroid_positions(values, centroids)
        next_centroids = centroids.copy()
        for centroid_index in range(k):
            member_values = values[assignments == centroid_index]
            if member_values.size > 0:
                next_centroids[centroid_index] = member_values.mean(axis=0)
        if np.allclose(next_centroids, centroids, rtol=1e-5, atol=1e-6):
            centroids = next_centroids
            break
        centroids = next_centroids
    assignments = _nearest_centroid_positions(values, centroids)
    return np.ascontiguousarray(centroids, dtype=VECTOR_DTYPE), assignments


def _farthest_first_initialization(values: np.ndarray, k: int) -> np.ndarray:
    center = values.mean(axis=0, keepdims=True)
    first = int(np.argmin(_squared_distances(values, center)[:, 0]))
    chosen = [first]
    min_distances = _squared_distances(values, values[[first]])[:, 0]
    for _ in range(1, k):
        next_index = int(np.argmax(min_distances))
        chosen.append(next_index)
        next_distances = _squared_distances(values, values[[next_index]])[:, 0]
        min_distances = np.minimum(min_distances, next_distances)
    return np.ascontiguousarray(values[chosen], dtype=VECTOR_DTYPE)


def _nearest_centroid_positions(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmin(_squared_distances(values, centroids), axis=1).astype(np.int64)


def _squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    value_norms = np.sum(values * values, axis=1, keepdims=True)
    centroid_norms = np.sum(centroids * centroids, axis=1, keepdims=True).T
    return (
        value_norms
        + centroid_norms
        - np.float32(2.0) * np.matmul(values, centroids.T)
    )
