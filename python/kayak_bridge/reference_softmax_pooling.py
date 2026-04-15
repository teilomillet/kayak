"""Owns smooth late-interaction pooling reference scorers.

This module owns:
- exact NumPy reference scorers that use temperature-smoothed token pooling
- a drop-in pure late-interaction alternative to hard MaxSim

This module does not own:
- candidate generation
- query fusion
- backend dispatch
"""

from __future__ import annotations

import numpy as np

from .dtypes import MIN_SCORE, SCORE_DTYPE
from .late_scores import LateScores


def softmax_similarity_scores(
    query: "LateQuery",
    index: "LateIndex",
    *,
    temperature: float,
    backend: str | None = None,
) -> LateScores:
    if temperature <= 0.0:
        raise ValueError("softmax pooling requires a positive temperature")
    if query.vector_dim != index.vector_dim:
        raise ValueError("query and index must share the same vector dimension")

    query_vectors = query.as_vector_matrix()
    index_vectors = index.as_packed_token_matrix()
    scores = np.empty(index.document_count, dtype=SCORE_DTYPE)

    effective_backend = (
        f"softmax_similarity_tau{temperature:g}"
        if backend is None
        else backend
    )

    inverse_temperature = np.float32(1.0 / temperature)
    for document_index in range(index.document_count):
        start = int(index.doc_offsets[document_index])
        stop = int(index.doc_offsets[document_index + 1])

        if start == stop:
            scores[document_index] = np.float32(query.vector_count) * MIN_SCORE
            continue

        similarities = query_vectors @ index_vectors[start:stop].T
        scaled = similarities * inverse_temperature
        row_max = np.max(scaled, axis=1, keepdims=True)
        stabilized = scaled - row_max
        weights = np.exp(stabilized, dtype=np.float32)
        pooled_similarity = np.sum(
            weights * similarities,
            axis=1,
            dtype=SCORE_DTYPE,
        ) / np.sum(weights, axis=1, dtype=SCORE_DTYPE)
        scores[document_index] = np.sum(pooled_similarity, dtype=SCORE_DTYPE)

    return LateScores.from_values(effective_backend, index.doc_ids, scores)
