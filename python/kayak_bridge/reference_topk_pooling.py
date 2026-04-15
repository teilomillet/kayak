"""Owns alternative token-pooling late-interaction reference scorers.

This module owns:
- exact NumPy reference scorers that pool over the top-m token matches
- explicit fallback to MaxSim when `match_k == 1`

This module does not own:
- candidate generation
- query fusion
- backend dispatch
"""

from __future__ import annotations

import numpy as np

from .dtypes import MIN_SCORE, SCORE_DTYPE
from .late_scores import LateScores


def topk_mean_similarity_scores(
    query: "LateQuery",
    index: "LateIndex",
    *,
    match_k: int,
    backend: str | None = None,
) -> LateScores:
    if match_k <= 0:
        raise ValueError("top-k token pooling requires match_k to be positive")
    if query.vector_dim != index.vector_dim:
        raise ValueError("query and index must share the same vector dimension")

    query_vectors = query.as_vector_matrix()
    index_vectors = index.as_packed_token_matrix()
    scores = np.empty(index.document_count, dtype=SCORE_DTYPE)

    effective_backend = (
        f"topk_mean_similarity_k{match_k}" if backend is None else backend
    )

    for document_index in range(index.document_count):
        start = int(index.doc_offsets[document_index])
        stop = int(index.doc_offsets[document_index + 1])

        if start == stop:
            scores[document_index] = np.float32(query.vector_count) * MIN_SCORE
            continue

        similarities = query_vectors @ index_vectors[start:stop].T
        if match_k == 1:
            pooled_similarity = similarities.max(axis=1)
        else:
            effective_k = min(match_k, similarities.shape[1])
            topk_similarities = np.partition(
                similarities,
                kth=similarities.shape[1] - effective_k,
                axis=1,
            )[:, -effective_k:]
            pooled_similarity = np.mean(
                topk_similarities,
                axis=1,
                dtype=SCORE_DTYPE,
            )

        scores[document_index] = np.sum(pooled_similarity, dtype=SCORE_DTYPE)

    return LateScores.from_values(effective_backend, index.doc_ids, scores)
