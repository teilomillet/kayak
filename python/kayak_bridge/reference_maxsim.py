"""Provides the explicit NumPy reference backend for exact late-interaction scoring."""

from __future__ import annotations

import numpy as np

from .dtypes import MIN_SCORE, SCORE_DTYPE
from .late_scores import LateScores
from .layouts import NUMPY_REFERENCE_BACKEND


def maxsim_scores(
    query: "LateQuery",
    index: "LateIndex",
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> LateScores:
    if backend != NUMPY_REFERENCE_BACKEND:
        raise ValueError(f"unsupported backend: {backend}")
    if query.vector_dim != index.vector_dim:
        raise ValueError("query and index must share the same vector dimension")

    query_vectors = query.as_vector_matrix()
    index_vectors = index.as_packed_token_matrix()
    scores = np.empty(index.document_count, dtype=SCORE_DTYPE)

    for document_index in range(index.document_count):
        start = int(index.doc_offsets[document_index])
        stop = int(index.doc_offsets[document_index + 1])

        if start == stop:
            scores[document_index] = np.float32(query.vector_count) * MIN_SCORE
            continue

        similarities = query_vectors @ index_vectors[start:stop].T
        best_similarity = similarities.max(axis=1)
        scores[document_index] = np.sum(best_similarity, dtype=SCORE_DTYPE)

    return LateScores.from_values(backend, index.doc_ids, scores)
