"""Dispatches late-interaction scoring across explicit Python backends."""

from __future__ import annotations

import numpy as np

from .dtypes import SCORE_DTYPE
from .late_scores import LateScores
from .layouts import MOJO_EXACT_CPU_BACKEND, NUMPY_REFERENCE_BACKEND
from .mojo_exact_cpu import load_module as load_mojo_exact_cpu_module
from .reference_maxsim import maxsim_scores as numpy_maxsim_scores


def _mojo_scores_for_query_and_index(
    query: "LateQuery",
    index: "LateIndex",
) -> np.ndarray:
    module = load_mojo_exact_cpu_module()

    doc_ids = list(index.doc_ids)
    doc_offsets = [int(offset) for offset in index.doc_offsets]
    packed_vectors = index.as_packed_token_matrix().tolist()

    if index.layout == "packed":
        if query.layout != "nested":
            raise ValueError("packed indexes require a nested query layout")

        return np.asarray(
            module.exact_scores_packed(
                query.as_vector_matrix().tolist(),
                doc_ids,
                doc_offsets,
                packed_vectors,
            ),
            dtype=SCORE_DTYPE,
        )

    if index.layout != "hybrid_flat_dim128":
        raise ValueError(f"unsupported index layout: {index.layout}")

    token_values = index.as_flat_token_values().tolist()
    if query.layout == "nested":
        return np.asarray(
            module.exact_scores_hybrid_flat_dim128(
                query.as_vector_matrix().tolist(),
                doc_ids,
                doc_offsets,
                packed_vectors,
                token_values,
            ),
            dtype=SCORE_DTYPE,
        )

    if query.layout == "flat_dim128":
        return np.asarray(
            module.exact_scores_hybrid_flat_dim128_with_flat_query(
                query.as_flat_values().tolist(),
                doc_ids,
                doc_offsets,
                packed_vectors,
                token_values,
            ),
            dtype=SCORE_DTYPE,
        )

    raise ValueError(f"unsupported query layout: {query.layout}")


def maxsim_scores(
    query: "LateQuery",
    index: "LateIndex",
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> LateScores:
    if backend == NUMPY_REFERENCE_BACKEND:
        return numpy_maxsim_scores(query, index, backend=backend)

    if backend == MOJO_EXACT_CPU_BACKEND:
        if query.vector_dim != index.vector_dim:
            raise ValueError("query and index must share the same vector dimension")

        values = _mojo_scores_for_query_and_index(query, index)
        return LateScores.from_values(backend, index.doc_ids, values)

    raise ValueError(f"unsupported backend: {backend}")
