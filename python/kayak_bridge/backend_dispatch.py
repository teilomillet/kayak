"""Dispatches late-interaction scoring across explicit Python backends."""

from __future__ import annotations

import numpy as np

from .dtypes import SCORE_DTYPE
from .late_scores import LateScores
from .layouts import MOJO_EXACT_CPU_BACKEND, NUMPY_REFERENCE_BACKEND
from .backend_info import _unsupported_backend_error
from .mojo_exact_cpu import load_module as load_mojo_exact_cpu_module
from .mojo_payloads import index_payload, query_payload, MojoIndexPayload
from .prepared_index_cache import prepared_packed_index_object
from .reference_maxsim import maxsim_scores as numpy_maxsim_scores


def _mojo_scores_for_query_and_index(
    query: "LateQuery",
    index: "LateIndex",
    *,
    module: object | None = None,
    payload: MojoIndexPayload | None = None,
) -> np.ndarray:
    if module is None:
        module = load_mojo_exact_cpu_module()

    if index.layout == "packed":
        if query.layout != "nested":
            raise ValueError("packed indexes require a nested query layout")

        prepared_index = prepared_packed_index_object(index, module=module)
        return np.asarray(
            module.exact_scores_prepared_packed(
                query_payload(query),
                prepared_index,
            ),
            dtype=SCORE_DTYPE,
        )

    if payload is None:
        payload = index_payload(index)

    if index.layout != "hybrid_flat_dim128":
        raise ValueError(f"unsupported index layout: {index.layout}")

    assert payload.flat_token_values is not None
    if query.layout == "nested":
        return np.asarray(
            module.exact_scores_hybrid_flat_dim128(
                query_payload(query),
                payload.doc_ids,
                payload.doc_offsets,
                payload.packed_vectors,
                payload.flat_token_values,
            ),
            dtype=SCORE_DTYPE,
        )

    if query.layout == "flat_dim128":
        return np.asarray(
            module.exact_scores_hybrid_flat_dim128_with_flat_query(
                query_payload(query),
                payload.doc_ids,
                payload.doc_offsets,
                payload.packed_vectors,
                payload.flat_token_values,
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

    raise _unsupported_backend_error(backend)
