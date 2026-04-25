"""Dispatches explicit batched late-interaction scoring without hiding query raggedness."""

from __future__ import annotations

import numpy as np

from .backend_info import _unsupported_backend_error
from .dtypes import SCORE_DTYPE
from .backend_dispatch import _mojo_scores_for_query_and_index, maxsim_scores
from .late_scores import LateScores
from .layouts import MOJO_EXACT_CPU_BACKEND, NUMPY_REFERENCE_BACKEND
from .mojo_exact_cpu import load_module as load_mojo_exact_cpu_module
from .mojo_payload_cache import cached_index_payload
from .mojo_payloads import query_payload
from .prepared_index_cache import prepared_packed_index_object


def maxsim_scores_batch(
    query_batch: "LateQueryBatch",
    index: "LateIndex",
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> tuple[LateScores, ...]:
    if query_batch.vector_dim != index.vector_dim:
        raise ValueError("query batch and index must share the same vector dimension")

    if backend == NUMPY_REFERENCE_BACKEND:
        return tuple(
            maxsim_scores(query, index, backend=backend)
            for query in query_batch.queries
        )

    if backend == MOJO_EXACT_CPU_BACKEND:
        module = load_mojo_exact_cpu_module()
        if index.layout == "packed" and all(
            query.layout == "nested" for query in query_batch.queries
        ):
            prepared_index = prepared_packed_index_object(index, module=module)
            scores_by_query = module.exact_scores_prepared_packed_batch(
                [query_payload(query) for query in query_batch.queries],
                prepared_index,
            )
            return tuple(
                LateScores.from_values(
                    backend,
                    index.doc_ids,
                    np.asarray(scores, dtype=SCORE_DTYPE),
                )
                for scores in scores_by_query
            )

        payload = cached_index_payload(index)
        return tuple(
            LateScores.from_values(
                backend,
                index.doc_ids,
                _mojo_scores_for_query_and_index(
                    query,
                    index,
                    module=module,
                    payload=payload,
                ),
            )
            for query in query_batch.queries
        )

    raise _unsupported_backend_error(backend)
