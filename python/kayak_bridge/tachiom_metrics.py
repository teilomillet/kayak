"""Reference search and recall metrics for TAC probe validation."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .dtypes import VECTOR_DTYPE
from .tachiom_arrays import _as_document_tensor, _as_query_tensor, _top_positions


def exact_search_positions(
    *,
    documents: np.ndarray,
    queries: np.ndarray,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    """Small NumPy exact MaxSim reference used by the Tachiom probe tests."""

    document_tensor = _as_document_tensor(documents)
    query_tensor = _as_query_tensor(queries, vector_dim=int(document_tensor.shape[2]))
    rows: list[tuple[int, ...]] = []
    for query in query_tensor:
        scores = np.empty(int(document_tensor.shape[0]), dtype=VECTOR_DTYPE)
        for doc_index, document in enumerate(document_tensor):
            scores[doc_index] = np.max(np.matmul(query, document.T), axis=1).sum()
        rows.append(tuple(int(position) for position in _top_positions(scores, final_k)))
    return tuple(rows)


def mean_recall_at_k(
    *,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_positions_by_query: Sequence[Sequence[int]],
    k: int,
) -> float:
    if len(candidate_positions_by_query) != len(reference_positions_by_query):
        raise ValueError("candidate and reference query counts must match")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    for candidate_positions, reference_positions in zip(
        candidate_positions_by_query,
        reference_positions_by_query,
        strict=True,
    ):
        reference = set(reference_positions[:k])
        denominator = min(k, len(reference))
        if denominator == 0:
            recalls.append(0.0)
            continue
        candidate = set(candidate_positions[:k])
        recalls.append(len(candidate & reference) / float(denominator))
    return float(sum(recalls) / len(recalls))


def mean_candidate_set_recall_at_k(
    *,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_positions_by_query: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Return whether the candidate window contains the exact top-k documents."""

    if len(candidate_positions_by_query) != len(reference_positions_by_query):
        raise ValueError("candidate and reference query counts must match")
    if k <= 0:
        raise ValueError("k must be positive")
    recalls: list[float] = []
    for candidate_positions, reference_positions in zip(
        candidate_positions_by_query,
        reference_positions_by_query,
        strict=True,
    ):
        reference = set(reference_positions[:k])
        denominator = min(k, len(reference))
        if denominator == 0:
            recalls.append(0.0)
            continue
        candidate = set(candidate_positions)
        recalls.append(len(candidate & reference) / float(denominator))
    return float(sum(recalls) / len(recalls))
