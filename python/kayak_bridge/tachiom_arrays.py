"""Array validation and small ranking helpers for the TAC probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, TOKEN_ID_DTYPE, VECTOR_DTYPE

if TYPE_CHECKING:
    from .late_query import LateQuery
    from .late_query_batch import LateQueryBatch


def _top_positions(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be positive")
    count = min(k, int(scores.shape[0]))
    positions = np.arange(int(scores.shape[0]))
    if count == int(scores.shape[0]):
        order = np.lexsort((positions, -scores))
        return positions[order]

    selected = np.argpartition(-scores, count - 1)[:count]
    cutoff_score = np.min(scores[selected])
    tied_positions = positions[scores >= cutoff_score]
    if tied_positions.shape[0] < count:
        order = np.lexsort((positions, -scores))
        return positions[order[:count]]

    order = np.lexsort((tied_positions, -scores[tied_positions]))
    return tied_positions[order[:count]]


def _regular_doc_offsets(
    *,
    document_count: int,
    document_vector_count: int,
) -> np.ndarray:
    return np.arange(
        0,
        (document_count + 1) * document_vector_count,
        document_vector_count,
        dtype=INDEX_OFFSET_DTYPE,
    )


def _as_doc_offsets(
    doc_offsets: np.ndarray,
    *,
    document_count: int,
    total_vector_count: int,
) -> np.ndarray:
    if document_count <= 0:
        raise ValueError("doc_ids must contain at least one document")
    array = np.asarray(doc_offsets, dtype=INDEX_OFFSET_DTYPE)
    if array.shape != (document_count + 1,):
        raise ValueError("doc_offsets must align with doc_ids")
    if int(array[0]) != 0:
        raise ValueError("doc_offsets must start at zero")
    if int(array[-1]) != total_vector_count:
        raise ValueError("doc_offsets must end at total vector count")
    if np.any(array[1:] < array[:-1]):
        raise ValueError("doc_offsets must be monotonic")
    if np.any(array[1:] == array[:-1]):
        raise ValueError("Tachiom TAC requires at least one vector per document")
    return np.ascontiguousarray(array, dtype=INDEX_OFFSET_DTYPE)


def _doc_positions_from_offsets(doc_offsets: np.ndarray) -> np.ndarray:
    vector_counts = np.diff(doc_offsets)
    return np.repeat(
        np.arange(len(vector_counts), dtype=INDEX_OFFSET_DTYPE),
        vector_counts,
    )


def _regular_vector_count_or_none(doc_offsets: np.ndarray) -> int | None:
    vector_counts = np.diff(doc_offsets)
    if vector_counts.size == 0:
        return None
    first = int(vector_counts[0])
    if np.all(vector_counts == first):
        return first
    return None


def _as_document_tensor(documents: np.ndarray) -> np.ndarray:
    array = np.asarray(documents, dtype=VECTOR_DTYPE)
    if array.ndim != 3:
        raise ValueError("documents must have shape D x T x dim")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError("documents dimensions must be positive")
    return np.ascontiguousarray(array, dtype=VECTOR_DTYPE)


def _as_query_tensor(queries: np.ndarray, *, vector_dim: int) -> np.ndarray:
    array = np.asarray(queries, dtype=VECTOR_DTYPE)
    if array.ndim == 2:
        array = array.reshape(1, array.shape[0], array.shape[1])
    if array.ndim != 3:
        raise ValueError("queries must have shape Q x T x dim")
    if int(array.shape[2]) != vector_dim:
        raise ValueError("queries vector dimension must match the index")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("queries dimensions must be positive")
    return np.ascontiguousarray(array, dtype=VECTOR_DTYPE)


def _as_token_matrix(token_values: np.ndarray) -> np.ndarray:
    array = np.asarray(token_values, dtype=VECTOR_DTYPE)
    if array.ndim != 2:
        raise ValueError("token_values must have shape N x dim")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("token_values dimensions must be positive")
    return np.ascontiguousarray(array, dtype=VECTOR_DTYPE)


def _as_token_id_matrix(
    token_ids: np.ndarray,
    *,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    array = np.asarray(token_ids, dtype=TOKEN_ID_DTYPE)
    if array.shape != expected_shape:
        raise ValueError("token_ids must align with document token vectors")
    return np.ascontiguousarray(array, dtype=TOKEN_ID_DTYPE)


def _as_flat_token_ids(token_ids: np.ndarray, *, expected_length: int) -> np.ndarray:
    array = np.asarray(token_ids, dtype=TOKEN_ID_DTYPE).reshape(-1)
    if array.shape != (expected_length,):
        raise ValueError("token_ids must align with token_values")
    return np.ascontiguousarray(array, dtype=TOKEN_ID_DTYPE)


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _single_query_batch(query: "LateQuery") -> "LateQueryBatch":
    from .late_query_batch import LateQueryBatch

    return LateQueryBatch.from_queries([query])
