"""Normalizes NumPy/PyTorch inputs into explicit readonly late-interaction arrays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the managed env.
    torch = None


def _is_torch_tensor(value: object) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def readonly_array(value: Any, *, dtype: np.dtype | type) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _numpy_from_input(value: Any) -> np.ndarray:
    if _is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_doc_ids(doc_ids: Sequence[object], owner: str) -> tuple[str, ...]:
    normalized = tuple(str(doc_id) for doc_id in doc_ids)
    if not normalized:
        raise ValueError(f"{owner} must contain at least one document id")
    return normalized


def to_optional_text(value: object | None, owner: str) -> str | None:
    if value is None:
        return None

    normalized = str(value)
    if normalized == "":
        raise ValueError(f"{owner} text must be non-empty when provided")
    return normalized


def to_optional_doc_texts(
    doc_texts: Sequence[object] | None,
    owner: str,
    *,
    expected_length: int,
) -> tuple[str, ...] | None:
    if doc_texts is None:
        return None
    if isinstance(doc_texts, (str, bytes)):
        raise ValueError(f"{owner} texts must be a sequence, not one string")

    normalized = tuple(str(text) for text in doc_texts)
    if len(normalized) != expected_length:
        raise ValueError(f"{owner} texts must align with document ids")
    if any(text == "" for text in normalized):
        raise ValueError(f"{owner} texts must be non-empty when provided")
    return normalized


def to_vector_matrix(value: Any, owner: str) -> np.ndarray:
    array = _numpy_from_input(value)
    if array.ndim != 2:
        raise ValueError(f"{owner} must be a 2D array")
    if array.shape[0] <= 0:
        raise ValueError(f"{owner} must contain at least one vector")
    if array.shape[1] <= 0:
        raise ValueError(f"{owner} vectors must not be empty")
    return readonly_array(array, dtype=VECTOR_DTYPE)


def to_flat_vector_values(value: Any, owner: str) -> np.ndarray:
    array = _numpy_from_input(value)
    if array.ndim != 1:
        raise ValueError(f"{owner} must be a flat 1D array")
    if array.shape[0] <= 0:
        raise ValueError(f"{owner} must contain at least one scalar value")
    return readonly_array(array, dtype=VECTOR_DTYPE)


def to_document_matrices(value: Any, owner: str) -> tuple[np.ndarray, ...]:
    if _is_torch_tensor(value):
        tensor = value.detach().cpu().numpy()
        if tensor.ndim != 3:
            raise ValueError(f"{owner} torch input must be a 3D tensor")
        return tuple(
            to_vector_matrix(tensor[index], f"{owner}[{index}]")
            for index in range(tensor.shape[0])
        )

    if isinstance(value, np.ndarray):
        if value.ndim != 3:
            raise ValueError(f"{owner} ndarray input must be a 3D array")
        return tuple(
            to_vector_matrix(value[index], f"{owner}[{index}]")
            for index in range(value.shape[0])
        )

    if not isinstance(value, Sequence):
        raise ValueError(f"{owner} must be a sequence of 2D arrays")

    matrices = tuple(
        to_vector_matrix(document_vectors, f"{owner}[{index}]")
        for index, document_vectors in enumerate(value)
    )
    if not matrices:
        raise ValueError(f"{owner} must contain at least one document")
    return matrices


def to_query_matrices(value: Any, owner: str) -> tuple[np.ndarray, ...]:
    if _is_torch_tensor(value):
        tensor = value.detach().cpu().numpy()
        if tensor.ndim != 3:
            raise ValueError(f"{owner} torch input must be a 3D tensor")
        return tuple(
            to_vector_matrix(tensor[index], f"{owner}[{index}]")
            for index in range(tensor.shape[0])
        )

    if isinstance(value, np.ndarray):
        if value.ndim != 3:
            raise ValueError(f"{owner} ndarray input must be a 3D array")
        return tuple(
            to_vector_matrix(value[index], f"{owner}[{index}]")
            for index in range(value.shape[0])
        )

    if not isinstance(value, Sequence):
        raise ValueError(f"{owner} must be a sequence of 2D arrays")

    matrices = tuple(
        to_vector_matrix(query_vectors, f"{owner}[{index}]")
        for index, query_vectors in enumerate(value)
    )
    if not matrices:
        raise ValueError(f"{owner} must contain at least one query")
    return matrices


def to_index_offsets(
    value: Any, owner: str, *, expected_length: int
) -> np.ndarray:
    array = np.asarray(value, dtype=INDEX_OFFSET_DTYPE)
    if array.ndim != 1:
        raise ValueError(f"{owner} must be a 1D array")
    if array.shape[0] != expected_length:
        raise ValueError(
            f"{owner} must contain exactly {expected_length} entries"
        )
    return readonly_array(array, dtype=INDEX_OFFSET_DTYPE)


def flatten_vector_matrix(matrix: np.ndarray) -> np.ndarray:
    return readonly_array(np.reshape(matrix, (-1,)), dtype=VECTOR_DTYPE)


def reshape_flat_values(token_values: np.ndarray, vector_dim: int) -> np.ndarray:
    matrix = np.reshape(token_values, (-1, vector_dim))
    matrix.setflags(write=False)
    return matrix
