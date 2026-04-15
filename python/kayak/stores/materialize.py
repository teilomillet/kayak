"""Builds trusted packed indexes for store-backed materialization paths."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from kayak_bridge.dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE
from kayak_bridge.late_index import LateIndex
from kayak_bridge.layouts import INDEX_LAYOUT_PACKED


def packed_index_from_matrices(
    doc_ids: Sequence[str],
    matrices: Sequence[np.ndarray],
    *,
    doc_texts: Sequence[str] | None = None,
) -> LateIndex:
    if not matrices:
        raise ValueError("store materialization must produce at least one document")

    vector_dim = int(matrices[0].shape[1])
    offsets = [0]
    running_offset = 0
    for matrix in matrices:
        if matrix.ndim != 2:
            raise ValueError("store document matrices must be 2D")
        if matrix.shape[1] != vector_dim:
            raise ValueError(
                "store materialization requires a shared vector dimension"
            )
        running_offset += int(matrix.shape[0])
        offsets.append(running_offset)

    token_vectors = np.concatenate(matrices, axis=0).astype(
        VECTOR_DTYPE,
        copy=False,
    )
    token_vectors.setflags(write=False)
    offset_array = np.asarray(offsets, dtype=INDEX_OFFSET_DTYPE)
    offset_array.setflags(write=False)
    return packed_index_from_parts(
        tuple(str(doc_id) for doc_id in doc_ids),
        offset_array,
        token_vectors,
        doc_texts=None if doc_texts is None else tuple(str(text) for text in doc_texts),
    )


def packed_index_from_parts(
    doc_ids: tuple[str, ...],
    doc_offsets: np.ndarray,
    token_vectors: np.ndarray,
    *,
    doc_texts: tuple[str, ...] | None = None,
) -> LateIndex:
    doc_offsets.setflags(write=False)
    token_vectors.setflags(write=False)
    return LateIndex(
        layout=INDEX_LAYOUT_PACKED,
        doc_ids=doc_ids,
        doc_offsets=doc_offsets,
        vector_dim=int(token_vectors.shape[1]),
        document_count=len(doc_ids),
        total_vector_count=int(token_vectors.shape[0]),
        doc_texts=doc_texts,
        token_vectors=token_vectors,
    )
