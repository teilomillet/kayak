"""Owns Arrow-to-Kayak index conversion for LanceDB-backed benchmarks.

This module exists to keep storage-controlled comparisons honest. Rebuilding a
Kayak index from LanceDB rows via ``to_pylist()`` creates large amounts of
Python object churn that is unrelated to retrieval itself. The benchmark path
only needs document ids, document offsets, and contiguous vector values, so it
should stay columnar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import kayak

from .dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE


def table_to_packed_kayak_index(table: Any) -> kayak.LateIndex:
    """Materialize one LanceDB table into one Kayak packed index.

    Reason:
    - LanceDB already exposes the vectors as Arrow list buffers
    - Kayak packed indexes already consume contiguous ids, offsets, and values
    - staying columnar removes avoidable Python row and float allocation
    """

    arrow_table = table.to_arrow()
    doc_ids = tuple(
        str(doc_id)
        for doc_id in arrow_table.column("doc_id").combine_chunks().to_pylist()
    )

    vector_column = arrow_table.column("vector").combine_chunks()
    if not hasattr(vector_column, "offsets"):
        raise TypeError("expected LanceDB vector column to expose list offsets")

    offsets = np.asarray(
        vector_column.offsets.to_numpy(zero_copy_only=False),
        dtype=INDEX_OFFSET_DTYPE,
    )
    offsets.setflags(write=False)

    fixed_size_vectors = vector_column.values
    vector_dim = int(fixed_size_vectors.type.list_size)
    flat_values = fixed_size_vectors.values.to_numpy(zero_copy_only=False)
    token_vectors = np.asarray(flat_values, dtype=VECTOR_DTYPE).reshape(
        -1, vector_dim
    )
    token_vectors.setflags(write=False)

    return kayak.packed_index(doc_ids, offsets, token_vectors)
