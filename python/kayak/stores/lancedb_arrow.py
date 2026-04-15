"""Keeps LanceDB row materialization and Arrow conversion columnar."""

from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

import numpy as np

from kayak_bridge import LateDocuments, LateIndex
from kayak_bridge.dtypes import INDEX_OFFSET_DTYPE, VECTOR_DTYPE

from .metadata import matches_metadata_filter
from .materialize import packed_index_from_parts


DOC_ID_COLUMN = "doc_id"
TEXT_COLUMN = "text"
METADATA_JSON_COLUMN = "metadata_json"
TOKEN_MATRIX_COLUMN = "token_matrix"


def lancedb_schema(pa: Any, *, vector_dim: int) -> Any:
    """Build the stable LanceDB row schema for one vector dimension."""

    return pa.schema(
        [
            pa.field(DOC_ID_COLUMN, pa.string()),
            pa.field(TEXT_COLUMN, pa.string()),
            pa.field(METADATA_JSON_COLUMN, pa.string()),
            pa.field(
                TOKEN_MATRIX_COLUMN,
                pa.list_(pa.list_(pa.float32(), vector_dim)),
            ),
        ]
    )


def documents_to_arrow_table(
    pa: Any,
    documents: LateDocuments,
    *,
    metadata_rows: Sequence[dict[str, object]] | None,
) -> Any:
    """Convert public late documents into one Arrow table for LanceDB writes."""

    rows = []
    for index, doc_id in enumerate(documents.doc_ids):
        rows.append(
            {
                DOC_ID_COLUMN: str(doc_id),
                TEXT_COLUMN: None if documents.texts is None else documents.texts[index],
                METADATA_JSON_COLUMN: (
                    None
                    if metadata_rows is None
                    else json.dumps(metadata_rows[index], sort_keys=True)
                ),
                TOKEN_MATRIX_COLUMN: np.asarray(
                    documents.token_matrices[index],
                    dtype=VECTOR_DTYPE,
                ).tolist(),
            }
        )
    return pa.Table.from_pylist(
        rows,
        schema=lancedb_schema(pa, vector_dim=documents.vector_dim),
    )


def arrow_table_to_metadata_rows(arrow_table: Any) -> tuple[dict[str, object] | None, ...]:
    """Decode JSON metadata lazily enough for exact-match filtering."""

    payloads = arrow_table.column(METADATA_JSON_COLUMN).combine_chunks().to_pylist()
    return tuple(
        None if payload is None else dict(json.loads(str(payload)))
        for payload in payloads
    )


def filter_arrow_table_by_metadata(
    pa: Any,
    arrow_table: Any,
    *,
    where: dict[str, object] | None,
) -> Any:
    """Apply exact-match metadata filtering after one Arrow read."""

    if where is None:
        return arrow_table

    metadata_rows = arrow_table_to_metadata_rows(arrow_table)
    selected_positions = [
        position
        for position, metadata in enumerate(metadata_rows)
        if matches_metadata_filter(metadata, where)
    ]
    return take_arrow_table(pa, arrow_table, selected_positions)


def reorder_arrow_table_by_doc_ids(
    pa: Any,
    arrow_table: Any,
    *,
    doc_ids: Sequence[str],
) -> Any:
    """Restore explicit caller order after a storage-engine subset read."""

    doc_id_column = arrow_table.column(DOC_ID_COLUMN).combine_chunks().to_pylist()
    positions_by_doc_id = {
        str(doc_id): position for position, doc_id in enumerate(doc_id_column)
    }
    selected_positions = [
        positions_by_doc_id[doc_id] for doc_id in doc_ids if doc_id in positions_by_doc_id
    ]
    return take_arrow_table(pa, arrow_table, selected_positions)


def take_arrow_table(pa: Any, arrow_table: Any, positions: Sequence[int]) -> Any:
    """Take rows from one Arrow table while staying columnar."""

    return arrow_table.take(pa.array(list(positions), type=pa.int64()))


def vector_dim_from_schema(schema: Any) -> int:
    """Read the token vector dimension from the stable LanceDB schema."""

    token_matrix_type = schema.field(TOKEN_MATRIX_COLUMN).type
    inner_type = token_matrix_type.value_type
    if not hasattr(inner_type, "list_size"):
        raise TypeError("LanceDB token matrix column must use fixed-size vectors")
    return int(inner_type.list_size)


def arrow_table_to_packed_index(
    arrow_table: Any,
    *,
    include_text: bool,
) -> LateIndex:
    """Materialize one Arrow table into one packed Kayak index."""

    doc_ids = tuple(
        str(doc_id)
        for doc_id in arrow_table.column(DOC_ID_COLUMN).combine_chunks().to_pylist()
    )

    token_column = arrow_table.column(TOKEN_MATRIX_COLUMN).combine_chunks()
    offsets = np.asarray(
        token_column.offsets.to_numpy(zero_copy_only=False),
        dtype=INDEX_OFFSET_DTYPE,
    )
    offsets.setflags(write=False)

    fixed_size_vectors = token_column.values
    vector_dim = int(fixed_size_vectors.type.list_size)
    flat_values = fixed_size_vectors.values.to_numpy(zero_copy_only=False)
    token_vectors = np.asarray(flat_values, dtype=VECTOR_DTYPE).reshape(-1, vector_dim)
    token_vectors.setflags(write=False)

    doc_texts = None
    if include_text:
        text_values = arrow_table.column(TEXT_COLUMN).combine_chunks().to_pylist()
        if any(value is not None for value in text_values):
            doc_texts = tuple("" if value is None else str(value) for value in text_values)

    return packed_index_from_parts(
        doc_ids,
        offsets,
        token_vectors,
        doc_texts=doc_texts,
    )
