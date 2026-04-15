"""Common record helpers shared by external vector-database store adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kayak_bridge import LateIndex

from .materialize import packed_index_from_matrices


@dataclass(frozen=True, slots=True)
class ExternalStoredDocument:
    doc_id: str
    token_matrix: np.ndarray
    text: str | None
    metadata: dict[str, object] | None


def packed_index_from_records(
    records: tuple[ExternalStoredDocument, ...],
    *,
    include_text: bool,
    layout: str,
) -> LateIndex:
    index = packed_index_from_matrices(
        tuple(record.doc_id for record in records),
        tuple(record.token_matrix for record in records),
        doc_texts=(
            tuple(record.text or "" for record in records)
            if include_text and any(record.text is not None for record in records)
            else None
        ),
    )
    return index.to_layout(layout)
