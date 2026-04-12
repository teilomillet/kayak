"""Owns ragged late-interaction documents before they are packed into an index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_conversions import to_doc_ids, to_document_matrices


@dataclass(frozen=True, slots=True)
class LateDocuments:
    doc_ids: tuple[str, ...]
    token_matrices: tuple[np.ndarray, ...]
    vector_dim: int
    document_count: int
    total_vector_count: int

    @classmethod
    def from_inputs(
        cls, doc_ids: object, token_vectors: object
    ) -> "LateDocuments":
        normalized_doc_ids = to_doc_ids(doc_ids, "documents")
        matrices = to_document_matrices(token_vectors, "documents")
        if len(normalized_doc_ids) != len(matrices):
            raise ValueError("documents doc_ids and token vectors must align")

        vector_dim = int(matrices[0].shape[1])
        total_vector_count = 0
        for matrix in matrices:
            if matrix.shape[1] != vector_dim:
                raise ValueError(
                    "all documents must share the same vector dimension"
                )
            total_vector_count += int(matrix.shape[0])

        return cls(
            doc_ids=normalized_doc_ids,
            token_matrices=matrices,
            vector_dim=vector_dim,
            document_count=len(normalized_doc_ids),
            total_vector_count=total_vector_count,
        )

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("documents must contain at least one document")
        if self.document_count != len(self.doc_ids):
            raise ValueError("document_count must match doc_ids")
        if self.document_count != len(self.token_matrices):
            raise ValueError("document_count must match token matrices")
        if self.vector_dim <= 0:
            raise ValueError("documents vector_dim must be positive")
        if self.total_vector_count <= 0:
            raise ValueError("documents must contain at least one vector")

    @property
    def vector_counts(self) -> tuple[int, ...]:
        return tuple(int(matrix.shape[0]) for matrix in self.token_matrices)

    def pack(self) -> "LateIndex":
        from .late_index import LateIndex

        doc_offsets = [0]
        running_offset = 0
        for matrix in self.token_matrices:
            running_offset += int(matrix.shape[0])
            doc_offsets.append(running_offset)

        token_vectors = np.concatenate(self.token_matrices, axis=0)
        return LateIndex.from_packed(self.doc_ids, doc_offsets, token_vectors)

    def to_layout(self, layout: str) -> "LateIndex":
        return self.pack().to_layout(layout)
