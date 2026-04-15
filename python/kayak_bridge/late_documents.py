"""Owns ragged late-interaction documents before they are packed into an index."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .api_types import DocIdsInput, DocumentMatricesInput, DocTextsInput
from .array_conversions import to_doc_ids, to_document_matrices, to_optional_doc_texts


@dataclass(frozen=True, slots=True)
class LateDocuments:
    """One ragged batch of document token matrices before index packing."""

    doc_ids: tuple[str, ...]
    token_matrices: tuple[np.ndarray, ...]
    vector_dim: int
    document_count: int
    total_vector_count: int
    texts: tuple[str, ...] | None = None

    @classmethod
    def from_inputs(
        cls,
        doc_ids: DocIdsInput,
        token_vectors: DocumentMatricesInput,
        *,
        texts: DocTextsInput | None = None,
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
            texts=to_optional_doc_texts(
                texts,
                "documents",
                expected_length=len(normalized_doc_ids),
            ),
        )

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("documents must contain at least one document")
        if self.document_count != len(self.doc_ids):
            raise ValueError("document_count must match doc_ids")
        if self.document_count != len(self.token_matrices):
            raise ValueError("document_count must match token matrices")
        if self.texts is not None and self.document_count != len(self.texts):
            raise ValueError("document_count must match document texts")
        if self.vector_dim <= 0:
            raise ValueError("documents vector_dim must be positive")
        if self.total_vector_count <= 0:
            raise ValueError("documents must contain at least one vector")

    @property
    def vector_counts(self) -> tuple[int, ...]:
        return tuple(int(matrix.shape[0]) for matrix in self.token_matrices)

    def pack(self) -> "LateIndex":
        """Pack ragged document matrices into one searchable packed index."""
        from .late_index import LateIndex

        doc_offsets = [0]
        running_offset = 0
        for matrix in self.token_matrices:
            running_offset += int(matrix.shape[0])
            doc_offsets.append(running_offset)

        token_vectors = np.concatenate(self.token_matrices, axis=0)
        return LateIndex.from_packed(
            self.doc_ids,
            doc_offsets,
            token_vectors,
            doc_texts=self.texts,
        )

    def to_layout(self, layout: str) -> "LateIndex":
        """Pack these documents and convert the result into the requested layout."""
        return self.pack().to_layout(layout)

    def with_texts(self, texts: DocTextsInput | None) -> "LateDocuments":
        """Return the same document vectors with replaced optional texts."""
        return LateDocuments.from_inputs(
            self.doc_ids,
            self.token_matrices,
            texts=texts,
        )
