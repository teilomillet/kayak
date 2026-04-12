"""Owns explicit packed and hybrid-flat index layouts for late interaction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_conversions import (
    flatten_vector_matrix,
    reshape_flat_values,
    to_doc_ids,
    to_flat_vector_values,
    to_index_offsets,
    to_vector_matrix,
)
from .dtypes import FLAT_DIM128_VECTOR_DIM
from .layouts import (
    INDEX_LAYOUT_HYBRID_FLAT_DIM128,
    INDEX_LAYOUT_PACKED,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
)


@dataclass(frozen=True, slots=True)
class LateIndex:
    layout: str
    doc_ids: tuple[str, ...]
    doc_offsets: np.ndarray
    vector_dim: int
    document_count: int
    total_vector_count: int
    token_vectors: np.ndarray | None = None
    token_values: np.ndarray | None = None

    @classmethod
    def from_packed(
        cls, doc_ids: object, doc_offsets: object, token_vectors: object
    ) -> "LateIndex":
        normalized_doc_ids = to_doc_ids(doc_ids, "packed index")
        offsets = to_index_offsets(
            doc_offsets,
            "packed index doc_offsets",
            expected_length=len(normalized_doc_ids) + 1,
        )
        matrix = to_vector_matrix(token_vectors, "packed index token_vectors")
        return cls(
            layout=INDEX_LAYOUT_PACKED,
            doc_ids=normalized_doc_ids,
            doc_offsets=offsets,
            vector_dim=int(matrix.shape[1]),
            document_count=len(normalized_doc_ids),
            total_vector_count=int(matrix.shape[0]),
            token_vectors=matrix,
        )

    @classmethod
    def from_hybrid_flat_dim128(
        cls, doc_ids: object, doc_offsets: object, token_values: object
    ) -> "LateIndex":
        normalized_doc_ids = to_doc_ids(doc_ids, "hybrid flat index")
        offsets = to_index_offsets(
            doc_offsets,
            "hybrid flat index doc_offsets",
            expected_length=len(normalized_doc_ids) + 1,
        )
        values = to_flat_vector_values(token_values, "hybrid flat token_values")
        if values.size % FLAT_DIM128_VECTOR_DIM != 0:
            raise ValueError(
                "hybrid flat token_values must be aligned to vector_dim"
            )

        return cls(
            layout=INDEX_LAYOUT_HYBRID_FLAT_DIM128,
            doc_ids=normalized_doc_ids,
            doc_offsets=offsets,
            vector_dim=FLAT_DIM128_VECTOR_DIM,
            document_count=len(normalized_doc_ids),
            total_vector_count=int(values.size // FLAT_DIM128_VECTOR_DIM),
            token_values=values,
        )

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("index must contain at least one document")
        if self.document_count != len(self.doc_ids):
            raise ValueError("index document_count must match doc_ids")
        if self.doc_offsets.shape != (self.document_count + 1,):
            raise ValueError("index doc_offsets length must equal doc_ids + 1")
        if int(self.doc_offsets[0]) != 0:
            raise ValueError("index doc_offsets must start at zero")
        if self.vector_dim <= 0:
            raise ValueError("index vector_dim must be positive")

        for index in range(1, len(self.doc_offsets)):
            if int(self.doc_offsets[index]) < int(self.doc_offsets[index - 1]):
                raise ValueError("index doc_offsets must be monotonic")

        if self.layout == INDEX_LAYOUT_PACKED:
            if self.token_vectors is None or self.token_values is not None:
                raise ValueError("packed index owns only token_vectors")
            if self.token_vectors.shape != (
                self.total_vector_count,
                self.vector_dim,
            ):
                raise ValueError("packed token_vectors do not match metadata")
        elif self.layout == INDEX_LAYOUT_HYBRID_FLAT_DIM128:
            if self.vector_dim != FLAT_DIM128_VECTOR_DIM:
                raise ValueError("hybrid flat index requires vector_dim=128")
            if self.token_values is None or self.token_vectors is not None:
                raise ValueError("hybrid flat index owns only token_values")
            if self.token_values.size != self.total_vector_count * self.vector_dim:
                raise ValueError("hybrid flat values do not match metadata")
        else:
            raise ValueError(f"unsupported index layout: {self.layout}")

        if int(self.doc_offsets[-1]) != self.total_vector_count:
            raise ValueError(
                "index doc_offsets must end at total vector count"
            )

    @property
    def vector_counts(self) -> tuple[int, ...]:
        return tuple(
            int(self.doc_offsets[index + 1] - self.doc_offsets[index])
            for index in range(self.document_count)
        )

    def as_packed_token_matrix(self) -> np.ndarray:
        if self.layout == INDEX_LAYOUT_PACKED:
            assert self.token_vectors is not None
            return self.token_vectors

        assert self.token_values is not None
        return reshape_flat_values(self.token_values, self.vector_dim)

    def as_flat_token_values(self) -> np.ndarray:
        if self.layout == INDEX_LAYOUT_HYBRID_FLAT_DIM128:
            assert self.token_values is not None
            return self.token_values
        if self.vector_dim != FLAT_DIM128_VECTOR_DIM:
            raise ValueError("only dim128 packed indexes can convert to hybrid flat")

        assert self.token_vectors is not None
        return flatten_vector_matrix(self.token_vectors)

    def to_layout(self, layout: str) -> "LateIndex":
        if layout == self.layout:
            return self
        if layout == INDEX_LAYOUT_PACKED:
            return LateIndex.from_packed(
                self.doc_ids, self.doc_offsets, self.as_packed_token_matrix()
            )
        if layout == INDEX_LAYOUT_HYBRID_FLAT_DIM128:
            return LateIndex.from_hybrid_flat_dim128(
                self.doc_ids, self.doc_offsets, self.as_flat_token_values()
            )
        raise ValueError(f"unsupported index layout: {layout}")

    def document_token_matrix(self, document_index: int) -> np.ndarray:
        if document_index < 0 or document_index >= self.document_count:
            raise IndexError("document index out of range")

        start = int(self.doc_offsets[document_index])
        stop = int(self.doc_offsets[document_index + 1])
        return self.as_packed_token_matrix()[start:stop]

    def select(self, doc_ids: object) -> "LateIndex":
        selected_doc_ids = to_doc_ids(doc_ids, "selected index doc_ids")
        positions = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}

        selected_offsets = [0]
        selected_matrices = []
        running_offset = 0
        for doc_id in selected_doc_ids:
            if doc_id not in positions:
                raise ValueError(f"document id not found in index: {doc_id}")

            matrix = self.document_token_matrix(positions[doc_id])
            selected_matrices.append(matrix)
            running_offset += int(matrix.shape[0])
            selected_offsets.append(running_offset)

        selected_vectors = np.concatenate(selected_matrices, axis=0)
        selected_index = LateIndex.from_packed(
            selected_doc_ids,
            selected_offsets,
            selected_vectors,
        )
        return selected_index.to_layout(self.layout)

    def maxsim(
        self, query: "LateQuery", *, backend: str = NUMPY_REFERENCE_BACKEND
    ) -> "LateScores":
        from .backend_dispatch import maxsim_scores

        return maxsim_scores(query, self, backend=backend)

    def search(
        self,
        query: "LateQuery",
        *,
        k: int,
        backend: str = NUMPY_REFERENCE_BACKEND,
    ) -> tuple["SearchHit", ...]:
        return self.maxsim(query, backend=backend).topk(k)
