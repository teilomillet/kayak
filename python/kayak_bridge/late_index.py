"""Owns explicit packed and hybrid-flat index layouts for late interaction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .api_types import (
    DocIdsInput,
    DocOffsetsInput,
    DocTextsInput,
    TokenMatrixInput,
    TokenValuesInput,
)
from .array_conversions import (
    flatten_vector_matrix,
    reshape_flat_values,
    to_doc_ids,
    to_flat_vector_values,
    to_index_offsets,
    to_optional_doc_texts,
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
    """One searchable late-interaction index with an explicit storage layout."""

    layout: str
    doc_ids: tuple[str, ...]
    doc_offsets: np.ndarray
    vector_dim: int
    document_count: int
    total_vector_count: int
    doc_texts: tuple[str, ...] | None = None
    token_vectors: np.ndarray | None = None
    token_values: np.ndarray | None = None
    _doc_id_positions: dict[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_packed(
        cls,
        doc_ids: DocIdsInput,
        doc_offsets: DocOffsetsInput,
        token_vectors: TokenMatrixInput,
        *,
        doc_texts: DocTextsInput | None = None,
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
            doc_texts=to_optional_doc_texts(
                doc_texts,
                "packed index",
                expected_length=len(normalized_doc_ids),
            ),
            token_vectors=matrix,
        )

    @classmethod
    def from_hybrid_flat_dim128(
        cls,
        doc_ids: DocIdsInput,
        doc_offsets: DocOffsetsInput,
        token_values: TokenValuesInput,
        *,
        doc_texts: DocTextsInput | None = None,
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
            doc_texts=to_optional_doc_texts(
                doc_texts,
                "hybrid flat index",
                expected_length=len(normalized_doc_ids),
            ),
            token_values=values,
        )

    def __post_init__(self) -> None:
        if self.document_count <= 0:
            raise ValueError("index must contain at least one document")
        if self.document_count != len(self.doc_ids):
            raise ValueError("index document_count must match doc_ids")
        if self.doc_texts is not None and self.document_count != len(self.doc_texts):
            raise ValueError("index document_count must match document texts")
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

        doc_id_positions: dict[str, int] = {}
        for index, doc_id in enumerate(self.doc_ids):
            if doc_id in doc_id_positions:
                raise ValueError("index doc_ids must be unique")
            doc_id_positions[doc_id] = index
        object.__setattr__(self, "_doc_id_positions", doc_id_positions)

    @property
    def vector_counts(self) -> tuple[int, ...]:
        return tuple(
            int(self.doc_offsets[index + 1] - self.doc_offsets[index])
            for index in range(self.document_count)
        )

    def as_packed_token_matrix(self) -> np.ndarray:
        """Return this index as one packed 2D token-vector matrix."""
        if self.layout == INDEX_LAYOUT_PACKED:
            assert self.token_vectors is not None
            return self.token_vectors

        assert self.token_values is not None
        return reshape_flat_values(self.token_values, self.vector_dim)

    def as_flat_token_values(self) -> np.ndarray:
        """Return this index as one flat dim128 value buffer."""
        if self.layout == INDEX_LAYOUT_HYBRID_FLAT_DIM128:
            assert self.token_values is not None
            return self.token_values
        if self.vector_dim != FLAT_DIM128_VECTOR_DIM:
            raise ValueError("only dim128 packed indexes can convert to hybrid flat")

        assert self.token_vectors is not None
        return flatten_vector_matrix(self.token_vectors)

    def to_layout(self, layout: str) -> "LateIndex":
        """Convert this index into another supported public layout."""
        if layout == self.layout:
            return self
        if layout == INDEX_LAYOUT_PACKED:
            return LateIndex.from_packed(
                self.doc_ids,
                self.doc_offsets,
                self.as_packed_token_matrix(),
                doc_texts=self.doc_texts,
            )
        if layout == INDEX_LAYOUT_HYBRID_FLAT_DIM128:
            return LateIndex.from_hybrid_flat_dim128(
                self.doc_ids,
                self.doc_offsets,
                self.as_flat_token_values(),
                doc_texts=self.doc_texts,
            )
        raise ValueError(f"unsupported index layout: {layout}")

    def document_token_matrix(self, document_index: int) -> np.ndarray:
        """Return the token matrix for one document position in this index."""
        if document_index < 0 or document_index >= self.document_count:
            raise IndexError("document index out of range")

        start = int(self.doc_offsets[document_index])
        stop = int(self.doc_offsets[document_index + 1])
        return self.as_packed_token_matrix()[start:stop]

    def select(self, doc_ids: DocIdsInput) -> "LateIndex":
        """Return a smaller index containing only the requested document ids."""
        selected_doc_ids = to_doc_ids(doc_ids, "selected index doc_ids")
        selected_positions: list[int] = []
        selected_offsets = np.empty(len(selected_doc_ids) + 1, dtype=self.doc_offsets.dtype)
        selected_offsets[0] = 0
        selected_texts = [] if self.doc_texts is not None else None
        total_selected_vectors = 0
        for selected_index, doc_id in enumerate(selected_doc_ids, start=1):
            position = self._doc_id_positions.get(doc_id)
            if position is None:
                raise ValueError(f"document id not found in index: {doc_id}")

            selected_positions.append(position)
            start = int(self.doc_offsets[position])
            stop = int(self.doc_offsets[position + 1])
            total_selected_vectors += stop - start
            selected_offsets[selected_index] = total_selected_vectors
            if selected_texts is not None:
                assert self.doc_texts is not None
                selected_texts.append(self.doc_texts[position])

        packed_matrix = self.as_packed_token_matrix()
        selected_vectors = np.empty(
            (total_selected_vectors, self.vector_dim),
            dtype=packed_matrix.dtype,
        )
        running_offset = 0
        for position in selected_positions:
            start = int(self.doc_offsets[position])
            stop = int(self.doc_offsets[position + 1])
            vector_count = stop - start
            selected_vectors[running_offset : running_offset + vector_count] = (
                packed_matrix[start:stop]
            )
            running_offset += vector_count

        selected_offsets.setflags(write=False)
        selected_vectors.setflags(write=False)
        if self.layout == INDEX_LAYOUT_PACKED:
            return LateIndex(
                layout=INDEX_LAYOUT_PACKED,
                doc_ids=selected_doc_ids,
                doc_offsets=selected_offsets,
                vector_dim=self.vector_dim,
                document_count=len(selected_doc_ids),
                total_vector_count=total_selected_vectors,
                doc_texts=(
                    None if selected_texts is None else tuple(selected_texts)
                ),
                token_vectors=selected_vectors,
            )

        return LateIndex.from_packed(
            selected_doc_ids,
            selected_offsets,
            selected_vectors,
            doc_texts=selected_texts,
        ).to_layout(self.layout)

    def with_texts(self, doc_texts: DocTextsInput | None) -> "LateIndex":
        """Return the same index data with replaced optional document texts."""
        if self.layout == INDEX_LAYOUT_PACKED:
            return LateIndex.from_packed(
                self.doc_ids,
                self.doc_offsets,
                self.as_packed_token_matrix(),
                doc_texts=doc_texts,
            )
        return LateIndex.from_hybrid_flat_dim128(
            self.doc_ids,
            self.doc_offsets,
            self.as_flat_token_values(),
            doc_texts=doc_texts,
        )

    def maxsim(
        self, query: "LateQuery", *, backend: str = NUMPY_REFERENCE_BACKEND
    ) -> "LateScores":
        """Return exact scores for one query against this index."""
        from .backend_dispatch import maxsim_scores

        return maxsim_scores(query, self, backend=backend)

    def generate_candidates(
        self,
        query: "LateQuery",
        generator: "CandidateGenerator",
        *,
        k: int,
        backend: str = NUMPY_REFERENCE_BACKEND,
    ) -> "CandidateStageResult":
        """Run one explicit candidate generator against this index."""
        from .candidate_stage import generate_candidates

        return generate_candidates(query, self, generator, k=k, backend=backend)

    def search(
        self,
        query: "LateQuery",
        *,
        k: int,
        backend: str = NUMPY_REFERENCE_BACKEND,
        approximation: object | None = None,
    ) -> tuple["SearchHit", ...]:
        """Return top-k hits, optionally through an explicit approximation."""
        from .late_ops import search

        return search(
            query,
            self,
            k=k,
            backend=backend,
            approximation=approximation,
        )

    def search_with_plan(
        self,
        query: "LateQuery",
        *,
        plan: "SearchPlan",
        backend: str = NUMPY_REFERENCE_BACKEND,
    ) -> "SearchPlanResult":
        from .planned_search import search_with_plan

        return search_with_plan(query, self, plan, backend=backend)
