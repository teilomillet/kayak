"""Runs explicit stage-1 candidate generation for the Python SDK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidate_generator import (
    CandidateGenerator,
    DOCUMENT_PROXY_CANDIDATE_GENERATOR,
    EXACT_FULL_SCAN_CANDIDATE_GENERATOR,
)
from .dtypes import SCORE_DTYPE, VECTOR_DTYPE
from .late_scores import LateScores, SearchHit
from .layouts import NUMPY_REFERENCE_BACKEND
from .search_stage_profile import SearchStageProfile


@dataclass(frozen=True, slots=True)
class CandidateStageResult:
    """Scores, hits, and profile emitted by one candidate-generation stage."""

    generator: CandidateGenerator
    scores: LateScores
    hits: tuple[SearchHit, ...]
    profile: SearchStageProfile

    def __post_init__(self) -> None:
        if len(self.hits) != self.profile.output_hit_count:
            raise ValueError("candidate hits must match the stage profile")

    @property
    def candidate_doc_ids(self) -> tuple[str, ...]:
        return tuple(hit.doc_id for hit in self.hits)

    def select_index(self, index: "LateIndex") -> "LateIndex":
        if not self.hits:
            raise ValueError("candidate stage did not produce any documents")
        return index.select(self.candidate_doc_ids)


def _effective_vector_budget(requested_budget: int, available_count: int) -> int:
    if available_count <= 0:
        raise ValueError("stage inputs must contain at least one vector")
    if requested_budget == 0 or requested_budget > available_count:
        return available_count
    return requested_budget


def _document_proxy_scores(
    query: "LateQuery",
    index: "LateIndex",
    generator: CandidateGenerator,
) -> tuple[np.ndarray, SearchStageProfile]:
    query_matrix = query.as_vector_matrix()
    query_budget = _effective_vector_budget(
        generator.query_vector_budget, query.vector_count
    )
    query_proxy = np.mean(
        query_matrix[:query_budget], axis=0, dtype=VECTOR_DTYPE
    ).astype(VECTOR_DTYPE, copy=False)

    token_matrix = index.as_packed_token_matrix()
    proxy_vectors = np.empty((index.document_count, index.vector_dim), dtype=VECTOR_DTYPE)
    for document_index in range(index.document_count):
        start = int(index.doc_offsets[document_index])
        stop = int(index.doc_offsets[document_index + 1])
        document_budget = _effective_vector_budget(
            generator.document_vector_budget, stop - start
        )
        proxy_vectors[document_index] = np.mean(
            token_matrix[start : start + document_budget],
            axis=0,
            dtype=VECTOR_DTYPE,
        )

    values = np.matmul(proxy_vectors, query_proxy).astype(SCORE_DTYPE, copy=False)
    profile = SearchStageProfile(
        stage_name=DOCUMENT_PROXY_CANDIDATE_GENERATOR,
        input_hit_count=index.document_count,
        output_hit_count=0,
        query_vector_count=1,
        document_count=index.document_count,
        document_vector_count=index.document_count,
    )
    return values, profile


def generate_candidates(
    query: "LateQuery",
    index: "LateIndex",
    generator: CandidateGenerator,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> CandidateStageResult:
    if k < 0:
        raise ValueError("candidate top-k must be non-negative")
    if query.vector_dim != index.vector_dim:
        raise ValueError("query and index must share the same vector dimension")

    if generator.kind == EXACT_FULL_SCAN_CANDIDATE_GENERATOR:
        scores = index.maxsim(query, backend=backend)
        hits = scores.topk(k)
        profile = SearchStageProfile(
            stage_name=EXACT_FULL_SCAN_CANDIDATE_GENERATOR,
            input_hit_count=index.document_count,
            output_hit_count=len(hits),
            query_vector_count=query.vector_count,
            document_count=index.document_count,
            document_vector_count=index.total_vector_count,
        )
        return CandidateStageResult(generator, scores, hits, profile)

    if generator.kind == DOCUMENT_PROXY_CANDIDATE_GENERATOR:
        values, profile = _document_proxy_scores(query, index, generator)
        scores = LateScores.from_values(generator.kind, index.doc_ids, values)
        hits = scores.topk(k)
        return CandidateStageResult(
            generator,
            scores,
            hits,
            SearchStageProfile(
                stage_name=profile.stage_name,
                input_hit_count=profile.input_hit_count,
                output_hit_count=len(hits),
                query_vector_count=profile.query_vector_count,
                document_count=profile.document_count,
                document_vector_count=profile.document_vector_count,
            ),
        )

    raise ValueError(f"unsupported candidate generator: {generator.kind}")
