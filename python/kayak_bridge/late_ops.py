"""Exposes the Python-facing late-interaction constructors and local ops."""

from __future__ import annotations

from .backend_info import BackendInfo, available_backends, backend_info
from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
)
from .candidate_stage import CandidateStageResult
from .dtypes import FLAT_DIM128_VECTOR_DIM
from .late_query_batch import LateQueryBatch
from .late_documents import LateDocuments
from .late_index import LateIndex
from .late_query import LateQuery
from .late_scores import LateScores, SearchHit
from .batch_dispatch import maxsim_scores_batch
from .layouts import MOJO_EXACT_CPU_BACKEND, NUMPY_REFERENCE_BACKEND
from .backend_dispatch import maxsim_scores
from .planned_search import SearchPlanResult
from .search_plan import (
    SearchPlan,
    document_proxy_search_plan,
    exact_full_scan_search_plan,
)
from .search_stage_profile import SearchStageProfile


def query(token_vectors: object) -> LateQuery:
    return LateQuery.from_vectors(token_vectors)


def query_batch(token_vectors: object) -> LateQueryBatch:
    return LateQueryBatch.from_inputs(token_vectors)


def flat_query_dim128(token_values: object) -> LateQuery:
    return LateQuery.from_flat_values(
        token_values, vector_dim=FLAT_DIM128_VECTOR_DIM
    )


def documents(doc_ids: object, token_vectors: object) -> LateDocuments:
    return LateDocuments.from_inputs(doc_ids, token_vectors)


def packed_index(
    doc_ids: object, doc_offsets: object, token_vectors: object
) -> LateIndex:
    return LateIndex.from_packed(doc_ids, doc_offsets, token_vectors)


def hybrid_flat_dim128_index(
    doc_ids: object, doc_offsets: object, token_values: object
) -> LateIndex:
    return LateIndex.from_hybrid_flat_dim128(doc_ids, doc_offsets, token_values)


def maxsim(
    late_query: LateQuery,
    late_index: LateIndex,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> LateScores:
    return maxsim_scores(late_query, late_index, backend=backend)


def maxsim_batch(
    late_query_batch: LateQueryBatch,
    late_index: LateIndex,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> tuple[LateScores, ...]:
    return maxsim_scores_batch(late_query_batch, late_index, backend=backend)


def search(
    late_query: LateQuery,
    late_index: LateIndex,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> tuple[SearchHit, ...]:
    return maxsim(late_query, late_index, backend=backend).topk(k)


def search_batch(
    late_query_batch: LateQueryBatch,
    late_index: LateIndex,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> tuple[tuple[SearchHit, ...], ...]:
    return tuple(
        scores.topk(k)
        for scores in maxsim_batch(late_query_batch, late_index, backend=backend)
    )


def generate_candidates(
    late_query: LateQuery,
    late_index: LateIndex,
    generator: CandidateGenerator,
    *,
    k: int,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> CandidateStageResult:
    return late_index.generate_candidates(
        late_query,
        generator,
        k=k,
        backend=backend,
    )


def search_with_plan(
    late_query: LateQuery,
    late_index: LateIndex,
    plan: SearchPlan,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> SearchPlanResult:
    return late_index.search_with_plan(
        late_query,
        plan=plan,
        backend=backend,
    )
