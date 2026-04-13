"""Internal implementation layer for the Python Kayak package.

This module exists so the monorepo can share implementation code between the
public ``kayak`` package and the internal Mojo-backed adapters.
It is not a stable public import surface; application code should import
from ``kayak`` instead.
"""

from .late_documents import LateDocuments
from .late_index import LateIndex
from .late_query_batch import LateQueryBatch
from .stage_artifact_materialization import StageArtifactMaterialization
from .late_ops import (
    BackendInfo,
    CandidateGenerator,
    CandidateStageResult,
    clause_text_stage2_operator,
    available_backends,
    backend_info,
    document_proxy_candidate_generator,
    document_proxy_search_plan,
    documents,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_candidate_generator,
    exact_full_scan_search_plan,
    exact_late_interaction_stage2_operator,
    flat_query_dim128,
    generate_candidates,
    hybrid_flat_dim128_index,
    maxsim,
    maxsim_batch,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    packed_index,
    query,
    query_batch,
    search,
    search_batch,
    SearchPlan,
    SearchPlanResult,
    SearchStageProfile,
    search_with_plan,
    Stage2Operator,
    noop_topk_stage2_operator,
)
from .late_query import LateQuery
from .late_scores import LateScores, SearchHit

__all__ = [
    "BackendInfo",
    "CandidateGenerator",
    "CandidateStageResult",
    "LateDocuments",
    "LateIndex",
    "LateQuery",
    "LateQueryBatch",
    "LateScores",
    "SearchHit",
    "MOJO_EXACT_CPU_BACKEND",
    "NUMPY_REFERENCE_BACKEND",
    "available_backends",
    "backend_info",
    "clause_text_stage2_operator",
    "document_proxy_candidate_generator",
    "document_proxy_search_plan",
    "documents",
    "exact_full_scan_clause_text_search_plan",
    "exact_full_scan_candidate_generator",
    "exact_full_scan_search_plan",
    "exact_late_interaction_stage2_operator",
    "flat_query_dim128",
    "generate_candidates",
    "hybrid_flat_dim128_index",
    "maxsim",
    "maxsim_batch",
    "noop_topk_stage2_operator",
    "packed_index",
    "query",
    "query_batch",
    "search",
    "search_batch",
    "SearchPlan",
    "SearchPlanResult",
    "SearchStageProfile",
    "StageArtifactMaterialization",
    "search_with_plan",
    "Stage2Operator",
]
