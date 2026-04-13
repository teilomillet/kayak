"""Public Python SDK for Kayak late-interaction programming.

Import from ``kayak`` when writing application or research code in Python.
This package owns the stable late-interaction object model, exact and
stage-aware local operations, and explicit backend selection for the SDK
surface.

It is not the hosted engine surface for collections, snapshots, or service
operations. The sibling ``kayak_bridge`` package remains an internal
implementation layer for this monorepo and is not a supported public import
surface.
"""

from kayak_bridge import (
    BackendInfo,
    CandidateGenerator,
    CandidateStageResult,
    LateDocuments,
    LateIndex,
    LateQuery,
    LateQueryBatch,
    LateScores,
    MOJO_EXACT_CPU_BACKEND,
    NUMPY_REFERENCE_BACKEND,
    SearchHit,
    SearchPlan,
    SearchPlanResult,
    SearchStageProfile,
    available_backends,
    backend_info,
    clause_text_stage2_operator,
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
    packed_index,
    query,
    query_batch,
    search,
    search_batch,
    search_with_plan,
    Stage2Operator,
    noop_topk_stage2_operator,
)

PUBLIC_API = (
    "BackendInfo",
    "CandidateGenerator",
    "CandidateStageResult",
    "LateDocuments",
    "LateIndex",
    "LateQuery",
    "LateQueryBatch",
    "LateScores",
    "SearchHit",
    "SearchPlan",
    "SearchPlanResult",
    "SearchStageProfile",
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
    "search_with_plan",
    "Stage2Operator",
)

__all__ = [
    *PUBLIC_API,
]
