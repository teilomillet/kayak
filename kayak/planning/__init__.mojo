from .candidate_budget import CandidateBudget
from .candidate_generator import (
    CandidateGenerator,
    CANDIDATE_GENERATOR_FAMILY_CENTROID,
    CANDIDATE_GENERATOR_FAMILY_EXACT,
    CANDIDATE_GENERATOR_FAMILY_GRAPH,
    CANDIDATE_GENERATOR_FAMILY_PROXY,
    centroid_heads_candidate_generator,
    centroid_postings_candidate_generator,
    centroid_postings_flat_candidate_generator,
    centroid_postings_head_auto_candidate_generator,
    centroid_postings_blockmax_candidate_generator,
    centroid_postings_head_candidate_generator,
    centroid_postings_imputed_candidate_generator,
    centroid_postings_imputed_flat_candidate_generator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
    gem_graph_candidate_generator,
)
from .candidate_set import CandidateSet
from .centroid_execution_contract import (
    CENTROID_EXECUTION_SCORE_VARIANT_BLOCKMAX,
    CENTROID_EXECUTION_SCORE_VARIANT_FLAT,
    CENTROID_EXECUTION_SCORE_VARIANT_HEAD,
    CENTROID_EXECUTION_SCORE_VARIANT_HEAD_AUTO,
    CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED,
    CENTROID_EXECUTION_SCORE_VARIANT_IMPUTED_FLAT,
    CENTROID_EXECUTION_SCORE_VARIANT_POSTINGS,
    CENTROID_EXECUTION_SHORTLIST_CANDIDATE_K,
    CENTROID_EXECUTION_SHORTLIST_FINAL_K,
    CENTROID_EXECUTION_SHORTLIST_NONE,
    CentroidExecutionContract,
    centroid_execution_contract,
)
from .graph_search_counters import (
    GraphSearchCounters,
    accumulate_graph_search_counters,
    graph_search_counters_have_activity,
)
from .stage1_capabilities import (
    Stage1Capabilities,
    STAGE1_ALIGNMENT_GRANULARITY_CENTROID,
    STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT,
    STAGE1_ALIGNMENT_GRANULARITY_DOCUMENT_TOKENS,
    STAGE1_ALIGNMENT_GRANULARITY_GRAPH_NODE,
    STAGE1_INTERACTION_SEMANTICS_APPROXIMATE_LATE_INTERACTION,
    STAGE1_INTERACTION_SEMANTICS_EXACT_LATE_INTERACTION,
    STAGE1_INTERACTION_SEMANTICS_NONE,
    STAGE1_SCORE_KIND_APPROXIMATE_INTERACTION_SCORE,
    STAGE1_SCORE_KIND_EXACT_SCORE,
    STAGE1_SCORE_KIND_PROXY_SCORE,
    stage1_capabilities_for_candidate_generator_kind,
    stage1_generator_supports_exact_doc_id_filter,
    stage1_generator_supports_filter_expression,
    stage1_generator_supports_match_all_filter,
    stage1_generator_supports_structured_filter,
    stage1_requires_search_artifact_family,
    stage1_required_search_artifact_families,
    stage1_single_required_search_artifact_family,
    stage1_supported_by_search_artifact_build_policy,
)
from .clause_text_stage import clause_text_rerank_candidates_for_plan
from .exact_late_interaction_clause_text_stage import (
    exact_late_interaction_clause_text_rerank_candidates_for_plan,
)
from .reference_scoring_semantics import (
    REFERENCE_SCORING_REQUIRED_ARTIFACT_LATE_INTERACTION,
    REFERENCE_SCORING_SCORE_KIND_EXACT,
    REFERENCE_SCORING_SEMANTICS_FAMILY_LATE_INTERACTION,
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .centroid_primitives import (
    ScoredCentroidSelection,
    accumulate_selected_centroid_scores,
)
from .collection_hit import CollectionHit, to_search_hit
from .execution import (
    candidate_generation_for_plan,
    candidate_recall_at_final_k,
    final_hits_for_plan,
    final_hits_to_search_hits,
    search_collection_for_plan,
)
from .execution_stage2 import stage2_result_for_plan
from .execution_stage3 import stage3_result_for_plan
from .explain import CollectionSearchExplain, explain_collection_search
from .faithfulness import (
    FaithfulnessAssessment,
    FaithfulnessPolicy,
    assess_faithfulness,
    best_effort_faithfulness_policy,
    exact_stage1_required_faithfulness_policy,
    oracle_full_recall_required_faithfulness_policy,
    stage1_generator_is_exact,
)
from .stage2_operator import (
    STAGE2_OPERATOR_FAMILY_IDENTITY,
    STAGE2_OPERATOR_FAMILY_HYBRID,
    STAGE2_OPERATOR_FAMILY_LATE_INTERACTION,
    STAGE2_OPERATOR_FAMILY_TEXT,
    STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
    STAGE2_REQUIRED_ARTIFACT_LATE_INTERACTION,
    Stage2Operator,
    clause_text_stage2_operator,
    exact_late_interaction_clause_text_stage2_operator,
    exact_late_interaction_stage2_operator,
    noop_topk_stage2_operator,
    stage2_operator_for_components,
)
from .stage2_reference_operator import (
    STAGE2_REFERENCE_OPERATOR_FAMILY_IDENTITY,
    STAGE2_REFERENCE_OPERATOR_FAMILY_LATE_INTERACTION,
    STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
    Stage2ReferenceOperator,
    exact_late_interaction_stage2_reference_operator,
    noop_topk_stage2_reference_operator,
)
from .stage3_verifier_operator import (
    STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
    STAGE3_VERIFIER_FAMILY_IDENTITY,
    STAGE3_VERIFIER_FAMILY_TEXT,
    Stage3VerifierOperator,
    clause_text_stage3_verifier_operator,
    none_stage3_verifier_operator,
)
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage2_result import Stage2Result
from .json import collection_search_explain_json
from .score_histogram import (
    ScoreHistogram,
    build_score_histogram,
    empty_score_histogram,
)
from .centroid_postings_blockmax_stage import (
    CentroidPostingBlockmaxProfile,
    CentroidPostingBlockmaxResult,
    centroid_posting_blockmax_scores_for_segment,
    centroid_posting_blockmax_scores_for_segment_profiled,
)
from .centroid_postings_imputed_stage import centroid_posting_imputed_scores_for_segment
from .centroid_postings_imputed_flat_stage import (
    centroid_posting_imputed_flat_scores_for_segment,
)
from .centroid_postings_stage import centroid_posting_scores_for_segment
from .centroid_postings_flat_stage import centroid_posting_flat_scores_for_segment
from .search_plan import (
    SearchPlan,
    SearchPlanCompatibilitySemantics,
    centroid_heads_search_plan,
    centroid_postings_flat_search_plan,
    centroid_postings_head_auto_search_plan,
    centroid_postings_blockmax_search_plan,
    centroid_postings_imputed_search_plan,
    centroid_postings_imputed_flat_search_plan,
    centroid_postings_head_search_plan,
    centroid_postings_search_plan,
    document_proxy_search_plan,
    exact_full_scan_clause_text_search_plan,
    exact_full_scan_search_plan,
    gem_graph_search_plan,
    same_search_plan_compatibility_semantics,
    search_plan_compatibility_semantics,
    search_plan_compatibility_semantics_for_components,
    search_plan_with_stage2_operator,
    search_plan_with_stage_components,
)
from .planning_goal import (
    SEARCH_PLANNING_GOAL_BALANCED,
    SEARCH_PLANNING_GOAL_EXACT_ONLY,
    SEARCH_PLANNING_GOAL_LATENCY_FIRST,
    SEARCH_PLANNING_GOAL_NATIVE_MULTI_VECTOR,
    require_search_planning_goal,
)
from .planner_registry import (
    SEARCH_PLANNER_STATUS_BENCHMARK_ONLY,
    SEARCH_PLANNER_STATUS_EXACT_FALLBACK,
    SEARCH_PLANNER_STATUS_EXPERIMENTAL,
    SEARCH_PLANNER_STATUS_PROMOTED,
    SEARCH_PLANNER_STATUS_REGRESSION_BASELINE,
    SearchPlannerRegistryEntry,
    default_candidate_generator_order_for_goal,
    planner_priority_for_goal,
    registered_search_planner_candidate_generator_kinds,
    registered_search_planner_entries,
    search_planner_registry_entry,
)
from .planner import (
    CandidateGeneratorOrderDecision,
    SearchPlanSelection,
    SearchPlanSelectionRequest,
    available_candidate_generator_kinds,
    candidate_generator_kind_is_available,
    search_plan_for_candidate_generator_kind,
    search_planning_goal_kinds,
    select_search_plan_for_availability,
)
from .planner_plan_factory import (
    planner_candidate_generator_for_kind,
    planner_default_search_plan_for_candidate_generator,
    planner_default_search_plan_for_kind,
)
from .stage_profile import SearchStageProfile
