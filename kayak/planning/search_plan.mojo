# Explicit multi-stage search plan contract.

from std.collections import List

from .candidate_budget import CandidateBudget
from .candidate_generator import (
    CandidateGenerator,
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
from .faithfulness import (
    FaithfulnessPolicy,
    exact_stage1_required_faithfulness_policy,
)
from .reference_scoring_semantics import (
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .stage2_operator import (
    Stage2Operator,
    clause_text_stage2_operator,
    reference_scoring_semantics_for_stage2_operator_kind,
    stage2_operator_for_components,
    stage2_reference_operator_for_stage2_operator_kind,
    stage3_verifier_for_stage2_operator_kind,
    exact_late_interaction_stage2_operator,
    noop_topk_stage2_operator,
)
from .stage2_reference_operator import Stage2ReferenceOperator
from .stage3_verifier_operator import (
    Stage3VerifierOperator,
    none_stage3_verifier_operator,
)


struct SearchPlanCompatibilitySemantics(Copyable):
    var stage2_kind: String
    var stage2_family: String
    var stage2_requires_query_text: Bool
    var stage2_required_artifact_families: List[String]
    var exact_stage_kind: String
    var reranker_kind: String

    def __init__(
        out self,
        read reference_scoring_semantics: ReferenceScoringSemantics,
        read stage2_operator: Stage2Operator,
        read stage3_verifier: Stage3VerifierOperator,
    ):
        self.stage2_kind = stage2_operator.kind.copy()
        self.stage2_family = stage2_operator.family.copy()
        self.stage2_requires_query_text = stage2_operator.requires_query_text
        self.stage2_required_artifact_families = (
            stage2_operator.required_artifact_families.copy()
        )
        self.exact_stage_kind = reference_scoring_semantics.kind.copy()
        self.reranker_kind = stage3_verifier.kind.copy()


def search_plan_compatibility_semantics_for_components(
    read reference_scoring_semantics: ReferenceScoringSemantics,
    read stage2_operator: Stage2Operator,
    read stage3_verifier: Stage3VerifierOperator,
) -> SearchPlanCompatibilitySemantics:
    return SearchPlanCompatibilitySemantics(
        reference_scoring_semantics,
        stage2_operator,
        stage3_verifier,
    )


struct SearchPlan(Copyable):
    var candidate_generator: CandidateGenerator
    var candidate_budget: CandidateBudget
    var reference_scoring_semantics: ReferenceScoringSemantics
    var stage2_reference_operator: Stage2ReferenceOperator
    var stage3_verifier: Stage3VerifierOperator
    var stage2_operator: Stage2Operator
    var faithfulness_policy: FaithfulnessPolicy

    def __init__(
        out self,
        candidate_generator: CandidateGenerator,
        candidate_budget: CandidateBudget,
        faithfulness_policy: FaithfulnessPolicy,
        reference_scoring_semantics: ReferenceScoringSemantics,
        stage2_reference_operator: Stage2ReferenceOperator,
        stage3_verifier: Stage3VerifierOperator,
    ) raises:
        if (
            faithfulness_policy.kind == "exact_stage1_required"
            and not candidate_generator.is_exact
        ):
            raise Error(
                "exact_stage1_required faithfulness policy is incompatible with non-exact candidate generation"
            )

        self.candidate_generator = candidate_generator.copy()
        self.candidate_budget = candidate_budget.copy()
        self.reference_scoring_semantics = reference_scoring_semantics.copy()
        self.stage2_reference_operator = stage2_reference_operator.copy()
        self.stage3_verifier = stage3_verifier.copy()
        self.stage2_operator = stage2_operator_for_components(
            self.stage2_reference_operator,
            self.stage3_verifier,
        )
        self.faithfulness_policy = faithfulness_policy.copy()

    def __init__(
        out self,
        candidate_generator: CandidateGenerator,
        candidate_budget: CandidateBudget,
        faithfulness_policy: FaithfulnessPolicy,
        stage2_operator: Stage2Operator,
    ) raises:
        if (
            faithfulness_policy.kind == "exact_stage1_required"
            and not candidate_generator.is_exact
        ):
            raise Error(
                "exact_stage1_required faithfulness policy is incompatible with non-exact candidate generation"
            )

        self.candidate_generator = candidate_generator.copy()
        self.candidate_budget = candidate_budget.copy()
        self.reference_scoring_semantics = (
            reference_scoring_semantics_for_stage2_operator_kind(
                stage2_operator.kind
            )
        )
        self.stage2_reference_operator = (
            stage2_reference_operator_for_stage2_operator_kind(
                stage2_operator.kind
            )
        )
        self.stage3_verifier = stage3_verifier_for_stage2_operator_kind(
            stage2_operator.kind
        )
        self.stage2_operator = stage2_operator.copy()
        self.faithfulness_policy = faithfulness_policy.copy()


def search_plan_compatibility_semantics(
    read plan: SearchPlan
) -> SearchPlanCompatibilitySemantics:
    return search_plan_compatibility_semantics_for_components(
        plan.reference_scoring_semantics,
        plan.stage2_operator,
        plan.stage3_verifier,
    )


def same_search_plan_compatibility_semantics(
    read left: SearchPlanCompatibilitySemantics,
    read right: SearchPlanCompatibilitySemantics,
) -> Bool:
    if left.stage2_kind != right.stage2_kind:
        return False
    if left.stage2_family != right.stage2_family:
        return False
    if left.stage2_requires_query_text != right.stage2_requires_query_text:
        return False
    if len(left.stage2_required_artifact_families) != len(
        right.stage2_required_artifact_families
    ):
        return False
    for index in range(len(left.stage2_required_artifact_families)):
        if (
            left.stage2_required_artifact_families[index]
            != right.stage2_required_artifact_families[index]
        ):
            return False
    if left.exact_stage_kind != right.exact_stage_kind:
        return False
    return left.reranker_kind == right.reranker_kind


def exact_full_scan_search_plan(final_k: Int, candidate_k: Int) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        exact_stage1_required_faithfulness_policy(),
        noop_topk_stage2_operator(),
    )


def exact_full_scan_search_plan(
    final_k: Int,
    candidate_k: Int,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        exact_stage1_required_faithfulness_policy(),
        stage2_operator,
    )


def document_proxy_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def document_proxy_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_flat_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_heads_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_heads_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_heads_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_heads_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_head_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_head_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_head_auto_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_auto_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_head_auto_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_auto_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_blockmax_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_blockmax_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_blockmax_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_blockmax_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_imputed_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_imputed_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def centroid_postings_imputed_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def centroid_postings_imputed_flat_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def gem_graph_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    cluster_top_k_per_query_token: Int = 2,
    beam_width: Int = 32,
) raises -> SearchPlan:
    return SearchPlan(
        gem_graph_candidate_generator(
            cluster_top_k_per_query_token, beam_width
        ),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_stage2_operator(),
    )


def gem_graph_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_operator: Stage2Operator,
    cluster_top_k_per_query_token: Int = 2,
    beam_width: Int = 32,
) raises -> SearchPlan:
    return SearchPlan(
        gem_graph_candidate_generator(
            cluster_top_k_per_query_token, beam_width
        ),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        stage2_operator,
    )


def exact_full_scan_clause_text_search_plan(
    final_k: Int, candidate_k: Int
) raises -> SearchPlan:
    return exact_full_scan_search_plan(
        final_k,
        candidate_k,
        clause_text_stage2_operator(),
    )
