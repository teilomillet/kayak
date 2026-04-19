# Explicit multi-stage search plan contract.

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
from .graph_frontier_policy import DEFAULT_GRAPH_FRONTIER_POLICY_KIND
from .faithfulness import (
    FaithfulnessPolicy,
    exact_stage1_required_faithfulness_policy,
)
from .reference_scoring_semantics import (
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .stage2_reference_operator import (
    Stage2ReferenceOperator,
    exact_late_interaction_stage2_reference_operator,
    noop_topk_stage2_reference_operator,
)
from .stage3_verifier_operator import (
    Stage3VerifierOperator,
    clause_text_stage3_verifier_operator,
    none_stage3_verifier_operator,
)


struct SearchPlan(Copyable):
    var candidate_generator: CandidateGenerator
    var candidate_budget: CandidateBudget
    var reference_scoring_semantics: ReferenceScoringSemantics
    var stage2_reference_operator: Stage2ReferenceOperator
    var stage3_verifier: Stage3VerifierOperator
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
        self.faithfulness_policy = faithfulness_policy.copy()


def exact_full_scan_search_plan(final_k: Int, candidate_k: Int) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        exact_stage1_required_faithfulness_policy(),
        exact_late_interaction_reference_scoring_semantics(),
        noop_topk_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def exact_full_scan_search_plan(
    final_k: Int,
    candidate_k: Int,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        exact_stage1_required_faithfulness_policy(),
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def document_proxy_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def document_proxy_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_flat_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_heads_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_heads_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_heads_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_heads_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_head_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_head_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_head_auto_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_auto_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_head_auto_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_auto_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_blockmax_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_blockmax_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_blockmax_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_blockmax_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_imputed_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_imputed_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def centroid_postings_imputed_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def centroid_postings_imputed_flat_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def gem_graph_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    cluster_top_k_per_query_token: Int = 2,
    beam_width: Int = 32,
    graph_frontier_policy_kind: String = DEFAULT_GRAPH_FRONTIER_POLICY_KIND,
) raises -> SearchPlan:
    return SearchPlan(
        gem_graph_candidate_generator(
            cluster_top_k_per_query_token,
            beam_width,
            graph_frontier_policy_kind,
        ),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )


def gem_graph_search_plan(
    final_k: Int,
    candidate_k: Int,
    faithfulness_policy: FaithfulnessPolicy,
    stage2_reference_operator: Stage2ReferenceOperator,
    stage3_verifier: Stage3VerifierOperator,
    cluster_top_k_per_query_token: Int = 2,
    beam_width: Int = 32,
    graph_frontier_policy_kind: String = DEFAULT_GRAPH_FRONTIER_POLICY_KIND,
) raises -> SearchPlan:
    return SearchPlan(
        gem_graph_candidate_generator(
            cluster_top_k_per_query_token,
            beam_width,
            graph_frontier_policy_kind,
        ),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        exact_late_interaction_reference_scoring_semantics(),
        stage2_reference_operator,
        stage3_verifier,
    )


def exact_full_scan_clause_text_search_plan(
    final_k: Int, candidate_k: Int
) raises -> SearchPlan:
    return exact_full_scan_search_plan(
        final_k,
        candidate_k,
        noop_topk_stage2_reference_operator(),
        clause_text_stage3_verifier_operator(),
    )
