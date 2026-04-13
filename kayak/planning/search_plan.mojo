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
from .faithfulness import (
    FaithfulnessPolicy,
    exact_stage1_required_faithfulness_policy,
)
from .stage2_operator import (
    Stage2Operator,
    clause_text_stage2_operator,
    exact_late_interaction_stage2_operator,
    noop_topk_stage2_operator,
)


struct SearchPlan(Copyable):
    var candidate_generator: CandidateGenerator
    var candidate_budget: CandidateBudget
    var stage2_operator: Stage2Operator
    var exact_stage_kind: String
    var reranker_kind: String
    var faithfulness_policy: FaithfulnessPolicy

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
        self.stage2_operator = stage2_operator.copy()
        self.exact_stage_kind = (
            self.stage2_operator.compatibility_exact_stage_kind.copy()
        )
        self.reranker_kind = (
            self.stage2_operator.compatibility_reranker_kind.copy()
        )
        self.faithfulness_policy = faithfulness_policy.copy()


def search_plan_with_stage2_operator(
    read plan: SearchPlan,
    stage2_operator: Stage2Operator,
) raises -> SearchPlan:
    return SearchPlan(
        plan.candidate_generator,
        plan.candidate_budget,
        plan.faithfulness_policy,
        stage2_operator,
    )


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
