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
)
from .faithfulness import (
    FaithfulnessPolicy,
    exact_stage1_required_faithfulness_policy,
)


struct SearchPlan(Copyable):
    var candidate_generator: CandidateGenerator
    var candidate_budget: CandidateBudget
    var exact_stage_kind: String
    var reranker_kind: String
    var faithfulness_policy: FaithfulnessPolicy

    def __init__(
        out self,
        candidate_generator: CandidateGenerator,
        candidate_budget: CandidateBudget,
        faithfulness_policy: FaithfulnessPolicy,
        var reranker_kind: String,
    ) raises:
        if reranker_kind != "none":
            raise Error("unknown reranker kind: " + reranker_kind)

        if (
            faithfulness_policy.kind == "exact_stage1_required"
            and candidate_generator.kind != "exact_full_scan"
        ):
            raise Error(
                "exact_stage1_required faithfulness policy is incompatible with non-exact candidate generation"
            )

        self.candidate_generator = candidate_generator.copy()
        self.candidate_budget = candidate_budget.copy()
        self.exact_stage_kind = "exact_late_interaction"
        self.reranker_kind = reranker_kind^
        self.faithfulness_policy = faithfulness_policy.copy()


def exact_full_scan_search_plan(final_k: Int, candidate_k: Int) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        exact_stage1_required_faithfulness_policy(),
        "none",
    )


def document_proxy_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_heads_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_heads_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_head_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_head_auto_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_head_auto_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_blockmax_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_blockmax_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_imputed_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )


def centroid_postings_imputed_flat_search_plan(
    final_k: Int, candidate_k: Int, faithfulness_policy: FaithfulnessPolicy
) raises -> SearchPlan:
    return SearchPlan(
        centroid_postings_imputed_flat_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        faithfulness_policy,
        "none",
    )
