# Explicit multi-stage search plan contract.

from .candidate_budget import CandidateBudget
from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
)


struct SearchPlan(Copyable):
    var candidate_generator: CandidateGenerator
    var candidate_budget: CandidateBudget
    var exact_stage_kind: String
    var reranker_kind: String

    def __init__(
        out self,
        candidate_generator: CandidateGenerator,
        candidate_budget: CandidateBudget,
        var reranker_kind: String,
    ) raises:
        if reranker_kind != "none":
            raise Error("unknown reranker kind: " + reranker_kind)

        self.candidate_generator = candidate_generator.copy()
        self.candidate_budget = candidate_budget.copy()
        self.exact_stage_kind = "exact_late_interaction"
        self.reranker_kind = reranker_kind^


def exact_full_scan_search_plan(final_k: Int, candidate_k: Int) raises -> SearchPlan:
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        "none",
    )


def document_proxy_search_plan(final_k: Int, candidate_k: Int) raises -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(),
        CandidateBudget(final_k, candidate_k),
        "none",
    )
