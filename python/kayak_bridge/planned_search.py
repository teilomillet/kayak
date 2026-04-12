"""Runs explicit two-stage local search plans for the Python SDK."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_stage import CandidateStageResult, generate_candidates
from .late_scores import LateScores, SearchHit
from .layouts import NUMPY_REFERENCE_BACKEND
from .search_plan import SearchPlan
from .search_stage_profile import SearchStageProfile


@dataclass(frozen=True, slots=True)
class SearchPlanResult:
    plan: SearchPlan
    candidate_stage: CandidateStageResult
    candidate_index: "LateIndex | None"
    exact_scores: LateScores | None
    hits: tuple[SearchHit, ...]
    exact_stage: SearchStageProfile

    def __post_init__(self) -> None:
        if len(self.hits) != self.exact_stage.output_hit_count:
            raise ValueError("final hits must match the exact-stage profile")


def search_with_plan(
    query: "LateQuery",
    index: "LateIndex",
    plan: SearchPlan,
    *,
    backend: str = NUMPY_REFERENCE_BACKEND,
) -> SearchPlanResult:
    candidate_stage = generate_candidates(
        query,
        index,
        plan.candidate_generator,
        k=plan.candidate_k,
        backend=backend,
    )

    if not candidate_stage.hits:
        exact_stage = SearchStageProfile(
            stage_name="exact_late_interaction",
            input_hit_count=0,
            output_hit_count=0,
            query_vector_count=query.vector_count,
            document_count=0,
            document_vector_count=0,
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=None,
            exact_scores=None,
            hits=(),
            exact_stage=exact_stage,
        )

    candidate_index = candidate_stage.select_index(index)
    exact_scores = candidate_index.maxsim(query, backend=backend)
    hits = exact_scores.topk(plan.final_k)
    exact_stage = SearchStageProfile(
        stage_name="exact_late_interaction",
        input_hit_count=len(candidate_stage.hits),
        output_hit_count=len(hits),
        query_vector_count=query.vector_count,
        document_count=candidate_index.document_count,
        document_vector_count=candidate_index.total_vector_count,
    )
    return SearchPlanResult(
        plan=plan,
        candidate_stage=candidate_stage,
        candidate_index=candidate_index,
        exact_scores=exact_scores,
        hits=hits,
        exact_stage=exact_stage,
    )
