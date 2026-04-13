"""Runs explicit two-stage local search plans for the Python SDK."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_stage import CandidateStageResult, generate_candidates
from .clause_text import clause_text_scores
from .late_scores import LateScores, SearchHit
from .layouts import NUMPY_REFERENCE_BACKEND
from .search_plan import SearchPlan
from .search_stage_profile import SearchStageProfile


@dataclass(frozen=True, slots=True)
class SearchPlanResult:
    plan: SearchPlan
    candidate_stage: CandidateStageResult
    candidate_index: "LateIndex | None"
    stage2_scores: LateScores | None
    hits: tuple[SearchHit, ...]
    stage2: SearchStageProfile

    def __post_init__(self) -> None:
        if len(self.hits) != self.stage2.output_hit_count:
            raise ValueError("final hits must match the stage-2 profile")

    @property
    def exact_scores(self) -> LateScores | None:
        return self.stage2_scores

    @property
    def exact_stage(self) -> SearchStageProfile:
        return self.stage2


def _empty_stage2_result(plan: SearchPlan) -> SearchStageProfile:
    return SearchStageProfile(
        stage_name=plan.stage2_operator.kind,
        input_hit_count=0,
        output_hit_count=0,
        query_vector_count=0,
        document_count=0,
        document_vector_count=0,
        document_text_count=0,
    )


def _noop_stage2_scores(candidate_stage: CandidateStageResult) -> LateScores:
    import numpy as np

    values = np.array(
        [hit.score for hit in candidate_stage.hits],
        dtype=np.float32,
    )
    return LateScores.from_values(
        "noop_topk",
        candidate_stage.candidate_doc_ids,
        values,
    )


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
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=None,
            stage2_scores=None,
            hits=(),
            stage2=_empty_stage2_result(plan),
        )

    if plan.stage2_operator.kind == "noop_topk":
        stage2_scores = _noop_stage2_scores(candidate_stage)
        hits = stage2_scores.topk(plan.final_k)
        stage2 = SearchStageProfile(
            stage_name=plan.stage2_operator.kind,
            input_hit_count=len(candidate_stage.hits),
            output_hit_count=len(hits),
            query_vector_count=0,
            document_count=len(candidate_stage.hits),
            document_vector_count=0,
            document_text_count=0,
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=None,
            stage2_scores=stage2_scores,
            hits=hits,
            stage2=stage2,
        )

    candidate_index = candidate_stage.select_index(index)

    if plan.stage2_operator.kind == "exact_late_interaction":
        stage2_scores = candidate_index.maxsim(query, backend=backend)
        hits = stage2_scores.topk(plan.final_k)
        stage2 = SearchStageProfile(
            stage_name=plan.stage2_operator.kind,
            input_hit_count=len(candidate_stage.hits),
            output_hit_count=len(hits),
            query_vector_count=query.vector_count,
            document_count=candidate_index.document_count,
            document_vector_count=candidate_index.total_vector_count,
            document_text_count=0,
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=candidate_index,
            stage2_scores=stage2_scores,
            hits=hits,
            stage2=stage2,
        )

    if plan.stage2_operator.kind == "clause_text":
        if not query.text:
            raise ValueError("clause_text stage-2 requires query.text")
        if candidate_index.doc_texts is None:
            raise ValueError(
                "clause_text stage-2 requires document texts on the candidate index"
            )

        stage2_scores = clause_text_scores(
            query.text,
            candidate_stage.hits,
            candidate_index.doc_texts,
        )
        hits = stage2_scores.topk(plan.final_k)
        stage2 = SearchStageProfile(
            stage_name=plan.stage2_operator.kind,
            input_hit_count=len(candidate_stage.hits),
            output_hit_count=len(hits),
            query_vector_count=0,
            document_count=candidate_index.document_count,
            document_vector_count=0,
            document_text_count=len(candidate_index.doc_texts),
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=candidate_index,
            stage2_scores=stage2_scores,
            hits=hits,
            stage2=stage2,
        )

    raise ValueError(
        f"unsupported stage-2 operator: {plan.stage2_operator.kind}"
    )
