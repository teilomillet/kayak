"""Runs explicit staged local search plans for the Python SDK."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_stage import CandidateStageResult, generate_candidates
from .clause_text import clause_text_scores
from .late_scores import LateScores, SearchHit
from .layouts import NUMPY_REFERENCE_BACKEND
from .search_plan import SearchPlan
from .search_stage_profile import SearchStageProfile
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage2_reference_operator import (
    STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
)
from .stage3_verifier_operator import STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT


@dataclass(frozen=True, slots=True)
class SearchPlanResult:
    """One executed search plan result with per-stage outputs and final hits."""

    plan: SearchPlan
    candidate_stage: CandidateStageResult
    candidate_index: "LateIndex | None"
    reference_scores: LateScores | None
    stage3_scores: LateScores | None
    hits: tuple[SearchHit, ...]
    stage2_reference: SearchStageProfile
    stage3_verifier: SearchStageProfile

    def __post_init__(self) -> None:
        if len(self.hits) != self.stage3_verifier.output_hit_count:
            raise ValueError("final hits must match the stage-3 profile")

    @property
    def stage2_scores(self) -> LateScores | None:
        return self.reference_scores

    @property
    def exact_scores(self) -> LateScores | None:
        return self.reference_scores

    @property
    def stage2(self) -> SearchStageProfile:
        return self.stage2_reference

    @property
    def exact_stage(self) -> SearchStageProfile:
        return self.stage2_reference


def _empty_stage_profile(stage_name: str) -> SearchStageProfile:
    return SearchStageProfile(
        stage_name=stage_name,
        input_hit_count=0,
        output_hit_count=0,
        query_vector_count=0,
        document_count=0,
        document_vector_count=0,
        document_text_count=0,
    )


def _noop_reference_scores(
    candidate_stage: CandidateStageResult, *, limit: int
) -> LateScores:
    import numpy as np

    values = np.array(
        [hit.score for hit in candidate_stage.hits[:limit]],
        dtype=np.float32,
    )
    doc_ids = candidate_stage.candidate_doc_ids[:limit]
    return LateScores.from_values(
        "noop_topk",
        doc_ids,
        values,
    )


def _hits_for_scores(scores: LateScores) -> tuple[SearchHit, ...]:
    return tuple(
        SearchHit(doc_id=doc_id, score=float(score))
        for doc_id, score in zip(scores.doc_ids, scores.values, strict=True)
    )


def _reference_output_k(plan: SearchPlan, candidate_count: int) -> int:
    if plan.stage3_verifier.kind == "none":
        return min(plan.final_k, candidate_count)
    return min(plan.candidate_k, candidate_count)


def _document_vector_count_for_doc_ids(
    candidate_index: "LateIndex", doc_ids: tuple[str, ...]
) -> int:
    positions = {
        doc_id: index for index, doc_id in enumerate(candidate_index.doc_ids)
    }
    vector_counts = candidate_index.vector_counts
    return sum(vector_counts[positions[doc_id]] for doc_id in doc_ids)


def _identity_stage3_profile(
    stage_name: str, *, input_hit_count: int
) -> SearchStageProfile:
    return SearchStageProfile(
        stage_name=stage_name,
        input_hit_count=input_hit_count,
        output_hit_count=input_hit_count,
        query_vector_count=0,
        document_count=input_hit_count,
        document_vector_count=0,
        document_text_count=0,
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
            reference_scores=None,
            stage3_scores=None,
            hits=(),
            stage2_reference=_empty_stage_profile(
                plan.stage2_reference_operator.kind
            ),
            stage3_verifier=_empty_stage_profile(plan.stage3_verifier.kind),
        )

    reference_limit = _reference_output_k(plan, len(candidate_stage.hits))
    candidate_index: "LateIndex | None" = None
    reference_scores: LateScores

    if plan.stage2_reference_operator.kind == "noop_topk":
        reference_scores = _noop_reference_scores(
            candidate_stage,
            limit=reference_limit,
        )
        reference_stage = SearchStageProfile(
            stage_name=plan.stage2_reference_operator.kind,
            input_hit_count=len(candidate_stage.hits),
            output_hit_count=len(reference_scores.doc_ids),
            query_vector_count=0,
            document_count=len(reference_scores.doc_ids),
            document_vector_count=0,
            document_text_count=0,
        )
    elif plan.stage2_reference_operator.kind == "exact_late_interaction":
        candidate_index = candidate_stage.select_index(index)
        exact_scores = candidate_index.maxsim(query, backend=backend)
        if reference_limit == len(exact_scores.doc_ids):
            reference_scores = exact_scores
        else:
            reference_hits = exact_scores.topk(reference_limit)
            import numpy as np

            reference_scores = LateScores.from_values(
                exact_scores.backend,
                tuple(hit.doc_id for hit in reference_hits),
                np.array(
                    [hit.score for hit in reference_hits],
                    dtype=np.float32,
                ),
            )
        reference_stage = SearchStageProfile(
            stage_name=plan.stage2_reference_operator.kind,
            input_hit_count=len(candidate_stage.hits),
            output_hit_count=len(reference_scores.doc_ids),
            query_vector_count=query.vector_count,
            document_count=len(reference_scores.doc_ids),
            document_vector_count=(
                0
                if candidate_index is None
                else _document_vector_count_for_doc_ids(
                    candidate_index,
                    reference_scores.doc_ids,
                )
            ),
            document_text_count=0,
            materialized_artifacts=(
                StageArtifactMaterialization(
                    family=STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
                    document_count=len(reference_scores.doc_ids),
                    document_vector_count=(
                        0
                        if candidate_index is None
                        else _document_vector_count_for_doc_ids(
                            candidate_index,
                            reference_scores.doc_ids,
                        )
                    ),
                ),
            ),
        )
    else:
        raise ValueError(
            f"unsupported stage-2 reference operator: {plan.stage2_reference_operator.kind}"
        )

    reference_hits = reference_scores.topk(len(reference_scores.doc_ids))

    if plan.stage3_verifier.kind == "none":
        final_hits = reference_hits[: plan.final_k]
        stage3_profile = _identity_stage3_profile(
            plan.stage3_verifier.kind,
            input_hit_count=len(reference_hits),
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=candidate_index,
            reference_scores=reference_scores,
            stage3_scores=None,
            hits=final_hits,
            stage2_reference=reference_stage,
            stage3_verifier=stage3_profile,
        )

    if candidate_index is None:
        candidate_index = candidate_stage.select_index(index)

    if plan.stage3_verifier.kind == "clause_text":
        if not query.text:
            raise ValueError("clause_text stage-3 requires query.text")
        if candidate_index.doc_texts is None:
            raise ValueError(
                "clause_text stage-3 requires document texts on the candidate index"
            )

        doc_texts_by_id = {
            doc_id: text
            for doc_id, text in zip(
                candidate_index.doc_ids, candidate_index.doc_texts, strict=True
            )
        }
        ordered_texts = tuple(
            doc_texts_by_id[hit.doc_id]
            for hit in reference_hits
        )
        stage3_scores = clause_text_scores(
            query.text,
            reference_hits,
            ordered_texts,
            backend="clause_text",
        )
        final_hits = stage3_scores.topk(plan.final_k)
        stage3_profile = SearchStageProfile(
            stage_name=plan.stage3_verifier.kind,
            input_hit_count=len(reference_hits),
            output_hit_count=len(final_hits),
            query_vector_count=0,
            document_count=len(reference_hits),
            document_vector_count=0,
            document_text_count=len(ordered_texts),
            materialized_artifacts=(
                StageArtifactMaterialization(
                    family=STAGE3_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
                    document_count=len(ordered_texts),
                    document_text_count=len(ordered_texts),
                ),
            ),
        )
        return SearchPlanResult(
            plan=plan,
            candidate_stage=candidate_stage,
            candidate_index=candidate_index,
            reference_scores=reference_scores,
            stage3_scores=stage3_scores,
            hits=final_hits,
            stage2_reference=reference_stage,
            stage3_verifier=stage3_profile,
        )

    raise ValueError(
        f"unsupported stage-3 verifier: {plan.stage3_verifier.kind}"
    )
