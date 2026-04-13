"""Owns explicit local search plans without hiding staged semantics."""

from __future__ import annotations

from dataclasses import dataclass, field

from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
)
from .reference_scoring_semantics import (
    ReferenceScoringSemantics,
    exact_late_interaction_reference_scoring_semantics,
)
from .stage2_operator import (
    Stage2Operator,
    clause_text_stage2_operator,
    exact_late_interaction_stage2_operator,
    reference_scoring_semantics_for_stage2_operator_kind,
    stage2_operator_for_components,
    stage2_reference_operator_for_stage2_operator_kind,
    stage3_verifier_for_stage2_operator_kind,
    noop_topk_stage2_operator,
)
from .stage2_reference_operator import (
    Stage2ReferenceOperator,
)
from .stage3_verifier_operator import (
    Stage3VerifierOperator,
)


@dataclass(frozen=True, slots=True)
class SearchPlan:
    candidate_generator: CandidateGenerator
    final_k: int
    candidate_k: int
    reference_scoring_semantics: ReferenceScoringSemantics
    stage2_reference_operator: Stage2ReferenceOperator
    stage3_verifier: Stage3VerifierOperator
    stage2_operator: Stage2Operator = field(init=False)
    exact_stage_kind: str = field(init=False)
    reranker_kind: str = field(init=False)

    def __post_init__(self) -> None:
        if self.final_k < 0:
            raise ValueError("final_k must be non-negative")
        if self.candidate_k < 0:
            raise ValueError("candidate_k must be non-negative")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")

        object.__setattr__(
            self,
            "stage2_operator",
            stage2_operator_for_components(
                self.stage2_reference_operator,
                self.stage3_verifier,
            ),
        )
        object.__setattr__(
            self, "exact_stage_kind", self.reference_scoring_semantics.kind
        )
        object.__setattr__(self, "reranker_kind", self.stage3_verifier.kind)


def _plan_from_stage2_operator(
    candidate_generator: CandidateGenerator,
    *,
    final_k: int,
    candidate_k: int,
    stage2_operator: Stage2Operator,
) -> SearchPlan:
    return SearchPlan(
        candidate_generator=candidate_generator,
        final_k=final_k,
        candidate_k=candidate_k,
        reference_scoring_semantics=reference_scoring_semantics_for_stage2_operator_kind(
            stage2_operator.kind
        ),
        stage2_reference_operator=stage2_reference_operator_for_stage2_operator_kind(
            stage2_operator.kind
        ),
        stage3_verifier=stage3_verifier_for_stage2_operator_kind(
            stage2_operator.kind
        ),
    )


def exact_full_scan_search_plan(
    final_k: int,
    *,
    candidate_k: int | None = None,
    stage2_operator: Stage2Operator | None = None,
) -> SearchPlan:
    effective_candidate_k = final_k if candidate_k is None else candidate_k
    effective_stage2_operator = (
        noop_topk_stage2_operator()
        if stage2_operator is None
        else stage2_operator
    )
    return _plan_from_stage2_operator(
        exact_full_scan_candidate_generator(),
        final_k=final_k,
        candidate_k=effective_candidate_k,
        stage2_operator=effective_stage2_operator,
    )


def exact_full_scan_clause_text_search_plan(
    final_k: int, *, candidate_k: int | None = None
) -> SearchPlan:
    return exact_full_scan_search_plan(
        final_k,
        candidate_k=candidate_k,
        stage2_operator=clause_text_stage2_operator(),
    )


def document_proxy_search_plan(
    final_k: int,
    candidate_k: int,
    *,
    query_vector_budget: int = 0,
    document_vector_budget: int = 0,
    stage2_operator: Stage2Operator | None = None,
) -> SearchPlan:
    effective_stage2_operator = (
        exact_late_interaction_stage2_operator()
        if stage2_operator is None
        else stage2_operator
    )
    return _plan_from_stage2_operator(
        document_proxy_candidate_generator(
            query_vector_budget=query_vector_budget,
            document_vector_budget=document_vector_budget,
        ),
        final_k=final_k,
        candidate_k=candidate_k,
        stage2_operator=effective_stage2_operator,
    )
