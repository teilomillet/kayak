"""Owns explicit local search plans without compatibility aliases."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
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


@dataclass(frozen=True, slots=True)
class SearchPlan:
    """One explicit staged retrieval plan from candidate generation to verifier."""

    candidate_generator: CandidateGenerator
    final_k: int
    candidate_k: int
    reference_scoring_semantics: ReferenceScoringSemantics
    stage2_reference_operator: Stage2ReferenceOperator
    stage3_verifier: Stage3VerifierOperator

    def __post_init__(self) -> None:
        if self.final_k < 0:
            raise ValueError("final_k must be non-negative")
        if self.candidate_k < 0:
            raise ValueError("candidate_k must be non-negative")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")


def _resolve_stage_components(
    *,
    default_reference_scoring_semantics: ReferenceScoringSemantics,
    default_stage2_reference_operator: Stage2ReferenceOperator,
    default_stage3_verifier: Stage3VerifierOperator,
    stage2_reference_operator: Stage2ReferenceOperator | None,
    stage3_verifier: Stage3VerifierOperator | None,
) -> tuple[
    ReferenceScoringSemantics,
    Stage2ReferenceOperator,
    Stage3VerifierOperator,
]:
    return (
        default_reference_scoring_semantics,
        default_stage2_reference_operator
        if stage2_reference_operator is None
        else stage2_reference_operator,
        default_stage3_verifier if stage3_verifier is None else stage3_verifier,
    )


def exact_full_scan_search_plan(
    final_k: int,
    *,
    candidate_k: int | None = None,
    stage2_reference_operator: Stage2ReferenceOperator | None = None,
    stage3_verifier: Stage3VerifierOperator | None = None,
) -> SearchPlan:
    """Build the exact full-scan correctness baseline plan."""
    effective_candidate_k = final_k if candidate_k is None else candidate_k
    (
        reference_scoring_semantics,
        effective_stage2_reference_operator,
        effective_stage3_verifier,
    ) = _resolve_stage_components(
        default_reference_scoring_semantics=(
            exact_late_interaction_reference_scoring_semantics()
        ),
        default_stage2_reference_operator=noop_topk_stage2_reference_operator(),
        default_stage3_verifier=none_stage3_verifier_operator(),
        stage2_reference_operator=stage2_reference_operator,
        stage3_verifier=stage3_verifier,
    )
    return SearchPlan(
        candidate_generator=exact_full_scan_candidate_generator(),
        final_k=final_k,
        candidate_k=effective_candidate_k,
        reference_scoring_semantics=reference_scoring_semantics,
        stage2_reference_operator=effective_stage2_reference_operator,
        stage3_verifier=effective_stage3_verifier,
    )


def exact_full_scan_clause_text_search_plan(
    final_k: int, *, candidate_k: int | None = None
) -> SearchPlan:
    """Build an exact full-scan plan with clause-text verification enabled."""
    return exact_full_scan_search_plan(
        final_k,
        candidate_k=candidate_k,
        stage3_verifier=clause_text_stage3_verifier_operator(),
    )


def document_proxy_search_plan(
    final_k: int,
    candidate_k: int,
    *,
    query_vector_budget: int = 0,
    document_vector_budget: int = 0,
    stage2_reference_operator: Stage2ReferenceOperator | None = None,
    stage3_verifier: Stage3VerifierOperator | None = None,
) -> SearchPlan:
    """Build a proxy-candidate plus exact-rerank staged retrieval plan."""
    (
        reference_scoring_semantics,
        effective_stage2_reference_operator,
        effective_stage3_verifier,
    ) = _resolve_stage_components(
        default_reference_scoring_semantics=(
            exact_late_interaction_reference_scoring_semantics()
        ),
        default_stage2_reference_operator=(
            exact_late_interaction_stage2_reference_operator()
        ),
        default_stage3_verifier=none_stage3_verifier_operator(),
        stage2_reference_operator=stage2_reference_operator,
        stage3_verifier=stage3_verifier,
    )
    return SearchPlan(
        candidate_generator=document_proxy_candidate_generator(
            query_vector_budget=query_vector_budget,
            document_vector_budget=document_vector_budget,
        ),
        final_k=final_k,
        candidate_k=candidate_k,
        reference_scoring_semantics=reference_scoring_semantics,
        stage2_reference_operator=effective_stage2_reference_operator,
        stage3_verifier=effective_stage3_verifier,
    )
