"""Owns explicit local search plans without hiding candidate-generation structure."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate_generator import (
    CandidateGenerator,
    document_proxy_candidate_generator,
    exact_full_scan_candidate_generator,
)


@dataclass(frozen=True, slots=True)
class SearchPlan:
    candidate_generator: CandidateGenerator
    final_k: int
    candidate_k: int

    def __post_init__(self) -> None:
        if self.final_k < 0:
            raise ValueError("final_k must be non-negative")
        if self.candidate_k < 0:
            raise ValueError("candidate_k must be non-negative")
        if self.candidate_k < self.final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")


def exact_full_scan_search_plan(
    final_k: int, *, candidate_k: int | None = None
) -> SearchPlan:
    effective_candidate_k = final_k if candidate_k is None else candidate_k
    return SearchPlan(
        exact_full_scan_candidate_generator(),
        final_k=final_k,
        candidate_k=effective_candidate_k,
    )


def document_proxy_search_plan(
    final_k: int,
    candidate_k: int,
    *,
    query_vector_budget: int = 0,
    document_vector_budget: int = 0,
) -> SearchPlan:
    return SearchPlan(
        document_proxy_candidate_generator(
            query_vector_budget=query_vector_budget,
            document_vector_budget=document_vector_budget,
        ),
        final_k=final_k,
        candidate_k=candidate_k,
    )
