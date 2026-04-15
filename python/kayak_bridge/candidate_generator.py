"""Owns explicit stage-1 candidate-generator choices for Python search plans."""

from __future__ import annotations

from dataclasses import dataclass, field


EXACT_FULL_SCAN_CANDIDATE_GENERATOR = "exact_full_scan"
DOCUMENT_PROXY_CANDIDATE_GENERATOR = "document_proxy"

SUPPORTED_CANDIDATE_GENERATORS = (
    EXACT_FULL_SCAN_CANDIDATE_GENERATOR,
    DOCUMENT_PROXY_CANDIDATE_GENERATOR,
)


@dataclass(frozen=True, slots=True)
class CandidateGenerator:
    """One explicit stage-1 candidate-generator choice with vector budgets."""

    kind: str
    query_vector_budget: int = 0
    document_vector_budget: int = 0
    family: str = field(init=False)
    interaction_semantics: str = field(init=False)
    alignment_granularity: str = field(init=False)
    score_kind: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in SUPPORTED_CANDIDATE_GENERATORS:
            raise ValueError(f"unsupported candidate generator: {self.kind}")
        if self.query_vector_budget < 0:
            raise ValueError("query_vector_budget must be non-negative")
        if self.document_vector_budget < 0:
            raise ValueError("document_vector_budget must be non-negative")
        if self.kind == EXACT_FULL_SCAN_CANDIDATE_GENERATOR and (
            self.query_vector_budget != 0 or self.document_vector_budget != 0
        ):
            raise ValueError(
                "exact_full_scan does not accept query or document vector budgets"
            )
        if self.kind == EXACT_FULL_SCAN_CANDIDATE_GENERATOR:
            family = "exact"
            interaction_semantics = "exact_late_interaction"
            alignment_granularity = "document_tokens"
            score_kind = "exact_score"
        else:
            family = "proxy"
            interaction_semantics = "none"
            alignment_granularity = "document"
            score_kind = "proxy_score"

        object.__setattr__(self, "family", family)
        object.__setattr__(
            self, "interaction_semantics", interaction_semantics
        )
        object.__setattr__(self, "alignment_granularity", alignment_granularity)
        object.__setattr__(self, "score_kind", score_kind)


def exact_full_scan_candidate_generator() -> CandidateGenerator:
    """Return the exact full-scan candidate generator."""
    return CandidateGenerator(EXACT_FULL_SCAN_CANDIDATE_GENERATOR)


def document_proxy_candidate_generator(
    *,
    query_vector_budget: int = 0,
    document_vector_budget: int = 0,
) -> CandidateGenerator:
    """Return the document-proxy candidate generator with explicit budgets."""
    return CandidateGenerator(
        DOCUMENT_PROXY_CANDIDATE_GENERATOR,
        query_vector_budget=query_vector_budget,
        document_vector_budget=document_vector_budget,
    )
