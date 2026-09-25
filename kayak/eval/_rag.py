"""Assess caller-recorded RAG boundaries without running or judging a pipeline."""

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._ranking import RankedOutput, RankingAssessment, _check_ids, assess_ranking
from ._schema import digest

RAGStage = Literal["routing", "retrieval", "reranking", "context", "answer"]
StageStatus = Literal["observed", "error", "blocked", "not_observed"]
_STAGES: tuple[RAGStage, ...] = ("routing", "retrieval", "reranking", "context", "answer")


class _Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class RAGContext(_Record):
    """Original source IDs and the exact rendered context; IDs alone prove no text retention."""

    ids: list[str]
    text: str

    @model_validator(mode="after")
    def valid_ids(self) -> Self:
        _check_ids(self.ids)
        return self


class RAGReplay(_Record):
    """A declared, caller-run diagnostic intervention, not an execution or causal proof."""

    original_trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    changed_boundary: RAGStage
    purpose: str = Field(min_length=1)


class RAGTrace(_Record):
    """Optional observations retain what ran; missing stages are never inferred.

    Reranking is a complete permutation of its known input by default. Set
    reranking_complete=False when only its best-first prefix was recorded.
    Explicit input IDs allow partial traces without manufacturing an earlier
    stage. Context may select a subset of observed results. Optional document
    snapshots cover all observed document IDs; IDs and provenance alone cannot
    reproduce model execution.
    """

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    documents: dict[str, str] | None = None
    route_candidates: list[str] | None = None
    route: str | None = None
    retrieval: RankedOutput | None = None
    reranking_input_ids: list[str] | None = None
    reranking: RankedOutput | None = None
    reranking_complete: bool = True
    context_input_ids: list[str] | None = None
    context: RAGContext | None = None
    answer: str | None = None
    errors: dict[RAGStage, str] = Field(default_factory=dict)
    blocked: dict[RAGStage, str] = Field(default_factory=dict)
    provenance: dict[str, str] = Field(default_factory=dict)
    replay: RAGReplay | None = None

    @field_validator("id", "query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trace ID and query must not be blank")
        return value

    @model_validator(mode="after")
    def valid_observations(self) -> Self:
        outputs = (self.route, self.retrieval, self.reranking, self.context, self.answer)
        for stage, output in zip(_STAGES, outputs, strict=True):
            if sum((output is not None, stage in self.errors, stage in self.blocked)) > 1:
                raise ValueError(f"{stage} cannot have both an output and another outcome")
        if any(not reason.strip() for reason in (*self.errors.values(), *self.blocked.values())):
            raise ValueError("error types and blocked reasons must not be blank")
        for identifiers in (
            self.route_candidates,
            self.reranking_input_ids,
            self.context_input_ids,
        ):
            if identifiers is not None:
                _check_ids(identifiers)
        if self.route is not None:
            _check_ids([self.route])
            if self.route_candidates is not None and self.route not in self.route_candidates:
                raise ValueError("selected route is absent from the routing candidates")
        if not self.reranking_complete and self.reranking is None:
            raise ValueError("an incomplete reranking requires an observed prefix")
        self._check_document_lineage()
        return self

    def _check_document_lineage(self) -> None:
        retrieved = self.retrieval.ids if self.retrieval is not None else None
        rank_input = self.ranking_input()
        if (
            retrieved is not None
            and rank_input is not None
            and not set(rank_input) <= set(retrieved)
        ):
            raise ValueError("reranking input contains an ID absent from retrieval")
        if self.reranking is not None:
            if rank_input is None:
                raise ValueError("reranking must be a permutation of its known input IDs")
            if self.reranking_complete and set(self.reranking.ids) != set(rank_input):
                raise ValueError("reranking must be a permutation of its known input IDs")
            if not set(self.reranking.ids) <= set(rank_input):
                raise ValueError("reranking prefix contains an ID absent from its known input")
        prior_context = self.reranking.ids if self.reranking is not None else retrieved
        if prior_context is not None and self.context_input_ids is not None:
            if not set(self.context_input_ids) <= set(prior_context):
                raise ValueError(
                    "context input contains an ID absent from its recorded predecessor"
                )
        context_input = self.packing_input()
        if self.context is not None:
            if context_input is None or not set(self.context.ids) <= set(context_input):
                raise ValueError("context sources must belong to its known input IDs")
        if self.documents is not None:
            _check_ids(list(self.documents))
            observed = [retrieved, rank_input, context_input]
            if any(ids is not None and not set(ids) <= self.documents.keys() for ids in observed):
                raise ValueError("document snapshots must cover every observed document ID")

    def ranking_input(self) -> list[str] | None:
        """Return recorded ranking input without claiming retrieval executed."""
        if self.reranking_input_ids is not None:
            return self.reranking_input_ids
        return self.retrieval.ids if self.retrieval is not None else None

    def packing_input(self) -> list[str] | None:
        """An explicit input supports partial traces; otherwise use the recorded predecessor."""
        if self.context_input_ids is not None:
            return self.context_input_ids
        if self.reranking is not None:
            return self.reranking.ids
        return self.retrieval.ids if self.retrieval is not None else None

    @property
    def sha256(self) -> str:
        """Bind exact recorded inputs, outputs, ordering, and declared provenance."""
        recorded = self.model_dump(mode="json")
        if self.reranking_complete:
            # The additive default must not invalidate existing trace reviews.
            del recorded["reranking_complete"]
        return digest(recorded)

    def evaluate(
        self,
        *,
        expected_answer: str | None = None,
        expected_sources: Sequence[str] | None = None,
        k: int = 5,
    ) -> "RAGAssessment":
        """Check a reference answer and jointly required sources without running a judge.

        Answer matching strips surrounding whitespace and remains case-sensitive.
        It does not establish grounding or semantic equivalence. Omitted labels
        and an unobserved answer remain unknown. Other sources remain unjudged.
        `k` is the ranking assessment cutoff, independent of context packing.
        Use assess_rag directly for graded relevance, alternative evidence sets,
        or independently supplied correctness/grounding reviews.
        """
        if expected_answer is not None and not isinstance(expected_answer, str):
            raise ValueError("expected_answer must be a string or None")
        sources = []
        if expected_sources is not None:
            _check_ids(expected_sources)
            sources = list(expected_sources)
        review = None
        if expected_answer is not None and self.answer is not None:
            review = RAGAnswerReview(
                trace_sha256=self.sha256,
                correct=self.answer.strip() == expected_answer.strip(),
                provenance={
                    "rubric": "stripped-exact-match-v1",
                    "expected_answer": expected_answer,
                },
            )
        judgments = RAGJudgments(
            relevance={identifier: 1 for identifier in sources},
            evidence_sets=[sources] if sources else [],
            answer=review,
        )
        return assess_rag(self, judgments, k=k)


class RAGAnswerReview(_Record):
    """Independent caller judgments bound to one exact trace, not an automatic judge."""

    trace_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    correct: bool | None = None
    grounded: bool | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_judgment(self) -> Self:
        if self.correct is None and self.grounded is None:
            raise ValueError("an answer review needs a correctness or grounding judgment")
        return self


class RAGJudgments(_Record):
    """Gold is separate from execution. Evidence routes are alternatives of required ID sets.

    Positive relevance grades are known positives; absent IDs are unjudged.
    Optional evidence_texts hold exact required text, not inferred facts. Empty
    evidence_sets means no known route, never a vacuously sufficient context.
    """

    relevance: dict[str, int] = Field(default_factory=dict)
    evidence_sets: list[list[str]] = Field(default_factory=list)
    evidence_texts: dict[str, str] = Field(default_factory=dict)
    acceptable_routes: list[str] | None = None
    answer: RAGAnswerReview | None = None

    @model_validator(mode="after")
    def valid_judgments(self) -> Self:
        _check_ids(list(self.relevance))
        if any(grade < 0 for grade in self.relevance.values()):
            raise ValueError("relevance grades must be nonnegative integers")
        routes: set[frozenset[str]] = set()
        for identifiers in self.evidence_sets:
            _check_ids(identifiers)
            route = frozenset(identifiers)
            if not route or route in routes:
                raise ValueError("evidence routes must be nonempty and distinct")
            if any(self.relevance.get(identifier) == 0 for identifier in route):
                raise ValueError("a required evidence source cannot be judged irrelevant")
            routes.add(route)
        known_sources = {identifier for route in routes for identifier in route}
        if not self.evidence_texts.keys() <= known_sources:
            raise ValueError("required texts must belong to an evidence route")
        if any(not text.strip() for text in self.evidence_texts.values()):
            raise ValueError("required evidence text must not be blank")
        if self.acceptable_routes is not None:
            _check_ids(self.acceptable_routes)
            if not self.acceptable_routes:
                raise ValueError("omit unjudged routing labels instead of supplying an empty list")
        return self


class RAGAssessment(_Record):
    """Boundary comparisons, never causal blame or automatic semantic judgments.

    Source coverage and literal required-text retention are independent.
    ordinary_aggregate_eligible excludes declared diagnostic interventions; it
    does not certify a complete trace or turn unknown quality into success.
    """

    trace_sha256: str
    judgments_sha256: str
    stage_status: dict[RAGStage, StageStatus]
    execution_errors: list[RAGStage]
    route_correct: bool | None
    retrieval: RankingAssessment | None
    reranking: RankingAssessment | None
    shortlist_reranking: RankingAssessment | None
    source_coverage: dict[str, bool | None]
    required_text_retained: bool | None
    answer_correct: bool | None
    answer_grounded: bool | None
    observed_losses: dict[str, list[str]]
    ordinary_aggregate_eligible: bool


def _source_coverage(ids: Sequence[str] | None, routes: list[list[str]]) -> bool | None:
    if ids is None or not routes:
        return None
    available = set(ids)
    return any(set(route) <= available for route in routes)


def _reranking_coverage(trace: RAGTrace, routes: list[list[str]], k: int) -> bool | None:
    """An unseen tail may complete a route only within the known input and cutoff."""
    if trace.reranking is None:
        return None
    observed = trace.reranking.ids
    coverage = _source_coverage(observed[:k], routes)
    if coverage is not False or trace.reranking_complete or k <= len(observed):
        return coverage
    possible = set(trace.ranking_input() or [])
    available = set(observed)
    remaining_slots = k - len(observed)
    for route in routes:
        required = set(route)
        if required <= possible and len(required - available) <= remaining_slots:
            return None
    return False


def _text_retained(context: RAGContext | None, judgments: RAGJudgments) -> bool | None:
    if context is None or not judgments.evidence_sets:
        return None
    unknown = False
    for route in judgments.evidence_sets:
        known = [
            judgments.evidence_texts[identifier]
            for identifier in route
            if identifier in judgments.evidence_texts
        ]
        if any(text not in context.text for text in known):
            continue
        if len(known) == len(route):
            return True
        unknown = True
    return None if unknown else False


def _losses(trace: RAGTrace, judgments: RAGJudgments, k: int) -> dict[str, list[str]]:
    positive = {identifier for identifier, grade in judgments.relevance.items() if grade > 0}
    losses: dict[str, list[str]] = {}
    if trace.retrieval is not None:
        losses["retrieval_missing_known_positives"] = sorted(positive - set(trace.retrieval.ids))
        if trace.reranking_input_ids is not None:
            losses["reranking_input_omitted_known_positives"] = sorted(
                (positive & set(trace.retrieval.ids)) - set(trace.reranking_input_ids)
            )
    if trace.reranking is not None:
        below = set(trace.reranking.ids[k:])
        losses["reranking_below_k_known_positives"] = sorted(positive & below)
        if not trace.reranking_complete:
            unreturned = set(trace.ranking_input() or []) - set(trace.reranking.ids)
            losses["reranking_unreturned_known_positives"] = sorted(positive & unreturned)
    if trace.context is not None:
        available = set(trace.packing_input() or [])
        losses["context_omitted_available_known_positives"] = sorted(
            (positive & available) - set(trace.context.ids)
        )
        # Another complete evidence route may still be present in the context.
        losses["context_absent_labeled_text"] = [
            identifier
            for identifier, text in judgments.evidence_texts.items()
            if text not in trace.context.text
        ]
    return losses


def assess_rag(trace: RAGTrace, judgments: RAGJudgments, *, k: int = 5) -> RAGAssessment:
    """Pure assessment of recorded boundaries; never call a model, route, or answer judge.

    Corpus metrics refer to known positives in relevance, not necessarily every
    true corpus positive. shortlist_reranking conditions on the recorded ranking
    input. Text retention checks literal content, not semantic sufficiency.
    Diagnostic traces are marked ineligible for ordinary aggregate quality.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    # Frozen records still contain mutable containers. Revalidate and detach them.
    trace = RAGTrace.model_validate(trace.model_dump())
    judgments = RAGJudgments.model_validate(judgments.model_dump())
    trace_sha256 = trace.sha256
    review = judgments.answer
    if review is not None:
        if review.trace_sha256 != trace_sha256:
            raise ValueError("answer review belongs to a different trace")
        if trace.answer is None:
            raise ValueError("an answer review requires an observed answer")
        if review.grounded is not None and trace.context is None:
            raise ValueError("a grounding review requires observed context")
    outputs = (trace.route, trace.retrieval, trace.reranking, trace.context, trace.answer)
    statuses: dict[RAGStage, StageStatus] = {}
    for stage, output in zip(_STAGES, outputs, strict=True):
        if output is not None:
            statuses[stage] = "observed"
        elif stage in trace.errors:
            statuses[stage] = "error"
        elif stage in trace.blocked:
            statuses[stage] = "blocked"
        else:
            statuses[stage] = "not_observed"
    retrieved = trace.retrieval.ids if trace.retrieval is not None else None
    reranked = trace.reranking.ids if trace.reranking is not None else None
    context_ids = trace.context.ids if trace.context is not None else None
    rank_input = set(trace.ranking_input() or [])
    # A prefix that contains every known input is also an observed complete ranking.
    ranking_complete = trace.reranking_complete or (
        reranked is not None and len(reranked) == len(rank_input)
    )
    conditional = {
        identifier: grade
        for identifier, grade in judgments.relevance.items()
        if identifier in rank_input
    }
    return RAGAssessment(
        trace_sha256=trace_sha256,
        judgments_sha256=digest(judgments.model_dump(mode="json")),
        stage_status=statuses,
        execution_errors=[stage for stage in _STAGES if stage in trace.errors],
        route_correct=(
            trace.route in judgments.acceptable_routes
            if trace.route is not None and judgments.acceptable_routes is not None
            else None
        ),
        retrieval=assess_ranking(retrieved, judgments.relevance, k=k)
        if retrieved is not None
        else None,
        reranking=assess_ranking(reranked, judgments.relevance, k=k, complete=ranking_complete)
        if reranked is not None
        else None,
        shortlist_reranking=(
            assess_ranking(reranked, conditional, k=k, complete=ranking_complete)
            if reranked is not None
            else None
        ),
        source_coverage={
            "retrieval": _source_coverage(retrieved, judgments.evidence_sets),
            "reranking_top_k": _reranking_coverage(trace, judgments.evidence_sets, k),
            "context": _source_coverage(context_ids, judgments.evidence_sets),
        },
        required_text_retained=_text_retained(trace.context, judgments),
        answer_correct=review.correct if review is not None else None,
        answer_grounded=review.grounded if review is not None else None,
        observed_losses=_losses(trace, judgments, k),
        ordinary_aggregate_eligible=trace.replay is None,
    )
