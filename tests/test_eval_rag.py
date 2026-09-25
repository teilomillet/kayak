"""Independently specified stage losses, unknowns, lineage, and reviewed answers."""

import pytest
from pydantic import ValidationError

from kayak.eval._rag import (
    RAGAnswerReview,
    RAGContext,
    RAGJudgments,
    RAGReplay,
    RAGTrace,
    assess_rag,
)
from kayak.eval._ranking import RankedOutput


def test_missing_retrieval_positive_stays_outside_conditional_shortlist_recall() -> None:
    trace = RAGTrace(
        id="two-facts",
        query="Which two documents describe the policy?",
        retrieval=RankedOutput(ids=["a"]),
        reranking=RankedOutput(ids=["a"]),
    )
    result = assess_rag(trace, RAGJudgments(relevance={"a": 1, "b": 1}), k=1)
    assert result.retrieval is not None
    assert result.reranking is not None
    assert result.shortlist_reranking is not None
    assert result.retrieval.metrics["known_recall_at_k"] == 0.5
    assert result.reranking.metrics["known_recall_at_k"] == 0.5
    assert result.shortlist_reranking.metrics["known_recall_at_k"] == 1.0
    assert result.observed_losses["retrieval_missing_known_positives"] == ["b"]
    assert result.source_coverage["retrieval"] is None


def test_unjudged_first_document_remains_unknown() -> None:
    trace = RAGTrace(
        id="incomplete-qrels",
        query="Find the document.",
        retrieval=RankedOutput(ids=["unjudged", "known"]),
        reranking=RankedOutput(ids=["unjudged", "known"]),
    )
    result = assess_rag(trace, RAGJudgments(relevance={"known": 2}), k=2)
    assert result.reranking is not None
    assert result.reranking.metrics["top1"] is None
    assert result.reranking.metrics["ndcg_at_k"] is None
    assert result.reranking.metrics["judged_fraction_at_k"] == 0.5
    assert result.reranking.metrics["hit_at_k"] == 1.0


@pytest.mark.parametrize("relevance", [{}, {"a": 0}])
def test_no_known_positive_does_not_create_a_recall_score(relevance: dict[str, int]) -> None:
    trace = RAGTrace(id="no-positive", query="Find a source.", retrieval=RankedOutput(ids=[]))
    result = assess_rag(trace, RAGJudgments(relevance=relevance))
    assert result.retrieval is not None
    assert result.retrieval.metrics["known_recall_at_k"] is None
    assert result.stage_status["retrieval"] == "observed"
    assert result.stage_status["reranking"] == "not_observed"
    assert result.required_text_retained is None
    assert result.execution_errors == []


def test_alternative_evidence_routes_are_or_of_and_sets() -> None:
    judgments = RAGJudgments(
        relevance={"a": 1, "b": 1, "c": 1},
        evidence_sets=[["a", "b"], ["c"]],
        evidence_texts={"a": "Step one.", "b": "Step two.", "c": "Both steps."},
    )
    sufficient = RAGTrace(
        id="alternate-route",
        query="Give both steps.",
        retrieval=RankedOutput(ids=["a", "b", "c"]),
        reranking=RankedOutput(ids=["c", "a", "b"]),
        context=RAGContext(ids=["c"], text="[c]\nBoth steps."),
    )
    result = assess_rag(sufficient, judgments, k=1)
    assert result.source_coverage == {
        "retrieval": True,
        "reranking_top_k": True,
        "context": True,
    }
    assert result.required_text_retained is True
    assert result.observed_losses["context_absent_labeled_text"] == ["a", "b"]
    partial = sufficient.model_copy(update={"context": RAGContext(ids=["a"], text="Step one.")})
    result = assess_rag(partial, judgments, k=1)
    assert result.source_coverage["context"] is False
    assert result.required_text_retained is False


def test_source_id_retained_does_not_mean_required_text_retained() -> None:
    trace = RAGTrace(
        id="truncated-context",
        query="When does access end?",
        retrieval=RankedOutput(ids=["policy"]),
        context=RAGContext(ids=["policy"], text="[policy]\nBilling information."),
    )
    judgments = RAGJudgments(
        evidence_sets=[["policy"]],
        evidence_texts={"policy": "Access ends on October 31."},
    )
    result = assess_rag(trace, judgments)
    assert result.source_coverage["context"] is True
    assert result.required_text_retained is False
    assert result.observed_losses["context_absent_labeled_text"] == ["policy"]
    without_text = assess_rag(trace, RAGJudgments(evidence_sets=[["policy"]]))
    assert without_text.required_text_retained is None


def test_incomplete_text_labels_can_leave_one_alternative_unknown() -> None:
    trace = RAGTrace(
        id="partial-text-labels",
        query="Give both facts.",
        context_input_ids=["a", "b", "c"],
        context=RAGContext(ids=["a", "b", "c"], text="Fact A."),
    )
    result = assess_rag(
        trace,
        RAGJudgments(
            evidence_sets=[["a", "b"], ["c"]], evidence_texts={"a": "Fact A.", "c": "Fact C."}
        ),
    )
    assert result.required_text_retained is None
    assert result.source_coverage["context"] is True


def test_literal_retention_and_source_coverage_are_independent() -> None:
    trace = RAGTrace(
        id="text-without-attribution",
        query="What is the date?",
        context_input_ids=[],
        context=RAGContext(ids=[], text="Access ends on October 31."),
    )
    labeled_text = assess_rag(
        trace,
        RAGJudgments(
            evidence_sets=[["policy"]],
            evidence_texts={"policy": "Access ends on October 31."},
        ),
    )
    assert labeled_text.source_coverage["context"] is False
    assert labeled_text.required_text_retained is True
    assert labeled_text.answer_grounded is None
    no_reference_text = assess_rag(trace, RAGJudgments(evidence_sets=[["policy"]]))
    assert no_reference_text.source_coverage["context"] is False
    assert no_reference_text.required_text_retained is None


def test_failure_and_blocked_downstream_are_separate_from_unknown_quality() -> None:
    trace = RAGTrace(
        id="failed-request",
        query="Find the policy.",
        errors={"retrieval": "TimeoutError"},
        blocked={"reranking": "No retrieval result", "context": "No candidates"},
    )
    result = assess_rag(trace, RAGJudgments(relevance={"a": 1}))
    assert result.stage_status == {
        "routing": "not_observed",
        "retrieval": "error",
        "reranking": "blocked",
        "context": "blocked",
        "answer": "not_observed",
    }
    assert result.execution_errors == ["retrieval"]
    assert result.retrieval is None
    assert result.reranking is None
    assert result.answer_correct is None
    assert result.observed_losses == {}


def test_partial_recording_uses_explicit_inputs_without_inventing_retrieval() -> None:
    trace = RAGTrace(
        id="recorded-rerank",
        query="Rank these snippets.",
        reranking_input_ids=["a", "b"],
        reranking=RankedOutput(ids=["b", "a"]),
        context=RAGContext(ids=["b"], text="Selected snippet."),
    )
    result = assess_rag(trace, RAGJudgments(relevance={"a": 0, "b": 1}), k=1)
    assert result.stage_status["retrieval"] == "not_observed"
    assert result.stage_status["reranking"] == "observed"
    assert result.shortlist_reranking is not None
    assert result.shortlist_reranking.metrics["known_recall_at_k"] == 1.0


@pytest.mark.parametrize("returned", [["a"], ["a", "invented"]])
def test_reranking_must_preserve_complete_known_input(returned: list[str]) -> None:
    with pytest.raises(ValidationError, match="permutation"):
        RAGTrace(
            id="bad-ranking",
            query="Rank both documents.",
            retrieval=RankedOutput(ids=["a", "b"]),
            reranking=RankedOutput(ids=returned),
        )


def test_explicit_reranking_subset_records_losses_before_the_ranker() -> None:
    trace = RAGTrace(
        id="filter-before-reranking",
        query="Find both sources.",
        retrieval=RankedOutput(ids=["a", "b", "noise"]),
        reranking_input_ids=["a", "noise"],
        reranking=RankedOutput(ids=["a", "noise"]),
    )
    result = assess_rag(trace, RAGJudgments(relevance={"a": 1, "b": 1, "noise": 0}), k=3)
    assert result.retrieval is not None and result.reranking is not None
    assert result.shortlist_reranking is not None
    assert result.retrieval.metrics["known_recall_at_k"] == 1
    assert result.reranking.metrics["known_recall_at_k"] == 0.5
    assert result.shortlist_reranking.metrics["known_recall_at_k"] == 1
    assert result.observed_losses["reranking_input_omitted_known_positives"] == ["b"]
    assert result.observed_losses["reranking_below_k_known_positives"] == []
    with pytest.raises(ValidationError, match="absent from retrieval"):
        RAGTrace(
            id="invented-ranking-input",
            query="Rank the retrieved sources.",
            retrieval=RankedOutput(ids=["a", "b"]),
            reranking_input_ids=["a", "invented"],
        )


def test_impossible_context_and_unknown_ranking_lineage_are_rejected() -> None:
    with pytest.raises(ValidationError, match="context sources"):
        RAGTrace(
            id="invented-context",
            query="Find a fact.",
            retrieval=RankedOutput(ids=["a"]),
            context=RAGContext(ids=["invented"], text="A fact."),
        )
    with pytest.raises(ValidationError, match="permutation"):
        RAGTrace(id="no-input", query="Find a fact.", reranking=RankedOutput(ids=["a"]))
    with pytest.raises(ValidationError, match="context input"):
        RAGTrace(
            id="invented-context-input",
            query="Find a fact.",
            retrieval=RankedOutput(ids=["a"]),
            context_input_ids=["invented"],
        )


def test_duplicate_ids_and_conflicting_execution_outcomes_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        RAGContext(ids=["a", "a"], text="A fact.")
    with pytest.raises(ValidationError, match="another outcome"):
        RAGTrace(
            id="two-outcomes",
            query="Find a fact.",
            retrieval=RankedOutput(ids=[]),
            errors={"retrieval": "TimeoutError"},
        )
    with pytest.raises(ValidationError, match="another outcome"):
        RAGTrace(
            id="error-and-blocked",
            query="Find a fact.",
            errors={"retrieval": "TimeoutError"},
            blocked={"retrieval": "No route"},
        )


def test_answer_correctness_and_grounding_are_independent() -> None:
    trace = RAGTrace(
        id="remembered-answer",
        query="What is the capital of France?",
        context_input_ids=[],
        context=RAGContext(ids=[], text="No sources were supplied."),
        answer="Paris",
    )
    review = RAGAnswerReview(trace_sha256=trace.sha256, correct=True, grounded=False)
    result = assess_rag(trace, RAGJudgments(answer=review))
    assert result.answer_correct is True
    assert result.answer_grounded is False
    assert result.required_text_retained is None


def test_missing_answer_or_context_cannot_receive_bound_review() -> None:
    trace = RAGTrace(id="not-answered", query="What is the answer?")
    with pytest.raises(ValueError, match="observed answer"):
        assess_rag(
            trace, RAGJudgments(answer=RAGAnswerReview(trace_sha256=trace.sha256, correct=False))
        )
    trace = trace.model_copy(update={"answer": "A response."})
    with pytest.raises(ValueError, match="observed context"):
        assess_rag(
            trace, RAGJudgments(answer=RAGAnswerReview(trace_sha256=trace.sha256, grounded=True))
        )


@pytest.mark.parametrize(
    "field,value", [("answer", "Different answer"), ("query", "Different input")]
)
def test_changed_answer_or_input_rejects_stale_review(field: str, value: str) -> None:
    trace = RAGTrace(id="reviewed", query="Original query", answer="Original answer")
    judgments = RAGJudgments(answer=RAGAnswerReview(trace_sha256=trace.sha256, correct=True))
    changed = RAGTrace.model_validate({**trace.model_dump(), field: value})
    with pytest.raises(ValueError, match="different trace"):
        assess_rag(changed, judgments)


def test_changed_context_rejects_stale_grounding_review() -> None:
    trace = RAGTrace(
        id="context-bound",
        query="What is the date?",
        context_input_ids=["a"],
        context=RAGContext(ids=["a"], text="October 31"),
        answer="October 31",
    )
    judgments = RAGJudgments(answer=RAGAnswerReview(trace_sha256=trace.sha256, grounded=True))
    changed = trace.model_copy(update={"context": RAGContext(ids=["a"], text="November 15")})
    with pytest.raises(ValueError, match="different trace"):
        assess_rag(changed, judgments)


def test_diagnostic_relationship_does_not_run_or_establish_a_cause() -> None:
    original = RAGTrace(id="original", query="Find a source.", retrieval=RankedOutput(ids=[]))
    diagnostic = RAGTrace(
        id="diagnostic",
        query=original.query,
        retrieval=RankedOutput(ids=["a"]),
        replay=RAGReplay(
            original_trace_sha256=original.sha256,
            changed_boundary="retrieval",
            purpose="Caller injected a reviewed source to examine downstream behavior.",
        ),
    )
    judgments = RAGJudgments(relevance={"a": 1})
    before, after = assess_rag(original, judgments), assess_rag(diagnostic, judgments)
    assert before.ordinary_aggregate_eligible is True
    assert after.ordinary_aggregate_eligible is False
    assert after.stage_status["answer"] == "not_observed"
    assert after.answer_correct is None
    assert original.retrieval is not None and original.retrieval.ids == []


def test_assessment_detaches_mutable_inputs_and_revalidates_later_mutation() -> None:
    trace = RAGTrace(id="mutations", query="Find a source.", retrieval=RankedOutput(ids=["a"]))
    judgments = RAGJudgments(relevance={"a": 1})
    result = assess_rag(trace, judgments)
    judgments.relevance["a"] = 0
    assert result.retrieval is not None
    assert result.retrieval.relevance == {"a": 1}
    assert trace.retrieval is not None
    trace.retrieval.ids.append("a")
    with pytest.raises(ValidationError, match="unique"):
        assess_rag(trace, judgments)


def test_source_snapshots_cover_ids_and_change_trace_identity() -> None:
    trace = RAGTrace(
        id="snapshot",
        query="Find a source.",
        documents={"a": "Original source."},
        retrieval=RankedOutput(ids=["a"]),
    )
    changed = trace.model_copy(update={"documents": {"a": "Updated source."}})
    assert trace.sha256 != changed.sha256
    with pytest.raises(ValidationError, match="snapshots must cover"):
        RAGTrace(
            id="missing-snapshot",
            query="Find a source.",
            documents={},
            retrieval=RankedOutput(ids=["a"]),
        )


def test_routing_quality_requires_observation_and_labels() -> None:
    trace = RAGTrace(
        id="routing",
        query="Where is my invoice?",
        route_candidates=["billing", "docs"],
        route="billing",
    )
    assert assess_rag(trace, RAGJudgments()).route_correct is None
    assert assess_rag(trace, RAGJudgments(acceptable_routes=["billing"])).route_correct is True
    assert assess_rag(trace, RAGJudgments(acceptable_routes=["docs"])).route_correct is False
    with pytest.raises(ValidationError, match="routing candidates"):
        RAGTrace(
            id="invented-route", query="Find a source.", route_candidates=["billing"], route="other"
        )


def test_invalid_gold_and_cutoffs_are_rejected() -> None:
    with pytest.raises(ValidationError, match="nonempty"):
        RAGJudgments(evidence_sets=[[]])
    with pytest.raises(ValidationError, match="distinct"):
        RAGJudgments(evidence_sets=[["a", "b"], ["b", "a"]])
    with pytest.raises(ValidationError, match="judged irrelevant"):
        RAGJudgments(relevance={"a": 0}, evidence_sets=[["a"]])
    with pytest.raises(ValidationError, match="belong to an evidence route"):
        RAGJudgments(evidence_sets=[["a"]], evidence_texts={"other": "Other fact."})
    trace = RAGTrace(id="cutoff", query="Find a source.")
    with pytest.raises(ValueError, match="positive integer"):
        assess_rag(trace, RAGJudgments(), k=0)


def test_evaluate_matches_trimmed_text_and_binds_the_literal_reference() -> None:
    trace = RAGTrace(
        id="reference",
        query="How long are audit logs kept?",
        retrieval=RankedOutput(ids=["policy"]),
        context=RAGContext(ids=["policy"], text="Keep logs for 30 days."),
        answer=" 30 days\n",
    )
    original_hash = trace.sha256
    matched = trace.evaluate(expected_answer="30 days", expected_sources=["policy"])
    expected_judgments = RAGJudgments(
        relevance={"policy": 1},
        evidence_sets=[["policy"]],
        answer=RAGAnswerReview(
            trace_sha256=original_hash,
            correct=True,
            provenance={"rubric": "stripped-exact-match-v1", "expected_answer": "30 days"},
        ),
    )
    assert matched == assess_rag(trace, expected_judgments)
    assert matched.trace_sha256 == original_hash == trace.sha256
    assert matched.answer_correct is True and matched.answer_grounded is None
    assert matched.source_coverage["context"] is True
    assert matched.required_text_retained is None
    assert trace.evaluate(expected_answer="thirty days").answer_correct is False
    assert trace.evaluate(expected_answer="30 DAYS").answer_correct is False
    first = trace.evaluate(expected_answer="Wrong reference.")
    second = trace.evaluate(expected_answer="Another wrong reference.")
    assert first.answer_correct is second.answer_correct is False
    assert first.judgments_sha256 != second.judgments_sha256
    spaced = trace.evaluate(expected_answer=" 30 days\n", expected_sources=["policy"])
    assert spaced.answer_correct is True and spaced.judgments_sha256 != matched.judgments_sha256


def test_evaluate_omitted_labels_remain_unknown_for_an_observed_answer() -> None:
    trace = RAGTrace(
        id="unreviewed",
        query="How long are audit logs kept?",
        retrieval=RankedOutput(ids=["policy"]),
        context=RAGContext(ids=["policy"], text="Keep logs for 30 days."),
        answer="30 days",
    )
    assessment = trace.evaluate()
    assert assessment.answer_correct is None and assessment.answer_grounded is None
    assert assessment.required_text_retained is None
    assert assessment.source_coverage["context"] is None
    assert trace.evaluate(expected_sources=[]).source_coverage["context"] is None


@pytest.mark.parametrize("failed", [False, True])
def test_evaluate_missing_answer_stays_unknown_even_with_a_reference(failed: bool) -> None:
    trace = RAGTrace(
        id="unanswered",
        query="How long are audit logs kept?",
        retrieval=RankedOutput(ids=[]),
        context=RAGContext(ids=[], text=""),
        errors={"answer": "TimeoutError"} if failed else {},
    )
    assessment = trace.evaluate(expected_answer="30 days", expected_sources=["policy"])
    assert assessment.answer_correct is None and assessment.answer_grounded is None
    assert assessment.source_coverage["context"] is False
    assert assessment.stage_status["context"] == "observed"
    assert assessment.stage_status["answer"] == ("error" if failed else "not_observed")
    assert assessment.execution_errors == (["answer"] if failed else [])


def test_evaluate_sources_are_jointly_required_and_other_sources_stay_unjudged() -> None:
    trace = RAGTrace(
        id="partial-evidence",
        query="How long are audit logs kept?",
        retrieval=RankedOutput(ids=["policy", "other"]),
        context=RAGContext(ids=["policy", "other"], text="30 days\nOther text"),
        answer="30 days",
    )
    assessment = trace.evaluate(expected_sources=["policy", "missing"])
    assert assessment.source_coverage["retrieval"] is False
    assert assessment.source_coverage["context"] is False
    assert assessment.retrieval is not None
    assert assessment.retrieval.relevance == {"policy": 1, "missing": 1}
    assert assessment.retrieval.unjudged_ids == ["other"]
    assert assessment.answer_correct is None


def test_evaluate_ranking_cutoff_does_not_change_observed_context() -> None:
    trace = RAGTrace(
        id="reranked",
        query="How long are audit logs kept?",
        retrieval=RankedOutput(ids=["old", "new"]),
        reranking=RankedOutput(ids=["new", "old"]),
        context=RAGContext(ids=["new"], text="New policy."),
        answer="New policy.",
    )
    first = trace.evaluate(expected_sources=["old"], k=1)
    both = trace.evaluate(expected_sources=["old"], k=2)
    assert first.source_coverage["reranking_top_k"] is False
    assert both.source_coverage["reranking_top_k"] is True
    assert first.source_coverage["retrieval"] is both.source_coverage["retrieval"] is True
    assert first.source_coverage["context"] is both.source_coverage["context"] is False
