"""Top-K API outputs retain observed ranks without inventing an unseen tail."""

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from kayak.eval import (
    RAGAnswerReview,
    RAGContext,
    RAGJudgments,
    RAGTrace,
    RankedOutput,
    assess_rag,
    assess_ranking,
)
from kayak.eval._ranking import RankingAssessment


def test_cutoff_inside_the_recorded_prefix_has_known_top_k_metrics() -> None:
    result = assess_ranking(["noise", "a"], {"noise": 0, "a": 1, "b": 1}, k=2, complete=False)
    assert result.complete is False
    assert result.metrics == {
        "top1": 0.0,
        "hit_at_k": 1.0,
        "known_recall_at_k": 0.5,
        "reciprocal_rank": 0.5,
        "ndcg_at_k": pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3))),
        "judged_fraction_at_k": 1.0,
    }
    assert result.unavailable_reasons == {}


def test_cutoff_beyond_prefix_retains_only_established_metrics() -> None:
    result = assess_ranking(["noise", "a"], {"noise": 0, "a": 1, "b": 1}, k=3, complete=False)
    assert result.metrics == {
        "top1": 0.0,
        "hit_at_k": 1.0,
        "known_recall_at_k": None,
        "reciprocal_rank": 0.5,
        "ndcg_at_k": None,
        "judged_fraction_at_k": None,
    }
    assert set(result.unavailable_reasons) == {
        "known_recall_at_k",
        "ndcg_at_k",
        "judged_fraction_at_k",
    }


def test_recovering_every_known_positive_establishes_known_recall() -> None:
    result = assess_ranking(["a"], {"a": 1}, k=5, complete=False)
    assert result.metrics["known_recall_at_k"] == 1.0
    assert result.metrics["hit_at_k"] == 1.0
    assert result.metrics["reciprocal_rank"] == 1.0
    assert result.metrics["ndcg_at_k"] is None


def test_empty_prefix_is_not_an_observed_empty_complete_ranking() -> None:
    prefix = assess_ranking([], {"a": 1}, k=1, complete=False)
    assert all(value is None for value in prefix.metrics.values())
    assert set(prefix.unavailable_reasons) == set(prefix.metrics)
    complete = assess_ranking([], {"a": 1}, k=1)
    assert complete.metrics["hit_at_k"] == 0.0
    assert complete.metrics["known_recall_at_k"] == 0.0
    assert complete.metrics["reciprocal_rank"] == 0.0


def test_no_hit_in_prefix_does_not_establish_reciprocal_rank_zero() -> None:
    prefix = assess_ranking(["noise"], {"noise": 0, "a": 1}, k=1, complete=False)
    assert prefix.metrics["hit_at_k"] == 0.0
    assert prefix.metrics["known_recall_at_k"] == 0.0
    assert prefix.metrics["reciprocal_rank"] is None
    beyond = assess_ranking(["noise"], {"noise": 0, "a": 1}, k=2, complete=False)
    assert beyond.metrics["hit_at_k"] is None
    assert beyond.metrics["known_recall_at_k"] is None


def test_missing_positive_labels_do_not_create_an_evaluation_denominator() -> None:
    negative = assess_ranking(["noise"], {"noise": 0}, k=2, complete=False)
    assert negative.metrics["top1"] == 0.0
    assert negative.metrics["hit_at_k"] is None
    assert negative.metrics["known_recall_at_k"] is None
    assert negative.metrics["reciprocal_rank"] is None
    unjudged = assess_ranking(["a"], {}, k=1, complete=False)
    assert unjudged.metrics["top1"] is None
    assert unjudged.metrics["judged_fraction_at_k"] == 0.0


@pytest.mark.parametrize("complete", [0, 1, "false", None])
def test_completeness_requires_an_explicit_boolean(complete: object) -> None:
    with pytest.raises(ValueError, match="boolean"):
        assess_ranking(["a"], {"a": 1}, k=1, complete=complete)  # type: ignore[arg-type]


def test_old_ranking_assessments_default_to_complete() -> None:
    result = assess_ranking(["a"], {"a": 1}, k=1)
    legacy = result.model_dump(exclude={"complete"})
    assert RankingAssessment.model_validate(legacy).complete is True


def prefix_trace(returned: list[str]) -> RAGTrace:
    return RAGTrace(
        id="top-k-api",
        query="Find both facts.",
        retrieval=RankedOutput(ids=["a", "b", "noise"]),
        reranking=RankedOutput(ids=returned),
        reranking_complete=False,
    )


def test_rag_records_unreturned_positives_without_assigning_their_rank() -> None:
    result = assess_rag(
        prefix_trace(["a"]),
        RAGJudgments(relevance={"a": 1, "b": 1, "noise": 0}, evidence_sets=[["a", "b"]]),
        k=2,
    )
    assert result.reranking is not None and result.shortlist_reranking is not None
    assert result.reranking.complete is False
    assert result.shortlist_reranking.metrics["known_recall_at_k"] is None
    assert result.source_coverage["reranking_top_k"] is None
    assert result.observed_losses["reranking_below_k_known_positives"] == []
    assert result.observed_losses["reranking_unreturned_known_positives"] == ["b"]


def test_observed_below_cutoff_and_unreturned_positives_are_separate() -> None:
    result = assess_rag(prefix_trace(["noise", "a"]), RAGJudgments(relevance={"a": 1, "b": 1}), k=1)
    assert result.observed_losses["reranking_below_k_known_positives"] == ["a"]
    assert result.observed_losses["reranking_unreturned_known_positives"] == ["b"]


@pytest.mark.parametrize(
    ("routes", "k", "coverage"),
    [
        ([["a"]], 5, True),
        ([["a", "b"]], 1, False),
        ([["a", "b"]], 2, None),
        ([["outside"]], 2, False),
        ([["a", "b", "noise"]], 2, False),
        ([["outside"], ["a", "b"]], 2, None),
        ([], 2, None),
    ],
)
def test_prefix_evidence_coverage_respects_known_input_and_available_positions(
    routes: list[list[str]], k: int, coverage: bool | None
) -> None:
    result = assess_rag(prefix_trace(["a"]), RAGJudgments(evidence_sets=routes), k=k)
    assert result.source_coverage["reranking_top_k"] is coverage


def test_prefix_may_not_invent_input_ids_or_context_sources() -> None:
    with pytest.raises(ValidationError, match="absent from its known input"):
        prefix_trace(["invented"])
    with pytest.raises(ValidationError, match="observed prefix"):
        RAGTrace(id="missing", query="Find a fact.", reranking_complete=False)
    trace = prefix_trace(["a"])
    with pytest.raises(ValidationError, match="context sources"):
        assess_rag(
            trace.model_copy(update={"context": RAGContext(ids=["b"], text="Fact B.")}),
            RAGJudgments(),
        )


def test_top_k_api_returning_every_known_input_has_a_complete_observation() -> None:
    result = assess_rag(
        prefix_trace(["noise", "a", "b"]),
        RAGJudgments(relevance={"a": 1, "b": 1, "noise": 0}, evidence_sets=[["a", "b"]]),
        k=5,
    )
    assert result.reranking is not None and result.reranking.complete is True
    assert result.reranking.metrics["known_recall_at_k"] == 1.0
    assert result.reranking.metrics["ndcg_at_k"] is not None
    assert result.source_coverage["reranking_top_k"] is True


def legacy_trace() -> RAGTrace:
    return RAGTrace(
        id="legacy",
        query="Where is the policy?",
        documents={"a": "A policy.", "b": "Other text."},
        retrieval=RankedOutput(ids=["b", "a"]),
        reranking=RankedOutput(ids=["a", "b"]),
        context=RAGContext(ids=["a"], text="A policy."),
        answer="Here.",
        provenance={"index": "v1"},
    )


def test_default_completeness_preserves_existing_trace_digest_and_review() -> None:
    # Recorded from the legacy schema before adding reranking_complete.
    old_digest = "7dfa0473d44d3519e791869daad1bfac2083bd4ccd32719018398fe202750a6f"
    trace = legacy_trace()
    assert trace.sha256 == old_digest
    old_payload = trace.model_dump(exclude={"reranking_complete"})
    assert RAGTrace.model_validate(old_payload).sha256 == old_digest
    reviewed = RAGJudgments(answer=RAGAnswerReview(trace_sha256=old_digest, correct=True))
    assert assess_rag(trace, reviewed).answer_correct is True


def test_changed_completeness_invalidates_old_review_even_when_ids_match() -> None:
    trace = legacy_trace()
    changed = trace.model_copy(update={"reranking_complete": False})
    assert changed.sha256 != trace.sha256
    reviewed = RAGJudgments(answer=RAGAnswerReview(trace_sha256=trace.sha256, correct=True))
    with pytest.raises(ValueError, match="different trace"):
        assess_rag(changed, reviewed)
    assert RAGTrace.model_validate_json(changed.model_dump_json()).sha256 == changed.sha256


@given(
    order=st.permutations(["a", "b", "noise"]),
    length=st.integers(min_value=0, max_value=3),
    k=st.integers(min_value=1, max_value=5),
)
def test_every_known_prefix_metric_survives_all_possible_tail_orders(
    order: list[str], length: int, k: int
) -> None:
    relevance = {"a": 2, "b": 1, "noise": 0}
    prefix = order[:length]
    partial = assess_ranking(prefix, relevance, k=k, complete=False)
    for tail in itertools.permutations(order[length:]):
        completed = assess_ranking(prefix + list(tail), relevance, k=k)
        for name, value in partial.metrics.items():
            if value is not None:
                assert completed.metrics[name] == pytest.approx(value)
