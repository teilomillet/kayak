"""Independent ranking arithmetic, incomplete judgments, and adapter boundaries."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kayak.eval._ranking import RankedOutput, assess_ranking, ranking_metrics


def test_binary_arithmetic_preserves_positions_and_missing_positives() -> None:
    result = assess_ranking(["b", "a", "c"], {"a": 1, "b": 0, "c": 1, "d": 1}, k=2)
    assert result.metrics == {
        "top1": 0.0,
        "hit_at_k": 1.0,
        "known_recall_at_k": 1 / 3,
        "reciprocal_rank": 1 / 2,
        "ndcg_at_k": pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3))),
        "judged_fraction_at_k": 1.0,
    }
    assert result.unjudged_ids == [] and result.unavailable_reasons == {}
    assert ranking_metrics(["a", "c", "d"], ["b", "a", "c"], 2) == {
        "top1": 0.0,
        "hit_at_k": 1.0,
        "recall_at_k": 1 / 3,
        "mrr": 1 / 2,
    }


def test_unjudged_is_not_an_irrelevant_label_or_a_removed_rank() -> None:
    unknown = assess_ranking(["unjudged", "a"], {"a": 1}, k=2)
    assert unknown.unjudged_ids == ["unjudged"]
    assert unknown.metrics["top1"] is None
    assert unknown.metrics["reciprocal_rank"] is None
    assert unknown.metrics["ndcg_at_k"] is None
    assert unknown.metrics["hit_at_k"] == 1
    assert unknown.metrics["known_recall_at_k"] == 1
    assert unknown.metrics["judged_fraction_at_k"] == 0.5
    negative = assess_ranking(["unjudged", "a"], {"a": 1, "unjudged": 0}, k=2)
    assert negative.metrics["top1"] == 0
    assert negative.metrics["reciprocal_rank"] == 0.5
    assert set(unknown.unavailable_reasons) == {"top1", "reciprocal_rank", "ndcg_at_k"}


def test_unknown_after_the_first_positive_does_not_hide_observed_success() -> None:
    result = assess_ranking(["a", "unknown"], {"a": 1}, k=1)
    assert result.metrics["top1"] == result.metrics["reciprocal_rank"] == 1
    assert result.metrics["ndcg_at_k"] == 1
    assert result.unjudged_ids == ["unknown"]


def test_unknown_prefix_cannot_establish_no_hit() -> None:
    result = assess_ranking(["unknown", "a"], {"a": 1}, k=1)
    assert result.metrics["hit_at_k"] is None
    assert result.metrics["known_recall_at_k"] == 0
    assert result.unavailable_reasons["hit_at_k"]


def test_empty_results_are_different_from_missing_judgments() -> None:
    empty = assess_ranking([], {"a": 1}, k=10)
    assert empty.metrics["top1"] == empty.metrics["known_recall_at_k"] == 0
    assert empty.metrics["reciprocal_rank"] == empty.metrics["ndcg_at_k"] == 0
    unknown = assess_ranking(["a"], {}, k=10)
    assert unknown.metrics["known_recall_at_k"] is None
    assert unknown.metrics["top1"] is None
    no_positives = assess_ranking(["a"], {"a": 0}, k=10)
    assert no_positives.metrics["top1"] == 0
    assert no_positives.metrics["known_recall_at_k"] is None
    assert no_positives.metrics["ndcg_at_k"] is None
    assert ranking_metrics(["a"], [], 10) == {
        "top1": 0.0,
        "hit_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
    }


def test_ndcg_uses_linear_grades_and_an_explicit_ideal() -> None:
    result = assess_ranking(["b", "a"], {"a": 3, "b": 1}, k=2)
    expected = (1 + 3 / math.log2(3)) / (3 + 1 / math.log2(3))
    assert result.metrics["ndcg_at_k"] == pytest.approx(expected)
    large = assess_ranking(["b", "a"], {"a": 3 * 10**400, "b": 10**400}, k=2)
    assert large.metrics["ndcg_at_k"] == pytest.approx(expected)


@pytest.mark.parametrize("ids", [["a", "a"], [" "], [""]])
def test_invalid_document_ids_rejected(ids: list[str]) -> None:
    with pytest.raises(ValueError):
        assess_ranking(ids, {"a": 1}, k=1)
    with pytest.raises(ValueError):
        RankedOutput(ids=ids)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_cutoff_rejected(k: object) -> None:
    with pytest.raises(ValueError):
        assess_ranking(["a"], {"a": 1}, k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize("grade", [-1, True, 0.5, float("nan")])
def test_invalid_grade_rejected(grade: object) -> None:
    with pytest.raises(ValueError):
        assess_ranking(["a"], {"a": grade}, k=1)  # type: ignore[dict-item]


def test_output_keeps_native_scores_and_authoritative_order() -> None:
    output = RankedOutput(ids=["a", "b"], scores={"a": -12.0, "b": 40.0})
    assert output.ids == ["a", "b"]  # Distances and scores need not sort the same way.
    assert output.scores == {"a": -12.0, "b": 40.0}
    assert RankedOutput(ids=["a"]).scores is None
    assert RankedOutput(ids=[]).ids == []
    assert "probabilities" not in output.model_dump()


@pytest.mark.parametrize("scores", [{"b": 1.0}, {"a": float("inf")}, {"a": float("nan")}])
def test_invalid_native_scores_rejected(scores: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RankedOutput(ids=["a"], scores=scores)


def test_assessment_owns_its_input_snapshot() -> None:
    ids = ["a", "b"]
    relevance = {"a": 1, "b": 0}
    result = assess_ranking(ids, relevance, k=1)
    ids.reverse()
    relevance["a"] = 0
    assert result.ranked_ids == ["a", "b"] and result.relevance["a"] == 1
    assert result.metrics["top1"] == 1


@given(order=st.permutations(["a", "b", "c", "d"]), k=st.integers(min_value=1, max_value=6))
def test_binary_metrics_follow_independent_prefix_counts(order: list[str], k: int) -> None:
    relevant = {"b", "d"}
    metrics = ranking_metrics(sorted(relevant), order, k)
    prefix = order[:k]
    first = min(order.index(identifier) + 1 for identifier in relevant)
    assert metrics["top1"] == int(order[0] in relevant)
    assert metrics["recall_at_k"] == len(set(prefix) & relevant) / 2
    assert metrics["hit_at_k"] == int(bool(set(prefix) & relevant))
    assert metrics["mrr"] == 1 / first
