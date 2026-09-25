"""Challenge summary bindings and metrics independently of callback execution."""

import pytest

from kayak import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    InferenceError,
    ModelInfo,
    RankingRequest,
)
from kayak.eval import (
    ChoiceCase,
    ChoiceDataset,
    RankedOutput,
    RankingCase,
    RankingDataset,
    evaluate_cases,
    summarize_cases,
)


def ranking_dataset() -> RankingDataset:
    return RankingDataset(
        name="summary-fixture",
        kind="ranking",
        k=1,
        cases=[
            RankingCase(
                id=identifier,
                request=RankingRequest(
                    state="Where is my invoice?",
                    instructions="Rank these passages.",
                    candidates={"account": "Account settings", "invoice": "Download invoices"},
                ),
                relevant=["invoice"],
            )
            for identifier in ("first", "second")
        ],
    )


def ranked(request: RankingRequest) -> RankedOutput:
    return RankedOutput(ids=["invoice", "account"])


@pytest.mark.parametrize("change", ["state", "instructions", "description", "order", "gold", "k"])
def test_summary_rejects_changed_case_with_the_same_ids(change: str) -> None:
    dataset = ranking_dataset()
    reports = evaluate_cases(dataset, rank=ranked)
    case = dataset.cases[0]
    if change in {"state", "instructions"}:
        request = case.request.model_copy(update={change: "Changed text"})
        dataset.cases[0] = case.model_copy(update={"request": request})
    elif change == "description":
        case.request.candidates["invoice"] = "Changed passage"
    elif change == "order":
        ordered = dict(reversed(list(case.request.candidates.items())))
        case.request.candidates.clear()
        case.request.candidates.update(ordered)
    elif change == "gold":
        case.relevant[:] = ["account"]
    else:
        dataset = dataset.model_copy(update={"k": 2})
    with pytest.raises(ValueError, match="judgments|case hash"):
        summarize_cases(dataset, reports)


@pytest.mark.parametrize("metric", ["top1", "hit_at_k", "recall_at_k", "mrr"])
@pytest.mark.parametrize("value", [0.0, 2.0])
def test_summary_rejects_changed_metrics(metric: str, value: float) -> None:
    dataset = ranking_dataset()
    reports = evaluate_cases(dataset, rank=ranked)
    reports[0].metrics[metric] = value
    with pytest.raises(ValueError, match="metrics differ"):
        summarize_cases(dataset, reports)


def test_summary_rejects_changed_baseline_and_invented_result_ids() -> None:
    dataset = ranking_dataset()
    reports = evaluate_cases(dataset, rank=ranked)
    baseline = reports[0].input_order_metrics
    assert baseline is not None
    baseline["top1"] = 1.0
    with pytest.raises(ValueError, match="metrics differ"):
        summarize_cases(dataset, reports)

    reports = evaluate_cases(dataset, rank=ranked)
    result = reports[0].result
    assert isinstance(result, RankedOutput)
    result.ids[-1] = "invented"
    with pytest.raises(InferenceError, match="every supplied candidate"):
        summarize_cases(dataset, reports)


def test_summary_recomputes_failed_scores_and_preserves_the_full_denominator() -> None:
    dataset = ranking_dataset()
    calls = 0

    def fail_once(request: RankingRequest) -> RankedOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("callback failed")
        return ranked(request)

    reports = evaluate_cases(dataset, rank=fail_once)
    assert summarize_cases(dataset, reports) == {
        "cases": 2,
        "failed": 1,
        "top1": 0.5,
        "hit_at_k": 0.5,
        "recall_at_k": 0.5,
        "mrr": 0.5,
        "input_order_top1": 0.0,
        "input_order_hit_at_k": 0.0,
        "input_order_recall_at_k": 0.0,
        "input_order_mrr": 0.5,
    }
    assert len(reports[0].case_sha256) == 64
    reports[1].metrics["top1"] = 1.0
    with pytest.raises(ValueError, match="metrics differ"):
        summarize_cases(dataset, reports)


def test_choice_gold_and_native_model_change_remain_bound_to_reports() -> None:
    question = Choice(instructions="Choose a topic.", criteria={"a": "Topic A", "b": "Topic B"})
    dataset = ChoiceDataset(
        name="choice-summary",
        kind="choice",
        cases=[
            ChoiceCase(
                id=identifier,
                request=DecisionRequest(state="A message", questions={"topic": question}),
                expected={"topic": ["a"]},
            )
            for identifier in ("first", "second")
        ],
    )
    calls = 0

    def decide(request: DecisionRequest) -> DecisionResult:
        nonlocal calls
        calls += 1
        return DecisionResult(
            model=ModelInfo(
                id="fixture",
                revision=str(calls),
                fingerprint="mock",
                encoder="none",
                encoder_revision="none",
                device="none",
                dtype="none",
            ),
            input_tokens=0,
            answers={
                "topic": ChoiceAnswer(
                    choice="a", scores={"a": 0.0, "b": 0.0}, probabilities={"a": 0.5, "b": 0.5}
                )
            },
        )

    reports = evaluate_cases(dataset, decide=decide)
    assert summarize_cases(dataset, reports) == {
        "cases": 2,
        "failed": 1,
        "exact_match": 0.5,
        "question_accuracy": 0.5,
    }
    dataset.cases[0].expected["topic"] = ["b"]
    with pytest.raises(ValueError, match="judgments differ"):
        summarize_cases(dataset, reports)
    dataset.cases[0].expected["topic"] = ["a"]
    reports[1] = reports[1].model_copy(update={"error": None, "metrics": reports[0].metrics})
    with pytest.raises(ValueError, match="identity change"):
        summarize_cases(dataset, reports)
