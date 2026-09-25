"""Exercise public case callbacks at ownership, failure, and result boundaries."""

import pytest

from kayak import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResult,
    ModelInfo,
    RankedCandidate,
    RankingRequest,
    RankingResult,
)
from kayak.eval import (
    ChoiceCase,
    ChoiceDataset,
    RankedOutput,
    RankingCase,
    RankingDataset,
    evaluate_cases,
    parse_cases,
    summarize_cases,
)


def dataset() -> RankingDataset:
    return RankingDataset(
        name="adapter-evaluation",
        kind="ranking",
        k=1,
        cases=[
            RankingCase(
                id=identifier,
                request=RankingRequest(
                    state="Invoice query",
                    instructions="Rank passages by usefulness to the query.",
                    candidates={"account": "Account settings", "invoice": "Download invoices"},
                ),
                relevant=["invoice"],
            )
            for identifier in ("first", "second", "third")
        ],
    )


def test_rank_only_callback_and_independent_label_changes() -> None:
    calls: list[str] = []

    def rank(request: RankingRequest) -> RankedOutput:
        calls.append(request.model_dump_json())
        return RankedOutput(ids=["invoice", "account"], scores={"invoice": 42.0, "account": -3.0})

    cases = dataset()
    first = evaluate_cases(cases, rank=rank)
    for case in cases.cases:
        case.relevant[:] = ["account"]
    second = evaluate_cases(cases, rank=rank)
    assert calls[:3] == calls[3:]
    assert [report.result for report in first] == [report.result for report in second]
    assert first[0].metrics["top1"] == 1 and second[0].metrics["top1"] == 0
    assert isinstance(first[0].result, RankedOutput)
    assert first[0].result.provenance == {}  # Identity is unknown, never fabricated.
    assert "probabilities" not in first[0].result.model_dump()


def test_callback_mutation_cannot_change_retained_inputs_or_later_requests() -> None:
    cases = dataset()
    before = cases.model_dump_json()
    seen: list[list[str]] = []
    output = RankedOutput(ids=["invoice", "account"], provenance={"index": "v1"})

    def rank(request: RankingRequest) -> RankedOutput:
        seen.append(list(request.candidates))
        request.candidates.clear()
        return output

    reports = evaluate_cases(cases, rank=rank)
    output.ids.reverse()
    output.provenance["index"] = "changed-after-return"
    assert cases.model_dump_json() == before
    assert seen == [["account", "invoice"]] * 3
    assert all(report.error is None for report in reports)
    assert isinstance(reports[0].result, RankedOutput)
    assert reports[0].result.ids == ["invoice", "account"]
    assert reports[0].result.provenance["index"] == "v1"


def test_custom_exception_is_redacted_counted_once_and_later_cases_continue() -> None:
    calls = 0

    def rank(request: RankingRequest) -> RankedOutput:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private backend data and credentials")
        return RankedOutput(ids=["invoice", "account"])

    cases = dataset()
    reports = evaluate_cases(cases, rank=rank)
    summary = summarize_cases(cases, reports)
    assert calls == 3 and summary["failed"] == 1 and summary["top1"] == 2 / 3
    assert reports[1].error == {"type": "RuntimeError", "request_id": None}
    assert "private" not in reports[1].model_dump_json()
    assert reports[1].input_order_metrics is not None
    assert reports[1].input_order_metrics["mrr"] == 0.5


@pytest.mark.parametrize("ids", [["invoice"], ["invoice", "invented"], []])
def test_case_ranker_must_return_a_complete_permutation(ids: list[str]) -> None:
    reports = evaluate_cases(dataset(), rank=lambda request: RankedOutput(ids=ids))
    assert all(report.error is not None for report in reports)
    assert all(report.metrics["top1"] == 0 for report in reports)


def test_provenance_change_retains_evidence_and_zeroes_quality() -> None:
    calls = 0

    def rank(request: RankingRequest) -> RankedOutput:
        nonlocal calls
        calls += 1
        return RankedOutput(
            ids=["invoice", "account"], provenance={"index": "v2" if calls == 2 else "v1"}
        )

    reports = evaluate_cases(dataset(), rank=rank)
    assert reports[1].error == {"type": "ProvenanceChanged"}
    assert reports[1].result is not None and reports[1].metrics["top1"] == 0
    assert reports[2].error is None and reports[2].metrics["top1"] == 1


def test_missing_callback_and_invalid_mutated_data_fail_before_effects() -> None:
    with pytest.raises(ValueError, match="rank callback"):
        evaluate_cases(dataset())
    cases = dataset()
    cases.cases[1].relevant[:] = ["unknown"]
    calls: list[str] = []

    def rank(request: RankingRequest) -> RankedOutput:
        calls.append(request.state)
        return RankedOutput(ids=list(request.candidates))

    with pytest.raises(ValueError):
        evaluate_cases(cases, rank=rank)
    assert calls == []


def test_changing_native_and_external_result_types_counts_as_failure() -> None:
    native = RankingResult(
        model=ModelInfo(
            id="fixture",
            revision="1",
            fingerprint="fake",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        ranked=[
            RankedCandidate(id="account", score=0.0, probability=0.5),
            RankedCandidate(id="invoice", score=0.0, probability=0.5),
        ],
        input_tokens=0,
    )
    outputs: list[RankingResult | RankedOutput] = [
        native,
        RankedOutput(ids=["invoice", "account"]),
        native,
    ]
    pending = iter(outputs)
    reports = evaluate_cases(dataset(), rank=lambda request: next(pending))
    assert reports[0].error is None and reports[2].error is None
    assert reports[1].error == {"type": "ResultKindChanged"}
    assert reports[1].result is not None and reports[1].metrics["top1"] == 0


def test_sync_boundaries_and_original_error_wins_cleanup_error() -> None:
    events: list[str] = []

    def rank(request: RankingRequest) -> RankedOutput:
        events.append("rank")
        raise RuntimeError("original private error")

    def sync() -> None:
        events.append("sync")
        if len(events) % 3 == 0:
            raise ValueError("secondary cleanup error")

    reports = evaluate_cases(dataset(), rank=rank, sync=sync)
    assert events == ["sync", "rank", "sync"] * 3
    assert all(report.error == {"type": "RuntimeError", "request_id": None} for report in reports)
    assert all(report.duration_seconds >= 0 for report in reports)


def test_interrupt_propagates_without_retry_or_closing_the_callback_owner() -> None:
    class Ranker:
        calls = 0
        closed = False

        def rank(self, request: RankingRequest) -> RankedOutput:
            self.calls += 1
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    backend = Ranker()
    with pytest.raises(KeyboardInterrupt):
        evaluate_cases(dataset(), rank=backend.rank)
    assert backend.calls == 1 and not backend.closed


def test_summary_rejects_missing_or_reordered_cases() -> None:
    cases = dataset()
    reports = evaluate_cases(cases, rank=lambda request: RankedOutput(ids=list(request.candidates)))
    for incomplete in (reports[:-1], list(reversed(reports)), []):
        with pytest.raises(ValueError, match="one report per dataset case"):
            summarize_cases(cases, incomplete)
    assert parse_cases(cases.model_dump_json()) == cases


def test_decide_only_callback_checks_full_contract_and_partial_credit() -> None:
    question = Choice(
        instructions="Is this topic present?", criteria={"yes": "Present", "no": "Absent"}
    )
    cases = ChoiceDataset(
        name="tags",
        kind="choice",
        cases=[
            ChoiceCase(
                id="both",
                request=DecisionRequest(
                    state="two topics", questions={"a": question, "b": question}
                ),
                expected={"a": ["yes"], "b": ["no"]},
            )
        ],
    )

    def decide(request: DecisionRequest) -> DecisionResult:
        return DecisionResult(
            model=ModelInfo(
                id="fixture",
                revision="1",
                fingerprint="fake",
                encoder="none",
                encoder_revision="none",
                device="none",
                dtype="none",
            ),
            input_tokens=0,
            answers={
                question_id: ChoiceAnswer(
                    choice="yes",
                    scores={"yes": 0.0, "no": 0.0},
                    probabilities={"yes": 0.5, "no": 0.5},
                )
                for question_id in request.questions
            },
        )

    result = evaluate_cases(cases, decide=decide)[0]
    assert result.metrics == {"exact_match": 0.0, "question_accuracy": 0.5}
    assert result.error is None
