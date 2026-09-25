"""Exercise the example evaluator independently of a model's semantic accuracy."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from examples import evaluate_use_cases as evaluation
from kayak import (
    Choice,
    ChoiceAnswer,
    Client,
    DecisionRequest,
    DecisionResult,
    ModelInfo,
    RankingResult,
)

ROOT = Path(__file__).resolve().parents[1]
DATASETS = sorted((ROOT / "examples/evaluations").glob("*.json"))
MODEL = ModelInfo(
    id="test/model",
    revision="v1",
    fingerprint="test-fingerprint",
    encoder="test/encoder",
    encoder_revision="v1",
    device="cpu",
    dtype="float32",
)


def response(request: httpx.Request, winner: int = 0, model: ModelInfo = MODEL) -> httpx.Response:
    decision = DecisionRequest.model_validate_json(request.content)
    answers: dict[str, ChoiceAnswer] = {}
    for question_id, question in decision.questions.items():
        identifiers = list(question.criteria)
        scores = {
            identifier: float(index == winner) for index, identifier in enumerate(identifiers)
        }
        total = sum(math.exp(score) for score in scores.values())
        answers[question_id] = ChoiceAnswer(
            choice=identifiers[winner],
            scores=scores,
            probabilities={
                identifier: math.exp(score) / total for identifier, score in scores.items()
            },
        )
    result = DecisionResult(model=model, answers=answers, input_tokens=7)
    return httpx.Response(200, content=result.model_dump_json())


def choice_case(identifier: str, labels: dict[str, list[str]]) -> evaluation.ChoiceCase:
    return evaluation.ChoiceCase(
        id=identifier,
        request=DecisionRequest(
            state="Private input with a newline\nEt une facture en français.",
            questions={
                question_id: Choice(
                    instructions="Which category applies?",
                    criteria={"a": "First category", "b": "Second category"},
                )
                for question_id in labels
            },
        ),
        expected=labels,
    )


def choice_dataset() -> evaluation.ChoiceDataset:
    return evaluation.ChoiceDataset(
        name="test-choices",
        kind="choice",
        examples=["examples/local_decisions.py"],
        cases=[
            choice_case("one", {"topic": ["a"]}),
            choice_case("two", {"topic": ["b"], "urgent": ["a", "b"]}),
            choice_case("three", {"topic": ["a"]}),
        ],
    )


def test_choice_partial_credit_alternatives_failures_and_request_isolation() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 3:
            return httpx.Response(
                503,
                headers={"x-request-id": "busy-case"},
                json={"error": {"code": "overloaded", "message": "private-server-detail"}},
            )
        return response(request)

    dataset = choice_dataset()
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        reports = evaluation.evaluate(client, dataset)
    assert evaluation.summarize(dataset, reports) == {
        "cases": 3,
        "failed": 1,
        "exact_match": 1 / 3,
        "question_accuracy": 2 / 4,
    }
    assert reports[1].metrics == {"exact_match": 0.0, "question_accuracy": 0.5}
    assert reports[2].result is None
    assert reports[2].error == {
        "type": "RemoteError",
        "request_id": "busy-case",
        "status_code": 503,
        "code": "overloaded",
    }
    assert all(report.duration_seconds >= 0 for report in reports)
    assert len(calls) == 3
    for call, case, report in zip(calls, dataset.cases, reports, strict=True):
        assert json.loads(call.content) == case.request.model_dump()
        assert case.request.state not in report.model_dump_json()
        assert "private-server-detail" not in report.model_dump_json()


@pytest.mark.parametrize("winner", [0, 1])
def test_scores_change_when_returned_choices_change(winner: int) -> None:
    dataset = choice_dataset()
    with Client(
        base_url="http://test",
        transport=httpx.MockTransport(lambda request: response(request, winner)),
    ) as client:
        reports = evaluation.evaluate(client, dataset)
    assert reports[0].metrics["exact_match"] == float(winner == 0)
    assert reports[1].metrics["exact_match"] == float(winner == 1)


def ranking_dataset() -> evaluation.RankingDataset:
    return evaluation.RankingDataset.model_validate(
        {
            "name": "test-ranking",
            "kind": "ranking",
            "examples": ["examples/rerank_documents.py"],
            "k": 2,
            "cases": [
                {
                    "id": identifier,
                    "request": {
                        "state": "A fixed query",
                        "instructions": "Rank relevant passages.",
                        "candidates": {"a": "Passage A", "b": "Passage B", "c": "Passage C"},
                    },
                    "relevant": ["b", "c"],
                }
                for identifier in ("one", "two")
            ],
        }
    )


def test_ranking_arithmetic_and_failures_use_the_supplied_shortlist() -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ConnectError("private-connection-detail", request=request)
        return evaluation.simulated_response(request)  # Ties preserve a, b, c.

    dataset = ranking_dataset()
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        reports = evaluation.evaluate(client, dataset)
    assert reports[0].metrics == {"top1": 0.0, "hit_at_k": 1.0, "recall_at_k": 0.5, "mrr": 0.5}
    assert reports[1].metrics == dict.fromkeys(reports[0].metrics, 0.0)
    assert evaluation.summarize(dataset, reports) == {
        "cases": 2,
        "failed": 1,
        "top1": 0.0,
        "hit_at_k": 0.5,
        "recall_at_k": 0.25,
        "mrr": 0.25,
        "input_order_top1": 0.0,
        "input_order_hit_at_k": 1.0,
        "input_order_recall_at_k": 0.5,
        "input_order_mrr": 0.5,
    }
    result = reports[0].result
    assert isinstance(result, RankingResult)
    assert (
        evaluation.ranking_metrics(["b", "c"], [item.id for item in result.ranked], k=10)[
            "recall_at_k"
        ]
        == 1
    )
    assert reports[0].input_order_metrics == reports[1].input_order_metrics == reports[0].metrics
    assert reports[1].error == {"type": "TransportError", "request_id": None}
    for call, case in zip(calls, dataset.cases, strict=True):
        assert json.loads(call.content) == case.request.as_decision().model_dump()


def test_changed_model_is_retained_but_counts_as_failure() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        model = MODEL if calls != 2 else MODEL.model_copy(update={"revision": "v2"})
        return response(request, model=model)

    dataset = choice_dataset()
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        reports = evaluation.evaluate(client, dataset)
    assert reports[1].error == {"type": "ModelChanged"}
    assert reports[1].metrics == {"exact_match": 0.0, "question_accuracy": 0.0}
    assert isinstance(reports[1].result, DecisionResult)
    assert reports[1].result.model.revision == "v2"
    assert reports[2].error is None and reports[2].metrics["exact_match"] == 1
    assert evaluation.summarize(dataset, reports)["failed"] == 1


def test_malformed_response_counts_as_failure_and_next_case_runs() -> None:
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            wrong = choice_case("wrong", {"unexpected_question": ["a"]})
            return response(
                httpx.Request("POST", "http://test", content=wrong.request.model_dump_json())
            )
        return response(request)

    dataset = choice_dataset()
    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        reports = evaluation.evaluate(client, dataset)
    assert reports[0].error is not None and reports[0].error["type"] == "TransportError"
    assert reports[0].metrics["exact_match"] == 0
    assert reports[2].metrics["exact_match"] == 1
    assert calls == 3


@pytest.mark.parametrize(
    "invalid",
    [
        b"",
        b"not JSON",
        b"\xff",
        b"{}",
        b'{"kind":"choice","name":"empty","examples":["x"],"cases":[]}',
    ],
)
def test_bad_later_file_fails_before_any_client(
    invalid: bytes,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = tmp_path / "good.json"
    good.write_text(choice_dataset().model_dump_json())
    bad = tmp_path / "bad.json"
    bad.write_bytes(invalid)

    def unexpected_client(**kwargs: object) -> Client:
        pytest.fail("all datasets must validate before client creation")

    monkeypatch.setattr(evaluation, "Client", unexpected_client)
    monkeypatch.setattr(sys, "argv", ["evaluate_use_cases", str(good), str(bad)])
    assert evaluation.main() == 2
    output = capsys.readouterr()
    assert output.out == "" and "Evaluation could not complete:" in output.err


@pytest.mark.parametrize(
    "expected", [{}, {"topic": []}, {"topic": ["unknown"]}, {"topic": ["a", "a"]}, {"wrong": ["a"]}]
)
def test_invalid_choice_labels_rejected(expected: dict[str, list[str]]) -> None:
    payload = choice_case("one", {"topic": ["a"]}).model_dump()
    payload["expected"] = expected
    with pytest.raises(ValueError):
        evaluation.ChoiceCase.model_validate(payload)


@pytest.mark.parametrize("relevant", [[], ["unknown"], ["b", "b"]])
def test_invalid_ranking_labels_rejected(relevant: list[str]) -> None:
    payload = ranking_dataset().cases[0].model_dump()
    payload["relevant"] = relevant
    with pytest.raises(ValueError):
        evaluation.RankingCase.model_validate(payload)


@pytest.mark.parametrize("duplicate", ["name", "case"])
def test_duplicate_names_and_ids_rejected_before_execution(
    duplicate: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = choice_dataset()
    if duplicate == "case":
        dataset.cases.append(dataset.cases[0])
    source = tmp_path / "cases.json"
    source.write_text(dataset.model_dump_json())
    paths = [str(source)] * (2 if duplicate == "name" else 1)
    monkeypatch.setattr(sys, "argv", ["evaluate_use_cases", *paths, "--validate"])
    assert evaluation.main() == 2


@pytest.mark.parametrize("kind", ["choice", "ranking"])
@pytest.mark.parametrize("correct", [False, True])
def test_http_cli_configuration_reports_and_exit_status(
    kind: str,
    correct: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset: evaluation.Dataset
    if kind == "choice":
        dataset = choice_dataset().model_copy(update={"cases": [choice_case("one", {"q": ["b"]})]})
    else:
        dataset = ranking_dataset()
    source = tmp_path / "cases.json"
    source.write_text(dataset.model_dump_json())
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer private-key"
        return response(request, winner=1 if correct else 0)

    def client(*, base_url: str, api_key: str | None) -> Client:
        assert base_url == "http://evaluation.test"
        return Client(base_url=base_url, api_key=api_key, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(evaluation, "Client", client)
    monkeypatch.setenv("KAYAK_BASE_URL", "http://evaluation.test")
    monkeypatch.setenv("KAYAK_API_KEY", "private-key")
    monkeypatch.setattr(sys, "argv", ["evaluate_use_cases", str(source)])
    assert evaluation.main() == (0 if correct else 1)
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == "" and "private-key" not in output.out
    assert report["mode"] == "http" and report["model_quality"] == "starter_cases_only"
    assert report["dataset_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report["kind"] == kind
    assert report["summary"]["failed"] == 0
    assert report["summary"]["exact_match" if kind == "choice" else "top1"] == int(correct)
    if kind == "ranking":
        assert report["summary"]["input_order_top1"] == 0
        assert report["summary"]["input_order_mrr"] == 0.5
        assert report["summary"]["mrr"] == (1 if correct else 0.5)
    assert len(calls) == len(dataset.cases)


@pytest.mark.parametrize("path", DATASETS, ids=lambda path: path.stem)
def test_starter_dataset_validity_hashes_and_discoverability(path: Path) -> None:
    dataset, digest = evaluation.read_dataset(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(dataset.cases) >= 3  # Ordinary, confusable, and boundary cases.
    assert all((ROOT / example).is_file() for example in dataset.examples)
    guide = (path.parent / "README.md").read_text()
    assert f"]({path.name})" in guide


@pytest.mark.parametrize("simulate", [False, True])
def test_cli_modes_need_no_model_and_identify_simulation(simulate: bool) -> None:
    mode = "--simulate" if simulate else "--validate"
    completed = subprocess.run(
        [sys.executable, "-m", "examples.evaluate_use_cases", *map(str, DATASETS), mode],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(records) == len(DATASETS)
    assert completed.stderr == ""
    for record in records:
        assert record["mode"] == ("simulated" if simulate else "validate")
        assert record["model_quality"] == "not_measured"
        if simulate:
            assert record["summary"]["failed"] == 0
            for case in record["cases"]:
                assert case["result"]["model"]["fingerprint"] == "simulated-no-weights"
    if simulate:
        assert any(record["summary"].get("exact_match") == 0 for record in records)


def test_simulation_ignores_labels_and_does_not_import_inference() -> None:
    script = """
import sys
from examples import evaluate_use_cases as evaluation
from kayak import Client
import httpx
dataset, _ = evaluation.read_dataset(__import__('pathlib').Path(sys.argv[1]))
transport = httpx.MockTransport(evaluation.simulated_response)
with Client(base_url='http://test', transport=transport) as c:
    first = evaluation.evaluate(c, dataset)
    for case in dataset.cases:
        for question_id, question in case.request.questions.items():
            case.expected[question_id] = [list(question.criteria)[-1]]
    second = evaluation.evaluate(c, dataset)
assert [r.result for r in first] == [r.result for r in second]
assert not {'torch', 'transformers', 'kayak.inference'}.intersection(sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "examples/evaluations/feedback.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
