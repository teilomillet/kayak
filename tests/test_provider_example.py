"""The provider workflow preserves evidence without leaking labels into inference."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.evaluate_provider import run
from kayak.adapters import Jev, Laya
from kayak.eval import PredictionSet, Suite, load_suite

ROOT = Path(__file__).resolve().parents[1]


class Provider:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.calls: list[tuple[str, dict[str, dict[str, object]]]] = []
        self.failure = failure

    def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object:
        self.calls.append((state, copy.deepcopy(questions)))
        questions.clear()
        if self.failure is not None and len(self.calls) == 2:
            raise self.failure
        return {
            "model": "controlled-fixture",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": "shipping",
                    "probabilities": {"billing": 0.3333, "shipping": 0.3333, "account": 0.3333},
                    "confidence": 0.25,
                }
            },
        }

    def system_one(self, *, state: str, questions: dict[str, dict[str, object]]) -> SimpleNamespace:
        body = self.predict(state, questions)
        return SimpleNamespace(raw_http_response=SimpleNamespace(content=json.dumps(body).encode()))


@pytest.fixture
def suite() -> Suite:
    return load_suite(ROOT / "examples/suites/support.json")


@pytest.mark.parametrize("adapter", [Laya, Jev])
def test_shared_workflow_keeps_labels_local_and_rounded_evidence_intact(
    tmp_path: Path, suite: Suite, adapter: type[Laya] | type[Jev]
) -> None:
    before = suite.model_dump()
    provider = Provider()
    output = tmp_path / "run"
    predictions = run(
        adapter(provider), suite, output=output, system="fixture", evidence_kind="simulation"
    )
    assert suite.model_dump() == before
    assert provider.calls == [
        (case.text, {"intent": suite.question.model_dump(exclude_none=True)})
        for case in suite.examples
    ]
    assert len(predictions.predictions) == 8
    assert all(row.choice == "shipping" for row in predictions.predictions)
    assert all(row.probabilities is None for row in predictions.predictions)
    assert predictions.metadata["original_status"] == "complete"
    assert (
        PredictionSet.model_validate_json((output / "predictions.json").read_text()) == predictions
    )
    responses = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    raw_answer = responses[0]["result"]["raw"]["answers"]["intent"]
    assert raw_answer["confidence"] == 0.25
    assert raw_answer["probabilities"] == {"billing": 0.3333, "shipping": 0.3333, "account": 0.3333}
    report = json.loads((output / "comparison/benchmark.json").read_text())
    assert report["runs"]["provider"]["metrics"]["accuracy"]["value"] == 3 / 8
    assert report["comparisons"][0]["eligible"] is True

    # Changing only expected labels cannot alter the requests or predictions.
    for case in suite.examples:
        case.label = "account"
    changed_provider = Provider()
    changed = run(
        adapter(changed_provider),
        suite,
        output=tmp_path / "changed-labels",
        system="fixture",
        evidence_kind="simulation",
    )
    assert changed_provider.calls == provider.calls
    assert changed.predictions == predictions.predictions


@pytest.mark.parametrize("adapter", [Laya, Jev])
@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_CREDENTIAL"), KeyboardInterrupt()])
def test_failure_stops_calls_and_retains_full_denominator(
    tmp_path: Path, suite: Suite, adapter: type[Laya] | type[Jev], failure: BaseException
) -> None:
    provider = Provider(failure)
    output = tmp_path / "failed"
    with pytest.raises(type(failure)):
        run(adapter(provider), suite, output=output, system="fixture", evidence_kind="simulation")
    assert len(provider.calls) == 2  # No retry, and no calls after the failure.
    predictions = PredictionSet.model_validate_json((output / "predictions.json").read_text())
    assert len(predictions.suite.examples) == 8
    assert len(predictions.predictions) == 1
    assert predictions.metadata["original_status"] == "incomplete"
    journal = (output / "responses.jsonl").read_text()
    assert "PRIVATE_CREDENTIAL" not in journal
    assert json.loads(journal.splitlines()[1]) == {
        "id": suite.examples[1].id,
        "error": type(failure).__name__,
    }
    report = json.loads((output / "comparison/benchmark.json").read_text())
    assert report["runs"]["provider"]["examples"] == 8
    assert report["runs"]["provider"]["answered_examples"] == 1
    assert report["runs"]["provider"]["metrics"]["accuracy"]["value"] == 0.0
    assert report["comparisons"][0]["eligible"] is False


def test_existing_output_is_rejected_before_inference(tmp_path: Path, suite: Suite) -> None:
    provider = Provider()
    marker = tmp_path / "predictions.json"
    marker.write_text("previous evidence", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(Laya(provider), suite, output=tmp_path, system="fixture", evidence_kind="simulation")
    assert not provider.calls
    assert marker.read_text() == "previous evidence"


def test_default_command_needs_no_sdk_model_or_credentials(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "sys.argv = ['evaluate_provider', '--output', sys.argv[1]]; "
            "runpy.run_module('examples.evaluate_provider', run_name='__main__'); "
            "assert not {'laya', 'typesafe_sdk', 'torch', 'transformers'} & sys.modules.keys()",
            str(tmp_path / "offline"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SIMULATION" in result.stdout
    predictions = PredictionSet.model_validate_json(
        (tmp_path / "offline/predictions.json").read_text()
    )
    assert predictions.metadata["evidence_kind"] == "simulation"


def test_jev_requires_key_before_importing_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.evaluate_provider",
            "--provider",
            "jev",
            "--model",
            "fixture",
            "--output",
            str(tmp_path / "unused"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "set TYPESAFE_API_KEY" in result.stderr
    assert not (tmp_path / "unused").exists()
