"""Own-file classification validates before effects and preserves completed results."""

import copy
import json
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from examples.classify_file import ClassifiedRecord, main

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {
    "type": "choice",
    "instructions": "Where should this message go?",
    "criteria": {"sales": "Purchase enquiries", "help": "Help using the product"},
}
RECORDS = [{"id": "request-α", "text": "How much?\nDeux licences."}, {"id": "b", "text": "Help!"}]


class Provider:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.calls: list[tuple[str, dict[str, dict[str, object]]]] = []
        self.loads: list[tuple[str, str]] = []
        self.closed = False

    def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object:
        self.calls.append((state, copy.deepcopy(questions)))
        questions.clear()
        if self.failure is not None and len(self.calls) == 2:
            raise self.failure
        return {
            "model": "controlled-response",
            "answers": {"category": {"type": "choice", "choice": "help", "confidence": 0.42}},
        }

    @contextmanager
    def load(self, model: str, *, device: str) -> Iterator["Provider"]:
        self.loads.append((model, device))
        try:
            yield self
        finally:
            self.closed = True


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> Provider:
    fixture = Provider()
    module = ModuleType("laya")
    module.load = fixture.load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", module)
    return fixture


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    source = tmp_path / "requests.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in RECORDS) + "\n", encoding="utf-8")
    question = tmp_path / "categories.json"
    question.write_text(json.dumps(QUESTION), encoding="utf-8")
    output = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "classify_file",
            str(source),
            "--question",
            str(question),
            "--model",
            "local-model",
            "--output",
            str(output),
        ],
    )
    return source, question, output


def test_custom_categories_preserve_records_and_provider_evidence(
    files: tuple[Path, Path, Path], provider: Provider, capsys: pytest.CaptureFixture[str]
) -> None:
    main()
    output = files[2]
    records = [
        ClassifiedRecord.model_validate_json(line) for line in output.read_text().splitlines()
    ]
    assert [row.id for row in records] == ["request-α", "b"]
    assert [row.choice for row in records] == ["help", "help"]
    assert all(row.result.answers["category"].type == "choice" for row in records)
    assert all(
        row.result.raw["answers"]
        == {"category": {"type": "choice", "choice": "help", "confidence": 0.42}}
        for row in records
    )
    assert all(row.result.calibration == "unknown" for row in records)
    assert provider.calls == [(row["text"], {"category": QUESTION}) for row in RECORDS]
    assert provider.loads == [("local-model", "cpu")]
    assert provider.closed
    assert "Classified 2 records" in capsys.readouterr().out


def test_validation_runs_without_importing_laya(
    files: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, question, output = files
    monkeypatch.setitem(sys.modules, "laya", None)
    monkeypatch.setattr(
        sys, "argv", ["classify_file", str(source), "--question", str(question), "--validate"]
    )
    main()
    assert "Validated 2 records" in capsys.readouterr().out
    assert not output.exists()


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"id":"bad","text":"   "}',
        '{"id":"bad","text":"hello","label":"help"}',
        '{"id":"bad","text":5}',
        '{"id":"bad","text":"' + "a" * 65_537 + '"}',
        "not JSON",
    ],
)
def test_late_invalid_input_is_rejected_before_model_load(
    files: tuple[Path, Path, Path],
    provider: Provider,
    bad_line: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _, output = files
    source.write_text(json.dumps(RECORDS[0]) + "\n" + bad_line + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert f"{source}:2:" in capsys.readouterr().err
    assert not provider.loads
    assert not output.exists()


@pytest.mark.parametrize("fault", ["empty_input", "bad_question", "combined_limit"])
def test_request_errors_have_no_inference_or_output_effects(
    files: tuple[Path, Path, Path],
    provider: Provider,
    fault: str,
) -> None:
    source, question, output = files
    if fault == "empty_input":
        source.write_text("", encoding="utf-8")
    elif fault == "bad_question":
        question.write_text('{"instructions":"choose","criteria":{}}', encoding="utf-8")
    else:
        question.write_text(
            json.dumps(
                {"instructions": "choose", "criteria": {str(i): "a" * 65_536 for i in range(4)}}
            ),
            encoding="utf-8",
        )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not provider.loads
    assert not output.exists()


def test_existing_output_is_never_overwritten_or_model_loaded(
    files: tuple[Path, Path, Path],
    provider: Provider,
) -> None:
    output = files[2]
    output.write_text("previous results\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert not provider.loads
    assert output.read_text() == "previous results\n"


@pytest.mark.parametrize("failure", [RuntimeError("PRIVATE_PAYLOAD"), KeyboardInterrupt()])
def test_failure_retains_completed_rows_closes_model_and_never_retries(
    files: tuple[Path, Path, Path],
    provider: Provider,
    failure: BaseException,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider.failure = failure
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert len(provider.calls) == 2
    assert provider.closed
    lines = files[2].read_text().splitlines()
    assert len(lines) == 1
    assert ClassifiedRecord.model_validate_json(lines[0]).id == "request-α"
    captured = capsys.readouterr()
    assert "partial results" in captured.err
    assert "PRIVATE_PAYLOAD" not in captured.err
    assert "Classified" not in captured.out


def test_documented_validation_command() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.classify_file",
            "examples/tickets.jsonl",
            "--question",
            "examples/department.json",
            "--validate",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "Validated 2 records and the question. No model loaded.\n"
