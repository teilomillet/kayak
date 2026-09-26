"""Own-file classification validates before effects and preserves completed results."""

import copy
import json
import shlex
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import NoReturn

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


def test_missing_dependency_gives_a_copyable_command_and_allows_retry_at_same_path(
    files: tuple[Path, Path, Path],
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = sys.modules["laya"]
    monkeypatch.setitem(sys.modules, "laya", None)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert not files[2].exists()
    assert not provider.loads
    error = capsys.readouterr().err
    command = next(line.strip() for line in error.splitlines() if line.startswith("  uv "))
    assert shlex.split(command) == [
        "uv",
        "run",
        "--with",
        "laya==0.3.20",
        "--with",
        "transformers<5",
        "-m",
        "examples.classify_file",
        *sys.argv[1:],
    ]
    monkeypatch.setitem(sys.modules, "laya", module)
    main()
    assert len(files[2].read_text().splitlines()) == 2


def test_input_error_names_the_field_and_line_then_accepts_the_correction(
    files: tuple[Path, Path, Path],
    provider: Provider,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source, _, output = files
    original = source.read_text()
    source.write_text(original.splitlines()[0] + '\n{"id":"bad"}\n')
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert f"{source}:2: invalid ticket: text: Field required" in error
    assert "one JSON object" in error
    assert not provider.loads and not output.exists()
    source.write_text(original)
    main()
    assert len(output.read_text().splitlines()) == 2


def test_question_error_names_the_invalid_category_without_echoing_its_value(
    files: tuple[Path, Path, Path],
    provider: Provider,
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = files[1]
    question.write_text(json.dumps({**QUESTION, "criteria": {"help": ["PRIVATE_PAYLOAD"]}}))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert str(question) in error and "criteria.help" in error
    assert "Edit the instructions and criteria" in error
    assert "PRIVATE_PAYLOAD" not in error
    assert not provider.loads and not files[2].exists()


def test_preview_is_bounded_escapes_controls_and_preserves_complete_saved_values(
    files: tuple[Path, Path, Path],
    provider: Provider,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identifiers = [
        "unicode-α\n\x1b[31m\x9b\u2028",
        "x" * 200,
        "three",
        "four",
        "five",
        "six",
        "seven",
    ]
    files[0].write_text(
        "".join(
            json.dumps({"id": identifier, "text": "PRIVATE_TICKET"}) + "\n"
            for identifier in identifiers
        )
    )
    main()
    captured = capsys.readouterr()
    assert "Preview (first 5 predictions):" in captured.out
    preview = [line for line in captured.out.splitlines() if line.startswith("  ")]
    assert len(preview) == 5
    assert '\\u03b1\\n\\u001b[31m\\u009b\\u2028" -> "help"' in preview[0]
    assert "\x1b" not in captured.out and "PRIVATE_TICKET" not in captured.out + captured.err
    assert len(preview[1]) < 100 and "..." in preview[1]
    assert "Processing record 7/7" in captured.err
    # JSONL records end at file newlines, not Unicode separators inside JSON strings.
    with files[2].open(encoding="utf-8") as stream:
        saved = [json.loads(line) for line in stream]
    assert [row["id"] for row in saved] == identifiers
    assert all(row["choice"] == "help" and row["result"]["raw"] for row in saved)


def test_loading_and_record_progress_are_visible_before_the_operations(
    files: tuple[Path, Path, Path],
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_predict = provider.predict

    @contextmanager
    def load(model: str, *, device: str) -> Iterator[Provider]:
        assert "Loading Laya on cpu" in capsys.readouterr().err
        yield provider

    def predict(state: str, questions: dict[str, dict[str, object]]) -> object:
        captured = capsys.readouterr()
        assert not captured.out  # Preview follows progress, so a long run cannot bury it.
        error = captured.err
        assert f"Processing record {len(provider.calls) + 1}/2" in error
        if not provider.calls:
            assert "Model loaded" in error
        return original_predict(state, questions)

    monkeypatch.setattr(sys.modules["laya"], "load", load)
    monkeypatch.setattr(provider, "predict", predict)
    main()
    assert len(provider.calls) == 2


def test_load_failure_identifies_the_stage_and_retains_no_fabricated_predictions(
    files: tuple[Path, Path, Path],
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def load(model: str, *, device: str) -> NoReturn:
        raise FileNotFoundError("PRIVATE_PAYLOAD")

    monkeypatch.setattr(sys.modules["laya"], "load", load)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "during model loading (FileNotFoundError)" in captured.err
    assert "Check --model" in captured.err and "--device" in captured.err
    assert "PRIVATE_PAYLOAD" not in captured.err
    assert not captured.out and not provider.calls
    assert files[2].read_bytes() == b""


def test_output_created_during_import_is_preserved(
    files: tuple[Path, Path, Path],
    provider: Provider,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = sys.modules["laya"]

    def import_module(name: str) -> ModuleType:
        assert name == "laya"
        files[2].write_text("created by another run\n")
        return module

    monkeypatch.setattr("examples.classify_file.importlib.import_module", import_module)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "Choose a new --output path" in capsys.readouterr().err
    assert files[2].read_text() == "created by another run\n"
    assert not provider.loads
