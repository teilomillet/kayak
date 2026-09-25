"""Exercise the portable runner/reviewer handoff without Python pipeline objects."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kayak.eval import (
    RAGAttempt,
    RAGCase,
    RAGContext,
    RAGDataset,
    RAGEvalConfig,
    RAGGate,
    RAGInput,
    RAGJudgments,
    RAGOutput,
    RAGReview,
    RAGReviewInput,
    RAGReviewRecord,
    RAGTrace,
    load_rag_report,
)
from kayak.eval._rag_cli import main


def files(directory: Path) -> tuple[Path, Path, Path]:
    case = RAGCase(
        input=RAGInput(id="case-1", query="Which plan applies?", parameters={"tenant": "A"}),
        reference_answer="Plan A",
        judgments=RAGJudgments(
            relevance={"policy": 1},
            evidence_sets=[["policy"]],
            evidence_texts={"policy": "Plan A"},
        ),
    )
    dataset = RAGDataset(name="portable-fixture", cases=[case])
    config = RAGEvalConfig(
        system={"implementation": "fixture-v1"},
        gates=[RAGGate(metric="answer.correct", minimum=1.0)],
    )
    output = RAGOutput(
        final=RAGTrace(
            id=case.input.id,
            query=case.input.query,
            context_input_ids=["policy"],
            context=RAGContext(ids=["policy"], text="Plan A"),
            answer="Plan A",
        )
    )
    attempt = RAGAttempt(input=case.input, system=config.system, output=output)
    paths = directory / "dataset.json", directory / "config.json", directory / "attempts.jsonl"
    for path, record in zip(paths, (dataset, config, attempt), strict=True):
        path.write_text(record.model_dump_json() + "\n")
    return paths


def test_input_export_contains_application_data_without_gold(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, config, _ = files(tmp_path)
    assert main(["inputs", str(dataset), "--config", str(config)]) == 0
    raw = capsys.readouterr().out
    row = json.loads(raw)
    assert row["input"]["parameters"] == {"tenant": "A"}
    assert row["system"] == {"implementation": "fixture-v1"}
    assert row["repeat"] == 0
    assert "judgments" not in raw and "reference_answer" not in raw and "Plan A" not in raw


def test_foreign_reviewer_uses_provided_fingerprint_and_native_json_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, config, attempts = files(tmp_path)
    assert main(["review-inputs", str(dataset), str(attempts), "--config", str(config)]) == 0
    request = json.loads(capsys.readouterr().out)
    # A foreign process may reorder keys or add whitespace. It returns the supplied
    # fingerprint, rather than hashing its own JSON serialization.
    wire = json.dumps(request["review_input"], sort_keys=True, indent=4)
    observed = RAGReviewInput.model_validate_json(wire)
    assert observed.sha256 == request["review_input_sha256"]
    assert observed.case.reference_answer == "Plan A"
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        json.dumps(
            {
                "case_id": request["case_id"],
                "repeat": request["repeat"],
                "review": {
                    "review_input_sha256": request["review_input_sha256"],
                    "correct": True,
                    "provenance": {"rubric": "independent fixture exact match"},
                },
            }
        )
        + "\n"
    )
    output = tmp_path / "report.json"
    arguments = [
        "score",
        str(dataset),
        str(attempts),
        "--config",
        str(config),
        "--reviews",
        str(reviews),
        "--output",
        str(output),
    ]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
    report = load_rag_report(output)
    assert report.summary.metrics["answer.correct"].mean == 1.0
    assert report.summary.metrics["answer.grounded"].mean is None
    assert report.results[0].attempt.duration_seconds is None
    assert main(["report", str(output)]) == 0
    assert "unknown" in capsys.readouterr().out.lower()
    original = output.read_bytes()
    assert main(arguments) == 2
    assert output.read_bytes() == original


def test_unknown_quality_fails_configured_gate_and_missing_records_remain_visible(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, config, attempts = files(tmp_path)
    output = tmp_path / "unknown.json"
    assert (
        main(
            ["score", str(dataset), str(attempts), "--config", str(config), "--output", str(output)]
        )
        == 1
    )
    report = load_rag_report(output)
    assert report.status == "complete" and report.passed is False
    assert report.summary.metrics["answer.correct"].observed == 0
    assert report.summary.metrics["answer.correct"].unknown == 1
    attempts.write_text("")
    output = tmp_path / "missing.json"
    assert (
        main(
            ["score", str(dataset), str(attempts), "--config", str(config), "--output", str(output)]
        )
        == 1
    )
    report = load_rag_report(output)
    assert report.status == "incomplete" and report.summary.missing == 1
    assert report.passed is False
    assert not capsys.readouterr().err


@pytest.mark.parametrize("change", ["parameters", "reference", "system"])
def test_changed_review_material_or_system_cannot_reuse_successful_review(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    dataset_path, config_path, attempts_path = files(tmp_path)
    dataset = RAGDataset.model_validate_json(dataset_path.read_bytes())
    attempt = RAGAttempt.model_validate_json(attempts_path.read_bytes())
    assert attempt.output is not None
    request = RAGReviewInput(case=dataset.cases[0], output=attempt.output)
    record = RAGReviewRecord(
        case_id=attempt.case_id,
        review=RAGReview(
            review_input_sha256=request.sha256,
            correct=True,
        ),
    )
    reviews_path = tmp_path / "reviews.jsonl"
    reviews_path.write_text(record.model_dump_json() + "\n")
    if change == "parameters":
        dataset.cases[0].input.parameters["tenant"] = "B"
    elif change == "reference":
        dataset.cases[0] = dataset.cases[0].model_copy(update={"reference_answer": "Plan B"})
    else:
        config = RAGEvalConfig.model_validate_json(config_path.read_bytes())
        config.system["implementation"] = "a different model"
        config_path.write_text(config.model_dump_json())
    dataset_path.write_text(dataset.model_dump_json())
    output = tmp_path / "invalid.json"
    assert (
        main(
            [
                "score",
                str(dataset_path),
                str(attempts_path),
                "--config",
                str(config_path),
                "--reviews",
                str(reviews_path),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    assert capsys.readouterr().err


def test_invalid_external_records_do_not_expose_their_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset, config, attempts = files(tmp_path)
    attempts.write_text('{"input": "PRIVATE-SERVICE-PAYLOAD"}\n')
    output = tmp_path / "invalid.json"
    assert (
        main(
            ["score", str(dataset), str(attempts), "--config", str(config), "--output", str(output)]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "line 1" in error and "PRIVATE-SERVICE-PAYLOAD" not in error
    assert not output.exists()


@pytest.mark.parametrize("version", [True, 1.0, "1", 2])
def test_portable_schema_versions_require_the_declared_integer(version: object) -> None:
    with pytest.raises(ValueError, match="integer 1"):
        RAGEvalConfig.model_validate({"schema_version": version})


def test_schema_export_works_through_both_entry_points_without_optional_imports() -> None:
    for module in ("kayak", "kayak.eval"):
        arguments = [sys.executable, "-m", module]
        if module == "kayak":
            arguments.append("eval")
        completed = subprocess.run(
            [*arguments, "rag", "schema", "attempt"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        schema = json.loads(completed.stdout)
        assert "input" in schema["required"] and "$defs" in schema
        assert not completed.stderr
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from kayak.eval._rag_cli import main; main(['schema', 'review_input']); import sys; "
            "assert not {'torch', 'transformers', 'huggingface_hub', 'fastapi'} "
            "& sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
