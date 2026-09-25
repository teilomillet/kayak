"""Public reports retain failure scope, identify fixtures, and reject altered evidence."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.mock_evaluation import MockBackend, mock_suite, run
from kayak.eval import _cli, evaluate, load_report, summarize
from kayak.eval._report import render_report


def test_mock_campaign_keeps_failed_and_interrupted_runs(tmp_path: Path) -> None:
    root = tmp_path / "audit"
    run(root)
    expected = {
        "complete": (0.5, 0, 0),
        "failed": (0.5, 1, 2),
        "interrupted": (0.0, 3, 0),
    }
    for name, (accuracy, missing, failed_calls) in expected.items():
        report = load_report(root / name)
        assert report.status == name
        assert report.summary["examples"] == 4
        assert report.summary["accuracy"] == accuracy
        assert report.summary["failed_or_missing_examples"] == missing
        assert report.summary["failed_calls"] == failed_calls
        rendered = (root / name / "report.md").read_text()
        assert "MOCK EVIDENCE" in rendered
        assert "model quality and inference performance are not measured" in rendered
        assert f"**Run status:** `{name}`" in rendered
        assert "Maximum score change from first successful attempt" in rendered
        assert "| Selected examples (quality denominator) | 4 |" in rendered
        if name != "complete":
            assert "not an accepted completed benchmark" in rendered
        if name == "interrupted":
            assert "An interrupted in-flight call may have no saved result or duration" in rendered
    with pytest.raises(FileExistsError):
        run(root)


def test_report_command_verifies_before_printing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "run"
    evaluate(MockBackend(), mock_suite(), output=output, config={"evidence_kind": "mock"})
    assert _cli.main(["report", str(output)]) == 0
    assert "MOCK EVIDENCE" in capsys.readouterr().out
    predictions = output / "predictions.jsonl"
    predictions.write_bytes(predictions.read_bytes() + b"\n")
    assert _cli.main(["report", str(output)]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert captured.err


def test_legacy_artifacts_do_not_gain_hash_guarantees(tmp_path: Path) -> None:
    output = tmp_path / "run"
    evaluate(MockBackend(), mock_suite(), output=output)
    metadata = output / "report.json"
    payload = json.loads(metadata.read_text())
    payload["summary"] = summarize(load_report(output), legacy=True)
    payload["schema_version"] = 1
    payload.pop("prediction_artifact")
    metadata.write_text(json.dumps(payload))
    rendered = render_report(output)
    assert "Legacy artifact" in rendered
    assert "no recorded digest" in rendered
    assert "CUSTOM BACKEND" in rendered
    assert "complete prediction file passed" not in rendered
    assert "Legacy timing boundaries" in rendered
    assert "| Support-weighted F1 | 0.333333 |" in rendered
    assert "| Supported intents with zero recall | 1 |" in rendered


def test_custom_local_synchronization_is_not_claimed_as_verified(tmp_path: Path) -> None:
    output = tmp_path / "run"
    evaluate(MockBackend(), mock_suite(), output=output)
    metadata = output / "report.json"
    payload = json.loads(metadata.read_text())
    payload["transport"] = "local"
    payload["environment"]["synchronization"] = "custom"
    metadata.write_text(json.dumps(payload))
    rendered = render_report(output)
    assert "any caller-supplied synchronization hook" in rendered
    assert "Completion of backend work is not independently verified" in rendered
    assert "recorded completion synchronization" not in rendered


def test_report_and_mock_campaign_need_no_inference_imports(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from benchmarks.mock_evaluation import run; "
            "run(Path(sys.argv[1])); "
            "assert not {'torch', 'transformers', 'fastapi'} & sys.modules.keys()",
            str(tmp_path / "mock"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
