"""Check hardware-report summaries without claiming accelerator validation."""

import json
import sys
from pathlib import Path

import pytest

import kayak
from kayak import DecisionResult, Model
from kayak.bundle import ModelSpec

pytestmark = pytest.mark.inference


def test_timings_separate_first_call_and_preserve_warm_spread() -> None:
    pytest.importorskip("torch")
    from scripts.validate_model import timing_summary

    assert timing_summary([100.0, 1.0, 3.0, 2.0]) == {
        "first_seconds": 100.0,
        "warm_runs": 3,
        "warm_median_seconds": 2.0,
        "warm_min_seconds": 1.0,
        "warm_max_seconds": 3.0,
    }
    with pytest.raises(ValueError, match="at least two"):
        timing_summary([1.0])


def test_cpu_memory_is_labeled_as_process_peak() -> None:
    pytest.importorskip("torch")
    from scripts.validate_model import memory_snapshot

    memory = memory_snapshot("cpu")
    assert set(memory) == {"process_peak_rss_bytes"}
    peak = memory["process_peak_rss_bytes"]
    assert peak is not None and peak > 0


def test_unavailable_process_memory_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    from scripts import validate_model

    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert validate_model.memory_snapshot("cpu") == {"process_peak_rss_bytes": None}


@pytest.mark.parametrize("device", ["mps", "mps:0"])
def test_mps_memory_includes_indexed_device(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    torch = pytest.importorskip("torch")
    from scripts.validate_model import memory_snapshot

    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 123)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 456)
    snapshot = memory_snapshot(device)
    assert snapshot["mps_current_tensor_bytes"] == 123
    assert snapshot["mps_current_driver_bytes"] == 456


@pytest.mark.parametrize("http_failure", [False, True])
def test_validator_writes_repeat_measurements_and_retains_failures(
    tiny_model: Model,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_failure: bool,
) -> None:
    from kayak.server import DecisionModel
    from scripts import validate_model

    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["validate_model", "--repeats", "4", "--output", str(output)])

    def resolve(
        model: str, *, cache_dir: str | None, local_files_only: bool
    ) -> tuple[ModelSpec, Path]:
        assert model == "Contrastive-LM/CLM-v0.1-8B"
        return tiny_model.spec, tmp_path / "heads.pt"

    def head_reference(checkpoint: Path, spec: ModelSpec) -> dict[str, object]:
        # The released heads have their own independent fixture test. This test
        # exercises the report and real tiny-model/HTTP paths without Hub access.
        assert checkpoint.name == "heads.pt" and spec.model_id == "tests/tiny-clm"
        return {"status": "not_run", "reason": "tiny-model test fixture"}

    def load(*, device: str, dtype: str, cache_dir: str | None, local_files_only: bool) -> Model:
        assert device == dtype == "auto"
        return tiny_model

    monkeypatch.setattr(validate_model, "resolve", resolve)
    monkeypatch.setattr(validate_model, "head_reference", head_reference)
    monkeypatch.setattr(kayak, "load", load)
    if http_failure:

        def fail_http(
            model: DecisionModel,
            case: validate_model.Case,
            direct: DecisionResult,
        ) -> bool:
            raise AssertionError("injected HTTP mismatch")

        monkeypatch.setattr(validate_model, "validate_http", fail_http)
        with pytest.raises(AssertionError, match="injected HTTP mismatch"):
            validate_model.main()
    else:
        validate_model.main()

    report = json.loads(output.read_text())
    assert report["repeats"] == 4
    assert report["model"]["id"] == "tests/tiny-clm"
    assert report["requested_device"] == "auto"
    assert report["model"]["device"] == "cpu"
    assert report["full_model"] == "passed"
    assert report["encoder_reference"] == "not_run"
    assert report["http"] == ("not_run" if http_failure else "passed")
    assert ("error" in report) == http_failure
    for case in report["cases"]:
        assert len(case["seconds"]) == len(case["scores"]) == len(case["choices"]) == 4
        assert case["timing"]["warm_runs"] == 3
        assert case["timing"]["first_seconds"] == case["seconds"][0]
        assert case["memory"]["process_peak_rss_bytes"] > 0
