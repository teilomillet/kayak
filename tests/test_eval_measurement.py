"""Controlled provenance and clocks; these checks never load an inference model."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import Mock

import pytest

from kayak import Choice, DecisionResult, Model, ModelInfo
from kayak.decisions import answer_from_scores
from kayak.eval import Comparison, Example, Suite, _compare, _measurement, _runner, evaluate
from kayak.eval._schema import Attempt, Observation, PredictionArtifact, Protocol, Report

MODEL = ModelInfo(
    id="controlled",
    revision="1",
    fingerprint="fixture",
    encoder="none",
    encoder_revision="none",
    device="cpu",
    dtype="float32",
)


def recorded_environment() -> dict[str, object]:
    return {
        "platform": "controlled OS",
        "machine": "controlled architecture",
        "host": "controlled host",
        "python": "controlled Python",
        "versions": {
            name: "1.0"
            for name in ("kayak", "torch", "transformers", "tokenizers", "pydantic", "httpx")
        },
        "kayak_source_sha256": "a" * 64,
        "hardware": {
            "cpu_model": "Controlled CPU",
            "logical_cpus": 8,
            "cpu_affinity": {"status": "recorded", "cpus": [0, 1]},
            "accelerator": None,
        },
        "runtime": {
            "device": "cpu",
            "batch_size": 1,
            "torch_build": "controlled build",
            "num_threads": 1,
            "num_interop_threads": 1,
            "float32_matmul_precision": "highest",
            "deterministic_algorithms": False,
            "deterministic_warn_only": False,
            "mkldnn_enabled": True,
            "environment_variables": {
                name: None for name in _measurement.RUNTIME_ENVIRONMENT_VARIABLES
            },
        },
        "synchronization": "synchronous_cpu",
    }


def controlled_report(*, seconds: float = 2.0, schema_version: Literal[1, 2] = 2) -> Report:
    suite = Suite(
        name="controlled",
        split="dev",
        question=Choice(instructions="Select", criteria={"a": "Alpha", "b": "Beta"}),
        examples=[Example(id="0", text="input", label="a")],
    )
    result = DecisionResult(
        model=MODEL, answers={"intent": answer_from_scores(["a", "b"], [2.0, 1.0])}, input_tokens=1
    )
    return Report(
        schema_version=schema_version,
        created_at="2026-01-01T00:00:00Z",
        status="complete",
        suite=suite,
        suite_sha256=suite.sha256,
        protocol=Protocol(warmups=0, repeats=2),
        transport="local",
        model=MODEL,
        environment=recorded_environment(),
        observations=[
            Observation(
                id="0",
                attempts=[
                    Attempt(seconds=seconds, result=result),
                    Attempt(seconds=seconds * 2, result=result),
                ],
            )
        ],
        prediction_artifact=(
            PredictionArtifact(sha256=hashlib.sha256(b"").hexdigest(), bytes=0)
            if schema_version == 2
            else None
        ),
        summary={
            "accuracy": 1.0,
            "top5_accuracy": 1.0,
            "macro_f1": 0.5,
            "latency": {"calls": 2, "mean_seconds": seconds * 1.5},
        },
    )


def comparison(monkeypatch: pytest.MonkeyPatch, before: Report, after: Report) -> Comparison:
    def load(path: str | Path) -> Report:
        assert path in {"baseline", "candidate"}
        return before if path == "baseline" else after

    monkeypatch.setattr(_compare, "load_report", load)
    return _compare.compare("baseline", "candidate")


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Mock:
    torch = Mock()
    torch.__config__ = SimpleNamespace(show=lambda: "controlled build")
    torch.get_num_threads.return_value = 3
    torch.get_num_interop_threads.return_value = 2
    torch.get_float32_matmul_precision.return_value = "highest"
    torch.are_deterministic_algorithms_enabled.return_value = False
    torch.is_deterministic_algorithms_warn_only_enabled.return_value = False
    torch.backends.mkldnn.enabled = True
    torch.version = SimpleNamespace(cuda="12.0", hip=None)
    torch.cuda.get_device_properties.return_value = SimpleNamespace(
        name="Controlled GPU",
        total_memory=2**30,
        major=8,
        minor=0,
        multi_processor_count=32,
        uuid="controlled-gpu-uuid",
    )
    torch.backends.cudnn.version.return_value = 9000
    for name in (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
    ):
        getattr(torch.backends.cuda, name).return_value = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.mps.recommended_max_memory.return_value = 2**30
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(_measurement, "_cpu_model", lambda: "Controlled CPU")
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        _measurement, "_cpu_affinity", lambda: {"status": "recorded", "cpus": [0, 1]}
    )
    monkeypatch.setattr(_measurement, "_cuda_driver", lambda: "controlled driver")
    return torch


def mock_model(device: str = "cpu", batch_size: int = 4) -> Model:
    backend = Mock(spec=Model)
    backend.info = MODEL.model_copy(update={"device": device})
    backend._batch_size = batch_size
    return cast(Model, backend)


def test_request_transforms_preserve_other_native_timing_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    for report in (before, after):
        report.config["request_transform"] = "question-first-v1"
    matched = comparison(monkeypatch, before, after)
    assert matched["recipe_changed"] is False
    assert matched["mean_latency_speedup"] == 2.0
    assert matched["latency_comparison_exclusions"] == []

    after.config["request_transform"] = "identity"
    with pytest.raises(ValueError, match="recipe changed"):
        _compare.compare("baseline", "candidate")
    changed = _compare.compare("baseline", "candidate", allow_recipe_change=True)
    assert changed["recipe_changed"] is True
    assert changed["mean_latency_speedup"] is None
    assert changed["latency_comparison_exclusions"] == ["input recipe changed"]


def test_environment_without_a_local_model_does_not_import_inference() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from kayak.eval._measurement import environment, synchronize; "
            "environment(); synchronize(None); synchronize('cpu'); "
            "assert not {'torch', 'transformers', 'huggingface_hub'} & sys.modules.keys()",
        ],
        check=True,
        timeout=10,
    )


def test_local_environment_records_resolved_settings_and_thread_changes(fake_torch: Mock) -> None:
    backend = mock_model()
    first = _measurement.environment(backend)
    runtime = first["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["batch_size"] == 4
    assert runtime["num_threads"] == 3
    assert runtime["num_interop_threads"] == 2
    assert runtime["float32_matmul_precision"] == "highest"
    assert first["hardware"] == {
        "cpu_model": "Controlled CPU",
        "logical_cpus": 8,
        "cpu_affinity": {"status": "recorded", "cpus": [0, 1]},
        "accelerator": None,
    }
    assert first["synchronization"] == "synchronous_cpu"
    fake_torch.get_num_threads.return_value = 7
    second = _measurement.environment(backend)
    changed_runtime = second["runtime"]
    assert isinstance(changed_runtime, dict)
    assert changed_runtime["num_threads"] == 7
    assert runtime != changed_runtime


def test_cuda_properties_and_numerical_settings_come_from_the_resolved_device(
    fake_torch: Mock,
) -> None:
    environment = _measurement.environment(mock_model("cuda:2"))
    fake_torch.cuda.get_device_properties.assert_called_once_with("cuda:2")
    hardware, runtime = environment["hardware"], environment["runtime"]
    assert isinstance(hardware, dict) and isinstance(runtime, dict)
    assert hardware["accelerator"] == {
        "kind": "cuda",
        "name": "Controlled GPU",
        "total_memory_bytes": 2**30,
        "compute_capability": [8, 0],
        "multiprocessors": 32,
        "uuid": "controlled-gpu-uuid",
    }
    assert runtime["cuda"]["driver"] == "controlled driver"
    assert runtime["cuda"]["matmul_allow_tf32"] is False
    assert runtime["cuda"]["cudnn_benchmark"] is False
    assert environment["synchronization"] == "torch.cuda.synchronize"


def test_mps_inventory_is_bounded_and_drops_display_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "SPDisplaysDataType": [
            {
                "sppci_model": "Controlled MPS GPU",
                "spdisplays_cores": "16",
                "spdisplays_ndrvs": [{"_spdisplays_display-serial-number": "private-serial"}],
            }
        ]
    }
    command = Mock(return_value=SimpleNamespace(stdout=json.dumps(payload)))
    monkeypatch.setattr(subprocess, "run", command)
    assert _measurement._mps_devices() == [
        {"name": "Controlled MPS GPU", "cores": 16, "memory": None}
    ]
    command.assert_called_once_with(
        ["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )


@pytest.mark.parametrize("raw", [None, "not json", "{}", '{"SPDisplaysDataType": [1]}'])
def test_unknown_mps_inventory_stays_unknown(
    raw: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_measurement, "_command_output", lambda arguments: raw)
    assert _measurement._mps_devices() is None


def test_failed_hardware_query_stays_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    command = Mock(side_effect=subprocess.TimeoutExpired("controlled", 5))
    monkeypatch.setattr(subprocess, "run", command)
    assert _measurement._command_output(["controlled"]) is None


def test_cpu_identity_retains_distinct_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: "model name : CPU B\nmodel name : CPU A\nmodel name : CPU B\n",
    )
    assert _measurement._cpu_model() == "CPU A; CPU B"


def test_unknown_affinity_is_distinct_from_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", Mock(side_effect=OSError), raising=False)
    assert _measurement._cpu_affinity() == {"status": "unavailable", "cpus": None}
    monkeypatch.delattr(os, "sched_getaffinity")
    assert _measurement._cpu_affinity() == {"status": "unsupported", "cpus": None}


@pytest.mark.parametrize("device", ["cuda:2", "mps", "cpu", None])
def test_synchronization_targets_the_requested_device(fake_torch: Mock, device: str | None) -> None:
    _measurement.synchronize(device)
    if device == "cuda:2":
        fake_torch.cuda.synchronize.assert_called_once_with("cuda:2")
        fake_torch.mps.synchronize.assert_not_called()
    elif device == "mps":
        fake_torch.mps.synchronize.assert_called_once_with()
        fake_torch.cuda.synchronize.assert_not_called()
    else:
        fake_torch.cuda.synchronize.assert_not_called()
        fake_torch.mps.synchronize.assert_not_called()


def test_comparison_keeps_ratio_arithmetic_and_allows_disclosed_interventions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    after.environment["kayak_source_sha256"] = "b" * 64
    versions, runtime = after.environment["versions"], after.environment["runtime"]
    assert isinstance(versions, dict) and isinstance(runtime, dict)
    versions["kayak"] = "2.0"
    runtime["batch_size"] = 4
    after.model = MODEL.model_copy(update={"revision": "2", "dtype": "bfloat16"})
    before.config = {"batch_size": 1, "setup_seconds": 10.0}
    after.config = {"batch_size": 4, "setup_seconds": 20.0}
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] == 2.0  # mean(2, 4) / mean(1, 2)
    assert result["latency_comparison_exclusions"] == []
    changes = result["changed_factors"]
    assert isinstance(changes, list)
    assert [change["factor"] for change in changes] == [
        "source",
        "model",
        "precision",
        "batch_size",
    ]
    assert changes[-1] == {"factor": "batch_size", "baseline": 1, "candidate": 4}
    assert result["baseline_environment"] == before.environment
    assert result["candidate_environment"] == after.environment
    assert result["baseline_config"] == before.config
    assert result["candidate_config"] == after.config


@pytest.mark.parametrize(
    ("section", "field"),
    [
        (None, "host"),
        (None, "kayak_source_sha256"),
        (None, "synchronization"),
        ("versions", "torch"),
        ("hardware", "cpu_model"),
        ("hardware", "logical_cpus"),
        ("hardware", "cpu_affinity"),
        ("runtime", "device"),
        ("runtime", "batch_size"),
        ("runtime", "torch_build"),
        ("runtime", "num_threads"),
        ("runtime", "num_interop_threads"),
        ("runtime", "float32_matmul_precision"),
        ("runtime", "deterministic_algorithms"),
        ("runtime", "deterministic_warn_only"),
        ("runtime", "mkldnn_enabled"),
        ("runtime", "environment_variables"),
    ],
)
def test_matching_missing_provenance_never_enables_a_speedup(
    section: str | None, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    for report in (before, after):
        fields = report.environment[section] if section else report.environment
        assert isinstance(fields, dict)
        fields.pop(field)
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] is None
    assert "missing timing provenance" in str(result["latency_comparison_exclusions"])
    assert result["baseline_latency"] == {"calls": 2, "mean_seconds": 3.0}
    assert result["candidate_latency"] == {"calls": 2, "mean_seconds": 1.5}
    assert result["quality_delta"] == {
        "accuracy": 0.0,
        "top5_accuracy": 0.0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "weighted_f1": 0.0,
        "matthews_correlation": 0.0,
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "host", "another host"),
        (None, "synchronization", "custom"),
        ("versions", "torch", "2.0"),
        ("hardware", "cpu_model", "Another CPU"),
        ("hardware", "logical_cpus", 16),
        ("hardware", "cpu_affinity", {"status": "recorded", "cpus": [1]}),
        ("runtime", "num_threads", 2),
        ("runtime", "num_interop_threads", 2),
        ("runtime", "float32_matmul_precision", "medium"),
        ("runtime", "deterministic_algorithms", True),
        ("runtime", "mkldnn_enabled", False),
    ],
)
def test_changed_execution_conditions_withhold_only_the_ratio(
    section: str | None, field: str, value: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    fields = after.environment[section] if section else after.environment
    assert isinstance(fields, dict)
    fields[field] = value
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] is None
    assert result["latency_comparison_exclusions"]
    assert result["candidate_latency"] == after.summary["latency"]


@pytest.mark.parametrize("transport", ["http", "custom"])
def test_unknown_execution_boundaries_never_receive_speedups(
    transport: Literal["http", "custom"], monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    before.transport = after.transport = transport
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] is None
    assert result["latency_comparison_exclusions"]


def test_legacy_reports_keep_quality_comparison_without_timing_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = controlled_report(schema_version=1)
    result = comparison(monkeypatch, report, report)
    assert result["mean_latency_speedup"] is None
    assert "current report schema" in str(result["latency_comparison_exclusions"])
    assert result["quality_delta"] == {
        "accuracy": 0.0,
        "top5_accuracy": 0.0,
        "macro_f1": 0.0,
        "balanced_accuracy": 0.0,
        "weighted_f1": 0.0,
        "matthews_correlation": 0.0,
    }


@pytest.mark.parametrize("before_seconds,after_seconds", [(0.0, 1.0), (1.0, 0.0), (1e300, 1e-300)])
def test_zero_or_overflowing_ratios_are_not_reported(
    before_seconds: float, after_seconds: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = comparison(
        monkeypatch,
        controlled_report(seconds=before_seconds),
        controlled_report(seconds=after_seconds),
    )
    assert result["mean_latency_speedup"] is None
    assert result["latency_comparison_exclusions"]


def test_runner_discloses_custom_synchronization_on_a_local_model(
    fake_torch: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = Mock(spec=Model)
    backend.info = MODEL
    backend._batch_size = 1
    fixture = controlled_report()
    backend.decide.return_value = fixture.observations[0].attempts[0].result
    monkeypatch.setattr(_runner, "memory_snapshot", lambda device: {"process_peak_rss_bytes": 100})
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(_runner, "perf_counter", lambda: next(ticks))
    output = tmp_path / "custom-sync"
    report = evaluate(
        cast(Model, backend), fixture.suite, output=output, warmups=0, sync=lambda: None
    )
    assert report.status == "complete"
    assert report.observations[0].attempts[0].seconds == 2.0
    assert report.environment["synchronization"] == "custom"
    result = _compare.compare(output, output)
    assert result["mean_latency_speedup"] is None
    assert "known synchronization boundary" in str(result["latency_comparison_exclusions"])


@pytest.mark.parametrize("device", ["cuda:2", "mps"])
@pytest.mark.parametrize("missing_package", [None, "torch", "transformers", "tokenizers"])
def test_mocked_gpu_provenance_requires_known_package_versions(
    device: str,
    missing_package: str | None,
    fake_torch: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def package_version(name: str) -> str:
        if name == missing_package:
            raise PackageNotFoundError(name)
        return "1.0"

    # Mocking the imported torch module does not supply distribution metadata.
    monkeypatch.setattr(_measurement, "version", package_version)
    monkeypatch.setattr(
        _measurement,
        "_mps_devices",
        lambda: [{"name": "Controlled MPS GPU", "cores": 16, "memory": None}],
    )
    backend = mock_model(device)
    report = controlled_report()
    report.model = backend.info
    report.environment = _measurement.environment(backend)
    expected = [] if missing_package is None else ["package versions"]
    assert _compare._missing_local_provenance(report) == expected


@pytest.mark.parametrize("cores", [None, 0, -1, True, "16"])
def test_unknown_or_invalid_mps_core_counts_do_not_match(
    cores: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    for report in (before, after):
        report.model = MODEL.model_copy(update={"device": "mps"})
        hardware, runtime = report.environment["hardware"], report.environment["runtime"]
        assert isinstance(hardware, dict) and isinstance(runtime, dict)
        runtime["device"] = "mps"
        hardware["accelerator"] = {
            "kind": "mps",
            "recommended_max_memory_bytes": 2**30,
            "devices": [{"name": "Controlled MPS GPU", "cores": cores, "memory": None}],
        }
        report.environment["synchronization"] = "torch.mps.synchronize"
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] is None
    assert "MPS device properties" in str(result["latency_comparison_exclusions"])


@pytest.mark.parametrize("device", ["cuda:2", "mps"])
def test_gpu_provenance_is_required_even_when_both_reports_omit_it(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before, after = controlled_report(), controlled_report(seconds=1.0)
    for report in (before, after):
        report.model = MODEL.model_copy(update={"device": device})
        runtime = report.environment["runtime"]
        assert isinstance(runtime, dict)
        runtime["device"] = device
        report.environment["synchronization"] = f"torch.{device.split(':')[0]}.synchronize"
    result = comparison(monkeypatch, before, after)
    assert result["mean_latency_speedup"] is None
    assert "device properties" in str(result["latency_comparison_exclusions"])
