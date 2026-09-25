"""Compare models on identical examples without confusing repeats with new labels."""

import math
import statistics
from pathlib import Path
from typing import TypedDict

from ._comparison_cases import CaseComparison, compare_cases, markdown_cell, render_case_changes
from ._measurement import RUNTIME_ENVIRONMENT_VARIABLES
from ._metrics import summarize
from ._runner import load_report
from ._schema import Report


class Comparison(TypedDict):
    """Verified native-run comparison; case records use first measured predictions."""

    baseline_model: dict[str, object] | None
    candidate_model: dict[str, object] | None
    examples: int
    recipe_changed: bool
    changed_factors: list[dict[str, object]]
    baseline_environment: dict[str, object]
    candidate_environment: dict[str, object]
    baseline_config: dict[str, object]
    candidate_config: dict[str, object]
    baseline_request_transform: object
    candidate_request_transform: object
    quality_delta: dict[str, float]
    paired_outcomes: dict[str, int]
    cases: list[CaseComparison]
    baseline_latency: object
    candidate_latency: object
    mean_latency_speedup: float | None
    latency_comparison_exclusions: list[str]
    interpretation: str


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _missing_local_provenance(report: Report) -> list[str]:
    """Unknown values must not become matching conditions just because both are absent."""
    missing = []
    if report.schema_version < 2:
        missing.append("current report schema")
    for field in ("platform", "machine", "host", "python"):
        if not isinstance(report.environment.get(field), str) or not report.environment[field]:
            missing.append(field)
    source = report.environment.get("kayak_source_sha256")
    if (
        not isinstance(source, str)
        or len(source) != 64
        or any(character not in "0123456789abcdef" for character in source)
    ):
        missing.append("source hash")
    versions = report.environment.get("versions")
    if not isinstance(versions, dict) or any(
        not isinstance(versions.get(name), str) or not versions[name]
        for name in ("kayak", "torch", "transformers", "tokenizers", "pydantic", "httpx")
    ):
        missing.append("package versions")
    hardware = report.environment.get("hardware")
    if not isinstance(hardware, dict):
        hardware = {}
    if not isinstance(hardware.get("cpu_model"), str) or not hardware["cpu_model"]:
        missing.append("CPU model")
    if not _positive_integer(hardware.get("logical_cpus")):
        missing.append("logical CPU count")
    affinity = hardware.get("cpu_affinity")
    if not isinstance(affinity, dict) or not (
        affinity.get("status") == "unsupported"
        or (
            affinity.get("status") == "recorded"
            and isinstance(affinity.get("cpus"), list)
            and affinity["cpus"]
            and all(type(cpu) is int and cpu >= 0 for cpu in affinity["cpus"])
        )
    ):
        missing.append("CPU affinity")
    runtime = report.environment.get("runtime")
    if not isinstance(runtime, dict):
        runtime = {}
    for field in ("batch_size", "num_threads", "num_interop_threads"):
        if not _positive_integer(runtime.get(field)):
            missing.append(field)
    for field in ("torch_build", "float32_matmul_precision"):
        if not isinstance(runtime.get(field), str) or not runtime[field]:
            missing.append(field)
    for field in ("deterministic_algorithms", "deterministic_warn_only", "mkldnn_enabled"):
        if type(runtime.get(field)) is not bool:
            missing.append(field)
    variables = runtime.get("environment_variables")
    if not isinstance(variables, dict) or any(
        name not in variables
        or (variables[name] is not None and not isinstance(variables[name], str))
        for name in RUNTIME_ENVIRONMENT_VARIABLES
    ):
        missing.append("runtime environment variables")
    device = report.model.device if report.model else None
    if device is None or runtime.get("device") != device:
        missing.append("resolved execution device")
    kind = device.split(":", 1)[0] if device else None
    synchronization = {
        "cpu": "synchronous_cpu",
        "cuda": "torch.cuda.synchronize",
        "mps": "torch.mps.synchronize",
    }.get(kind or "")
    if synchronization is None or report.environment.get("synchronization") != synchronization:
        missing.append("known synchronization boundary")
    accelerator = hardware.get("accelerator")
    if kind == "cuda":
        if not isinstance(accelerator, dict) or not (
            accelerator.get("kind") == "cuda"
            and isinstance(accelerator.get("name"), str)
            and accelerator["name"]
            and _positive_integer(accelerator.get("total_memory_bytes"))
            and _positive_integer(accelerator.get("multiprocessors"))
            and isinstance(accelerator.get("compute_capability"), list)
            and len(accelerator["compute_capability"]) == 2
            and all(type(value) is int for value in accelerator["compute_capability"])
        ):
            missing.append("CUDA device properties")
        cuda = runtime.get("cuda")
        if not isinstance(cuda, dict):
            cuda = {}
        if not isinstance(cuda.get("driver"), str) or not cuda["driver"]:
            missing.append("CUDA driver")
        if not cuda.get("cuda_version") and not cuda.get("hip_version"):
            missing.append("CUDA/HIP version")
        if "cudnn_version" not in cuda:
            missing.append("cuDNN version")
        for field in (
            "matmul_allow_tf32",
            "matmul_allow_fp16_reduced_precision_reduction",
            "matmul_allow_bf16_reduced_precision_reduction",
            "flash_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
            "cudnn_sdp_enabled",
            "cudnn_enabled",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cudnn_allow_tf32",
        ):
            if type(cuda.get(field)) is not bool:
                missing.append(field)
    elif kind == "mps":
        devices = accelerator.get("devices") if isinstance(accelerator, dict) else None
        if not (
            isinstance(accelerator, dict)
            and accelerator.get("kind") == "mps"
            and _positive_integer(accelerator.get("recommended_max_memory_bytes"))
            and isinstance(devices, list)
            and len(devices) == 1
            and isinstance(devices[0], dict)
            and isinstance(devices[0].get("name"), str)
            and devices[0]["name"]
            and _positive_integer(devices[0].get("cores"))
        ):
            missing.append("unambiguous MPS device properties")
    return missing


def compare(
    baseline: str | Path, candidate: str | Path, *, allow_recipe_change: bool = False
) -> Comparison:
    """Return descriptive quality/latency deltas, never an automatic promotion."""
    before, after = load_report(baseline), load_report(candidate)
    if before.status != "complete" or after.status != "complete":
        raise ValueError("comparison requires two complete runs without execution failures")
    for left, right in (
        (before.suite.name, after.suite.name),
        (before.suite.split, after.suite.split),
        (before.suite.provenance, after.suite.provenance),
        (before.suite.examples, after.suite.examples),
        (list(before.suite.question.criteria), list(after.suite.question.criteria)),
    ):
        if left != right:
            raise ValueError("comparison requires identical dataset, examples, labels and order")
    before_transform = before.config.get("request_transform", "identity")
    after_transform = after.config.get("request_transform", "identity")
    recipe_changed = (
        before.suite.question != after.suite.question or before_transform != after_transform
    )
    if recipe_changed and not allow_recipe_change:
        raise ValueError("input recipe changed; use allow_recipe_change for an explicit experiment")
    # Source, model, precision, and batching may be intentional interventions.
    # Expose those factors while requiring known, matching execution conditions.
    timing_reasons = []
    if recipe_changed:
        timing_reasons.append("input recipe changed")
    if before.transport != after.transport:
        timing_reasons.append("transport changed")
    if before.transport == "http" or after.transport == "http":
        timing_reasons.append("server environment is not recorded by the HTTP contract")
    if before.transport == "custom" or after.transport == "custom":
        timing_reasons.append("custom backend execution and synchronization are not verified")
    if before.model and after.model and before.model.device != after.model.device:
        timing_reasons.append("execution device changed")
    if before.protocol != after.protocol:
        timing_reasons.append("warmups, repeats or execution order changed")
    for name, report in (("baseline", before), ("candidate", after)):
        if report.transport == "local":
            missing = _missing_local_provenance(report)
            if missing:
                timing_reasons.append(f"{name} missing timing provenance: {', '.join(missing)}")
    for field in ("platform", "machine", "host", "python", "hardware", "synchronization"):
        if before.environment.get(field) != after.environment.get(field):
            timing_reasons.append(f"environment {field} changed")
    before_versions = before.environment.get("versions")
    after_versions = after.environment.get("versions")
    before_versions = before_versions if isinstance(before_versions, dict) else {}
    after_versions = after_versions if isinstance(after_versions, dict) else {}
    if {key: value for key, value in before_versions.items() if key != "kayak"} != {
        key: value for key, value in after_versions.items() if key != "kayak"
    }:
        timing_reasons.append("environment dependency versions changed")
    before_runtime = before.environment.get("runtime")
    after_runtime = after.environment.get("runtime")
    before_runtime = before_runtime if isinstance(before_runtime, dict) else {}
    after_runtime = after_runtime if isinstance(after_runtime, dict) else {}
    if {key: value for key, value in before_runtime.items() if key != "batch_size"} != {
        key: value for key, value in after_runtime.items() if key != "batch_size"
    }:
        timing_reasons.append("runtime settings changed")
    changed_factors = []
    for factor, baseline_factor, candidate_factor in (
        (
            "source",
            {
                "sha256": before.environment.get("kayak_source_sha256"),
                "version": before_versions.get("kayak"),
            },
            {
                "sha256": after.environment.get("kayak_source_sha256"),
                "version": after_versions.get("kayak"),
            },
        ),
        (
            "model",
            before.model.model_dump(exclude={"device", "dtype"}) if before.model else None,
            after.model.model_dump(exclude={"device", "dtype"}) if after.model else None,
        ),
        (
            "precision",
            before.model.dtype if before.model else None,
            after.model.dtype if after.model else None,
        ),
        ("batch_size", before_runtime.get("batch_size"), after_runtime.get("batch_size")),
    ):
        if baseline_factor != candidate_factor:
            changed_factors.append(
                {"factor": factor, "baseline": baseline_factor, "candidate": candidate_factor}
            )
    deltas: dict[str, float] = {}
    before_summary, after_summary = summarize(before), summarize(after)
    for name in (
        "accuracy",
        "top5_accuracy",
        "macro_f1",
        "balanced_accuracy",
        "weighted_f1",
        "matthews_correlation",
    ):
        left_value, right_value = before_summary[name], after_summary[name]
        assert isinstance(left_value, float) and isinstance(right_value, float)
        deltas[name] = right_value - left_value
    choices = []
    for report in (before, after):
        predictions: dict[str, str] = {}
        for row in report.observations:
            result = row.attempts[0].result
            assert result is not None
            predictions[row.id] = result.answers["intent"].choice
        choices.append(predictions)
    paired_outcomes, cases = compare_cases(before.suite.examples, choices[0], choices[1])
    left_times: list[float] = []
    right_times: list[float] = []
    for row in before.observations:
        left_times.extend(attempt.seconds for attempt in row.attempts)
    for row in after.observations:
        right_times.extend(attempt.seconds for attempt in row.attempts)
    speedup = None
    if not timing_reasons:
        left_mean = statistics.mean(left_times)
        right_mean = statistics.mean(right_times)
        if left_mean <= 0 or right_mean <= 0:
            timing_reasons.append("latency means must be positive to compute a speedup")
        else:
            ratio = left_mean / right_mean
            if math.isfinite(ratio) and ratio > 0:
                speedup = ratio
            else:
                timing_reasons.append("latency ratio is outside the representable range")
    return {
        "baseline_model": before.model.model_dump() if before.model else None,
        "candidate_model": after.model.model_dump() if after.model else None,
        "examples": len(before.suite.examples),
        "recipe_changed": recipe_changed,
        "changed_factors": changed_factors,
        "baseline_environment": before.environment,
        "candidate_environment": after.environment,
        "baseline_config": before.config,
        "candidate_config": after.config,
        "baseline_request_transform": before_transform,
        "candidate_request_transform": after_transform,
        "quality_delta": deltas,
        "paired_outcomes": paired_outcomes,
        "cases": cases,
        "baseline_latency": before.summary["latency"],
        "candidate_latency": after.summary["latency"],
        "mean_latency_speedup": speedup,
        "latency_comparison_exclusions": timing_reasons,
        "interpretation": (
            "Descriptive single-run comparison of the disclosed configurations. "
            "Changed factors may jointly explain timing differences; host load and thermal "
            "state remain uncontrolled. No significance or promotion claim."
        ),
    }


def render_comparison(result: Comparison) -> str:
    """Render an existing comparison; never read files or rerun a backend."""
    counts = result["paired_outcomes"]
    lines = [
        "# Classification comparison",
        "",
        result["interpretation"],
        "",
        f"Examples: {result['examples']}. Fixed: {counts['fixed']}. "
        f"Regressed: {counts['regressed']}. Still wrong: {counts['both_wrong']}.",
        "",
    ]
    for role in ("baseline", "candidate"):
        model = result["baseline_model"] if role == "baseline" else result["candidate_model"]
        config = result["baseline_config"] if role == "baseline" else result["candidate_config"]
        lines.append(
            f"{role.title()} model: {markdown_cell(model['id'] if model else 'unknown')}; "
            f"evidence: {markdown_cell(config.get('evidence_kind', 'unspecified'))}."
        )
    lines.extend(["", "| Metric | Candidate minus baseline |", "| --- | ---: |"])
    for metric, delta in result["quality_delta"].items():
        lines.append(f"| {markdown_cell(metric)} | {delta:+.6f} |")
    lines.append("")
    if result["changed_factors"]:
        lines.extend(["| Changed factor | Baseline | Candidate |", "| --- | --- | --- |"])
        for factor in result["changed_factors"]:
            lines.append(
                "| "
                + " | ".join(
                    markdown_cell(factor[key]) for key in ("factor", "baseline", "candidate")
                )
                + " |"
            )
        lines.append("")
    if result["recipe_changed"]:
        lines.extend(
            [
                "The question or declared method differs; "
                "recipe comparison was explicitly allowed.",
                "",
            ]
        )
    speedup = result["mean_latency_speedup"]
    if speedup is not None:
        lines.extend([f"Recorded mean latency speedup: {speedup:.6f}.", ""])
    else:
        lines.extend(
            [
                "Latency comparison unavailable: "
                + "; ".join(map(markdown_cell, result["latency_comparison_exclusions"]))
                + ".",
                "",
            ]
        )
    lines.append(render_case_changes(result["cases"]))
    return "\n".join(lines)
