"""Read-only multi-system analysis and portable report bundles."""

import csv
import hashlib
import io
import json
import math
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import NotRequired, TypedDict

from ._compare import compare
from ._comparison_cases import CaseComparison, compare_cases, render_case_changes
from ._comparison_cases import markdown_cell as _cell
from ._predictions import PredictionSet, predictions_from_report
from ._runner import load_report
from ._scoring import (
    Metric,
    MetricInput,
    MetricResult,
    classification,
    default_metrics,
    metric_input,
    score_metrics,
)


class BenchmarkComparison(TypedDict):
    """Case evidence exists only when the selected runs are comparable and complete."""

    baseline: str
    candidate: str
    eligible: bool
    exclusions: list[str]
    recipe_changed: bool
    paired_outcomes: NotRequired[dict[str, int]]
    cases: NotRequired[list[CaseComparison]]
    accuracy_delta: NotRequired[float]
    metric_delta: NotRequired[dict[str, float | None]]
    mcnemar_exact_p: NotRequired[float]
    mcnemar_bonferroni_p: NotRequired[float]
    comparison_family_size: NotRequired[int]
    mean_latency_speedup: NotRequired[float | None]
    latency_comparison_exclusions: NotRequired[list[str]]


class BenchmarkRun(TypedDict):
    system: str
    method: str
    metadata: dict[str, str]
    source: dict[str, object]
    suite_sha256: str
    dataset: str
    split: str
    provenance: dict[str, str]
    examples: int
    labels: int
    answered_examples: int
    metrics: dict[str, MetricResult]
    classification: dict[str, object]
    accuracy_interval: dict[str, float | str] | None


class BenchmarkResult(TypedDict):
    """JSON-compatible analysis; custom metric names remain caller-defined."""

    schema_version: int
    created_at: str
    python: str
    analysis_source_sha256: dict[str, str]
    runs: dict[str, BenchmarkRun]
    comparisons: list[BenchmarkComparison]
    interpretation: str


@dataclass
class _Input:
    predictions: PredictionSet
    data: MetricInput
    source: dict[str, object]
    native_path: Path | None = None

    @property
    def complete(self) -> bool:
        # Exporting first attempts cannot erase a native warmup/repeat failure.
        # Ordinary external predictions have no native execution status to check.
        return self.predictions.metadata.get("original_status", "complete") == "complete"


def _hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _read(source: str | Path | PredictionSet) -> _Input:
    if isinstance(source, PredictionSet):
        predictions = PredictionSet.model_validate(source.model_dump())
        return _Input(
            predictions,
            metric_input(predictions),
            {
                "kind": "external_predictions",
                "evidence_kind": predictions.metadata.get("evidence_kind", "unspecified"),
                "verification": "validated caller-supplied predictions",
                "sha256": hashlib.sha256(predictions.model_dump_json().encode()).hexdigest(),
            },
        )
    path = Path(source)
    metadata = path / "report.json" if path.is_dir() else path
    payload = json.loads(metadata.read_bytes())
    if isinstance(payload, dict) and "suite_sha256" in payload:
        files = [metadata, metadata.parent / "predictions.jsonl"]
        before = _hashes(files)
        report = load_report(metadata)
        after = _hashes(files)
        if before != after:
            raise ValueError("input artifacts changed while reading")
        predictions = predictions_from_report(
            report,
            system=report.model.id if report.model else "unknown",
            method=str(report.config.get("request_transform", "identity")),
        )
        return _Input(
            predictions,
            metric_input(predictions),
            {
                "kind": "kayak_run",
                "path": str(metadata),
                "sha256": before,
                "schema_version": report.schema_version,
                "status": report.status,
                "evidence_kind": report.config.get("evidence_kind", "unspecified"),
                "verification": "structural only"
                if report.schema_version == 1
                else "prediction byte digest",
                "model": report.model.model_dump() if report.model else None,
                "config": report.config,
            },
            metadata,
        )
    raw = metadata.read_bytes()
    predictions = PredictionSet.model_validate_json(raw)
    return _Input(
        predictions,
        metric_input(predictions),
        {
            "kind": "external_predictions",
            "evidence_kind": predictions.metadata.get("evidence_kind", "unspecified"),
            "path": str(metadata),
            "verification": "validated caller-supplied predictions; execution not verified",
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
    )


def accuracy_interval(correct: int, total: int) -> dict[str, float | str]:
    """Wilson 95% interval, conditional on independent representative examples."""
    if type(correct) is not int or type(total) is not int or total < 1 or not 0 <= correct <= total:
        raise ValueError(
            "accuracy interval requires integer counts, 0 <= correct <= total, total > 0"
        )
    z = NormalDist().inv_cdf(0.975)
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "method": "Wilson",
        "confidence": 0.95,
        # The algebraic endpoints are exact; subtracting rounded terms need not be.
        "lower": 0.0 if correct == 0 else max(0.0, center - radius),
        "upper": 1.0 if correct == total else min(1.0, center + radius),
    }


def mcnemar_exact(fixed: int, regressed: int) -> float:
    """Two-sided exact binomial McNemar p-value over discordant pairs."""
    if type(fixed) is not int or type(regressed) is not int or min(fixed, regressed) < 0:
        raise ValueError("McNemar counts must be nonnegative integers")
    n = fixed + regressed
    if n == 0 or fixed == regressed:
        return 1.0
    k = min(fixed, regressed)
    term = 1
    tail = 1
    for index in range(1, k + 1):
        term = term * (n - index + 1) // index
        tail += term
    return min(1.0, (2 * tail) / (1 << n))


def _pair(
    before: _Input, after: _Input, *, baseline: str, candidate: str, allow_recipe_change: bool
) -> BenchmarkComparison:
    left, right = before.predictions, after.predictions
    reasons = []
    for field in ("name", "split", "provenance", "examples"):
        if getattr(left.suite, field) != getattr(right.suite, field):
            reasons.append(f"suite {field} differs")
    if before.data.labels != after.data.labels:
        reasons.append("candidate labels or order differ")
    recipe_changed = left.suite.question != right.suite.question or left.method != right.method
    if recipe_changed and not allow_recipe_change:
        reasons.append("question or method differs; explicitly allow a recipe change to compare")
    if not before.complete or not after.complete:
        reasons.append("native run is not complete")
    if any(row.choice is None for data in (before.data, after.data) for row in data.samples):
        reasons.append("comparison requires predictions for every selected example")
    result: BenchmarkComparison = {
        "baseline": baseline,
        "candidate": candidate,
        "eligible": not reasons,
        "exclusions": reasons,
        "recipe_changed": recipe_changed,
    }
    if reasons:
        return result
    outcomes, cases = compare_cases(
        left.suite.examples,
        {row.id: row.choice for row in before.data.samples if row.choice is not None},
        {row.id: row.choice for row in after.data.samples if row.choice is not None},
    )
    result["paired_outcomes"] = outcomes
    result["cases"] = cases
    result["mcnemar_exact_p"] = mcnemar_exact(outcomes["fixed"], outcomes["regressed"])
    result["accuracy_delta"] = (outcomes["fixed"] - outcomes["regressed"]) / len(
        before.data.samples
    )
    result["mean_latency_speedup"] = None
    result["latency_comparison_exclusions"] = [
        "external predictions do not establish inference timing"
    ]
    if before.native_path is not None and after.native_path is not None:
        timing = compare(
            before.native_path, after.native_path, allow_recipe_change=allow_recipe_change
        )
        result["mean_latency_speedup"] = timing["mean_latency_speedup"]
        result["latency_comparison_exclusions"] = timing["latency_comparison_exclusions"]
    return result


def benchmark(
    runs: Mapping[str, str | Path | PredictionSet],
    *,
    metrics: Sequence[Metric] | None = None,
    allow_recipe_change: bool = False,
) -> BenchmarkResult:
    """Analyze one or more systems; compare each to the first. No inference or writes.

    Custom metric functions execute only when explicitly supplied by Python callers.
    They receive immutable first-attempt data and may not return nonfinite values.
    """
    if not runs or any(not name.strip() for name in runs):
        raise ValueError("benchmark needs at least one nonblank run name")
    selected = tuple(default_metrics() if metrics is None else metrics)
    if not selected or len({metric.name for metric in selected}) != len(selected):
        raise ValueError("select at least one metric, with unique names")
    inputs = {name: _read(source) for name, source in runs.items()}
    rows: dict[str, BenchmarkRun] = {}
    scores: dict[str, dict[str, MetricResult]] = {}
    for name, item in inputs.items():
        summary = classification(item.data)
        scores[name] = score_metrics(item.data, selected)
        correct = summary["correct"]
        assert isinstance(correct, int)
        rows[name] = {
            "system": item.predictions.system,
            "method": item.predictions.method,
            "metadata": item.predictions.metadata,
            "source": item.source,
            "suite_sha256": item.predictions.suite.sha256,
            "dataset": item.predictions.suite.name,
            "split": item.predictions.suite.split,
            "provenance": item.predictions.suite.provenance,
            "examples": len(item.data.samples),
            "labels": len(item.data.labels),
            "answered_examples": sum(row.choice is not None for row in item.data.samples),
            "metrics": scores[name],
            "classification": summary,
            "accuracy_interval": accuracy_interval(correct, len(item.data.samples))
            if item.complete and all(row.choice is not None for row in item.data.samples)
            else None,
        }
    baseline = next(iter(inputs))
    comparisons: list[BenchmarkComparison] = []
    for name, item in inputs.items():
        if name == baseline:
            continue
        pair = _pair(
            inputs[baseline],
            item,
            baseline=baseline,
            candidate=name,
            allow_recipe_change=allow_recipe_change,
        )
        if pair["eligible"]:
            deltas: dict[str, float | None] = {}
            for metric in selected:
                a, b = scores[baseline][metric.name]["value"], scores[name][metric.name]["value"]
                deltas[metric.name] = (
                    b - a if isinstance(a, float) and isinstance(b, float) else None
                )
            pair["metric_delta"] = deltas
        comparisons.append(pair)
    tests = sum(pair["eligible"] is True for pair in comparisons)
    for pair in comparisons:
        p = pair.get("mcnemar_exact_p")
        if isinstance(p, float):
            pair["mcnemar_bonferroni_p"] = min(1.0, p * tests)
            pair["comparison_family_size"] = tests
    # A custom metric may perform I/O. Detect changes to file inputs before
    # returning a report that binds results to their original artifact hashes.
    for item in inputs.values():
        source_path = item.source.get("path")
        fingerprint = item.source["sha256"]
        if isinstance(fingerprint, dict):
            if _hashes([Path(path) for path in fingerprint]) != fingerprint:
                raise ValueError("input artifacts changed during analysis")
        elif isinstance(source_path, str):
            if hashlib.sha256(Path(source_path).read_bytes()).hexdigest() != fingerprint:
                raise ValueError("input artifacts changed during analysis")
    source_files = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "analysis_source_sha256": _hashes(source_files),
        "runs": rows,
        "comparisons": comparisons,
        "interpretation": (
            "Descriptive first-attempt analysis. "
            "Probability and ranking metrics require full coverage. "
            "Input hashes detect changes, not authentic execution or training-data independence. "
            "External predictions and custom metric versions are caller declarations. "
            "Wilson intervals and McNemar tests assume independent representative examples; "
            "repeated "
            "calls are not extra samples. Bonferroni covers only the eligible baseline comparisons "
            "in this report, not prior development tuning. No automatic significance, leaderboard, "
            "calibration, or production-quality claim. "
            "Lower-is-better metrics improve when deltas are negative."
        ),
    }


def render_benchmark(result: Mapping[str, object]) -> str:
    """Markdown view of the same analysis used for JSON/CSV."""
    runs, comparisons = result["runs"], result["comparisons"]
    assert isinstance(runs, dict) and isinstance(comparisons, list)
    lines = [
        "# Classification benchmark",
        "",
        str(result["interpretation"]),
        "",
        "| Run | System | Method | Dataset / split | Answered / selected | Evidence |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for name, row in runs.items():
        source = row["source"]
        evidence = (
            f"{source['kind']}; {source.get('evidence_kind', 'caller supplied')}; "
            f"{source.get('status', 'execution unknown')}; {source['verification']}"
        )
        lines.append(
            "| "
            + " | ".join(
                _cell(cell)
                for cell in (
                    name,
                    row["system"],
                    row["method"],
                    f"{row['dataset']} / {row['split']}",
                    f"{row['answered_examples']} / {row['examples']}",
                    evidence,
                )
            )
            + " |"
        )
    first = next(iter(runs.values()))
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Better | " + " | ".join(map(_cell, runs)) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in runs) + " |",
        ]
    )
    for key, definition in first["metrics"].items():
        values = []
        for row in runs.values():
            score = row["metrics"][key]
            values.append("unavailable" if score["value"] is None else f"{score['value']:.6f}")
        lines.append(f"| {_cell(key)} | {definition['direction']} | " + " | ".join(values) + " |")
    lines.append("")
    for key, definition in first["metrics"].items():
        if definition["parameters"]:
            lines.extend([f"{_cell(key)} settings: {_cell(definition['parameters'])}.", ""])
    for name, row in runs.items():
        unavailable = [key for key, score in row["metrics"].items() if score["value"] is None]
        if unavailable:
            lines.extend(
                [
                    f"{_cell(name)} unavailable metrics: "
                    + ", ".join(map(_cell, unavailable))
                    + ". Coverage and reasons are recorded in the JSON report.",
                    "",
                ]
            )
        interval = row["accuracy_interval"]
        if interval:
            lines.extend(
                [
                    f"{_cell(name)} accuracy Wilson 95% interval: "
                    f"[{interval['lower']:.6f}, {interval['upper']:.6f}].",
                    "",
                ]
            )
    lines.extend(["", "## Comparisons to the first run", ""])
    for pair in comparisons:
        lines.append(f"### {_cell(pair['candidate'])} vs {_cell(pair['baseline'])}")
        lines.append("")
        if not pair["eligible"]:
            lines.append(
                "Comparison unavailable: " + "; ".join(map(_cell, pair["exclusions"])) + "."
            )
        else:
            counts = pair["paired_outcomes"]
            lines.append(
                f"Fixed {counts['fixed']}; regressed {counts['regressed']}; "
                f"both correct {counts['both_correct']}; both wrong {counts['both_wrong']}."
            )
            lines.extend(
                [
                    "",
                    f"Accuracy change: {pair['accuracy_delta']:+.6f}. "
                    f"Exact McNemar p: {pair['mcnemar_exact_p']:.6g}; "
                    f"Bonferroni-adjusted p: {pair['mcnemar_bonferroni_p']:.6g} "
                    f"({pair['comparison_family_size']} comparisons).",
                    "",
                ]
            )
            if pair["recipe_changed"]:
                lines.extend(
                    [
                        "Question or declared method differs; this is an explicitly allowed "
                        "recipe comparison.",
                        "",
                    ]
                )
            if pair["mean_latency_speedup"] is not None:
                lines.append(f"Recorded mean latency speedup: {pair['mean_latency_speedup']:.6f}.")
            else:
                lines.append(
                    "Latency comparison unavailable: "
                    + "; ".join(map(_cell, pair["latency_comparison_exclusions"]))
                    + "."
                )
        lines.append("")
        if pair["eligible"]:
            if "cases" in pair:
                lines.append(render_case_changes(pair["cases"]))
            else:
                lines.extend(
                    [
                        "Case details unavailable in this saved report. "
                        "Re-run benchmark analysis on the original predictions to inspect cases.",
                        "",
                    ]
                )
    lines.extend(
        [
            "Metric definitions, versions, parameters, per-class scores, confusion matrices,",
            "source hashes, and comparison exclusions are retained in `benchmark.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark(result: Mapping[str, object], output: str | Path) -> None:
    """Write JSON, Markdown, and a long-form metric CSV to a new directory."""
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    markdown = render_benchmark(result)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("run", "metric", "value", "direction", "available_examples", "total_examples"))
    runs = result["runs"]
    assert isinstance(runs, dict)
    for name, row in runs.items():
        # Avoid formula execution when a user opens arbitrary system names in a spreadsheet.
        csv_name = "'" + name if name.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else name
        for key, score in row["metrics"].items():
            writer.writerow(
                (
                    csv_name,
                    key,
                    score["value"],
                    score["direction"],
                    score["available_examples"],
                    score["total_examples"],
                )
            )
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for filename, content in (
        ("benchmark.json", encoded),
        ("benchmark.md", markdown),
        ("metrics.csv", stream.getvalue()),
    ):
        (root / filename).write_text(content, encoding="utf-8")
