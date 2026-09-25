"""Generate a Markdown account from checked saved observations, without inference."""

import json
from pathlib import Path

from ._metrics import summarize
from ._runner import load_report


def _number(value: object) -> str:
    if value is None:
        return "unavailable"
    if type(value) is int:
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    raise ValueError("expected a numeric summary value")


def render_report(path: str | Path) -> str:
    """Read, verify, and render one run; never turn mock outputs into model evidence."""
    report = load_report(path)
    summary = summarize(report)
    lines = ["# Kayak evaluation report", ""]
    if report.config.get("evidence_kind") == "mock":
        lines.extend(
            [
                "**MOCK EVIDENCE — model quality and inference performance are not measured.**",
                "The choices are controlled fixtures. Durations measure evaluator control flow.",
                "",
            ]
        )
    elif report.transport == "custom":
        lines.extend(
            [
                "**CUSTOM BACKEND — execution location and synchronization are not verified.**",
                "These results describe the supplied outputs and the caller's elapsed time.",
                "",
            ]
        )
    lines.append(f"**Run status:** `{report.status}`.")
    if report.status != "complete":
        lines.append(
            "This is not an accepted completed benchmark; retain its failures and missing cases."
        )
    else:
        lines.append(
            "Completion means every declared call succeeded; "
            "it does not mean every answer was correct."
        )
    lines.append("")
    if report.schema_version == 1:
        lines.append(
            "**Legacy artifact:** structural checks passed, "
            "but prediction bytes have no recorded digest."
        )
    elif report.status == "running":
        lines.append(
            "The checkpoint's prediction prefix passed its byte-length and SHA-256 checks."
        )
        lines.append("Later rows may be recovered, but are not bound by that checkpoint.")
    else:
        lines.append(
            "The complete prediction file passed its recorded byte-length and SHA-256 checks."
        )
    lines.extend(
        [
            "These checks detect inconsistent or altered artifacts; "
            "they do not authenticate execution",
            "or prevent someone from rewriting observations and their hashes together.",
            "",
            "## Outcomes against the supplied labels",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
    )
    for title, key in (
        ("Selected examples (quality denominator)", "examples"),
        ("Attempted examples", "attempted_examples"),
        ("Correct first attempts", "correct"),
        ("Failed or missing first attempts", "failed_or_missing_examples"),
        ("Accuracy", "accuracy"),
        (f"Top-{min(5, len(report.suite.question.criteria))} accuracy", "top5_accuracy"),
        ("Macro F1 over all candidate labels", "macro_f1"),
        ("Balanced accuracy over supported labels", "balanced_accuracy"),
        ("Support-weighted F1", "weighted_f1"),
        ("Matthews correlation coefficient (MCC)", "matthews_correlation"),
        ("Distinct predicted intents", "predicted_label_count"),
        ("Largest intent prediction share of selected examples", "largest_prediction_share"),
        ("Uniform random accuracy (expected reference)", "uniform_chance_accuracy"),
        ("Always-majority accuracy (label-distribution reference)", "majority_class_accuracy"),
        ("Failed measured calls, including repeats", "failed_calls"),
        ("Failed warmups", "failed_warmups"),
        ("Examples with changed choices across repeats", "unstable_examples"),
        ("Maximum score change from first successful attempt", "max_repeated_score_drift"),
    ):
        lines.append(f"| {title} | {_number(summary[key])} |")
    zero_recall = summary["zero_recall_labels"]
    assert isinstance(zero_recall, list)
    lines.append(f"| Supported intents with zero recall | {len(zero_recall)} |")
    lines.extend(
        [
            "",
            "Quality uses the first measured attempt for each selected example. Failed and missing",
            "first attempts stay in the denominator. "
            "Repeats do not add independent labeled examples.",
            "Zero-support candidate labels remain in macro F1. "
            "Top-k ties preserve candidate order.",
            "Balanced accuracy averages recall only over labels with gold examples; weighted F1",
            "weights each label by its gold support. "
            "Undefined class scores and degenerate MCC are zero.",
            "MCC includes failures as an extra predicted category with zero gold support.",
            "Prediction diversity excludes failures; its largest share uses all selected examples.",
            "Chance and majority references are computed from this suite, not measured model runs.",
            "On a balanced suite, balanced accuracy equals accuracy "
            "and weighted F1 equals macro F1",
            "when every candidate has support. "
            "These aggregates do not measure confidence calibration.",
            "Metrics are recalculated by the current evaluator. "
            "Execution metadata below describes the original run.",
            "",
            "## Observed call durations",
            "",
            "| Calls | Count | Mean seconds | Median seconds | "
            "p95 seconds | Min seconds | Max seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for title, key in (
        ("Successful measured calls", "latency"),
        ("All recorded measured attempts", "attempt_latency"),
        ("Warmups (excluded from measured calls)", "warmup_latency"),
    ):
        values = summary[key]
        if not isinstance(values, dict):
            raise ValueError("expected a latency distribution")
        cells = [title]
        for field in (
            "calls",
            "mean_seconds",
            "median_seconds",
            "p95_seconds",
            "min_seconds",
            "max_seconds",
        ):
            cells.append(_number(values[field]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if report.schema_version == 1:
        lines.append(
            "Legacy timing boundaries depend on the recorded evaluator source; "
            "they may include input preparation or omit synchronization after failures."
        )
    elif report.transport == "local" and report.environment.get("synchronization") in {
        "synchronous_cpu",
        "torch.cuda.synchronize",
        "torch.mps.synchronize",
    }:
        lines.append("Durations cover the decision call and recorded completion synchronization.")
    elif report.transport == "http":
        lines.append("HTTP durations cover the client call, transport, and server waiting.")
    else:
        lines.append(
            "Durations cover the decision call and any caller-supplied synchronization hook. "
            "Completion of backend work is not independently verified."
        )
    if report.schema_version >= 2:
        lines.extend(
            [
                "Model loading, evaluator input preparation, output verification, "
                "and report writing are excluded from these durations.",
            ]
        )
    lines.extend(
        [
            "Mock durations are not inference timings.",
            "p95 uses linear interpolation and is unavailable below 20 calls. "
            "These are descriptive",
            "statistics of this run, not confidence intervals, independent repetitions, "
            "or concurrent throughput.",
            "An interrupted in-flight call may have no saved result or duration. "
            "The tables cover retained attempts.",
            "",
            "## Dataset and protocol",
            "",
        ]
    )
    dataset = {
        "name": report.suite.name,
        "split": report.suite.split,
        "examples": len(report.suite.examples),
        "candidates": len(report.suite.question.criteria),
        "suite_sha256": report.suite_sha256,
        "provenance": report.suite.provenance,
        "question": report.suite.question.model_dump(mode="json"),
        "protocol": report.protocol.model_dump(mode="json"),
    }
    identity = {
        "created_at": report.created_at,
        "transport": report.transport,
        "model": report.model.model_dump(mode="json") if report.model else None,
        "environment": report.environment,
        "config": report.config,
        "schema_version": report.schema_version,
        "prediction_artifact": (
            report.prediction_artifact.model_dump(mode="json")
            if report.prediction_artifact is not None
            else None
        ),
        "run_error": report.error,
    }
    for heading, value in (
        (None, dataset),
        ("Execution and artifact identity", identity),
        ("Memory observations", report.memory),
    ):
        if heading is not None:
            lines.extend([f"## {heading}", ""])
        lines.extend(
            ["```json", json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), "```", ""]
        )
    lines.extend(
        [
            "RSS and CUDA peaks cover the process lifetime. MPS values are sampled snapshots,",
            "not guaranteed peaks. Counters overlap and must not be added. For HTTP/custom",
            "backends, caller memory does not establish model-server memory.",
            "",
            "## Interpretation and reproduction",
            "",
            "Keep `report.json` and `predictions.jsonl` together. The JSON artifacts retain the",
            "selected texts, supplied labels, every retained completed attempt, "
            "per-class metrics, and confusion matrix.",
            "Regenerate this document with `kayak eval report PATH_TO_RUN` "
            "using the current evaluator; older evaluator versions may report fewer metrics.",
            "Source and runtime metadata are snapshots before execution, "
            "not continuous monitoring of changes during a run.",
            "",
            "A subset score applies only to that subset. A full-split score describes this fixed",
            "dataset under this question and candidate recipe. It does not establish production",
            "generalization, training-data independence, calibrated confidence, "
            "statistical significance,",
            "or superiority over another system. "
            "Record dataset overlaps and any prior test exposure",
            "in the study protocol; this renderer cannot determine them from predictions.",
            "",
        ]
    )
    return "\n".join(lines)
