"""Evaluate support-routing suggestions; every outcome still needs human review.

Install: uv sync
Validate: uv run -m examples.evaluate_support --validate
Simulate: uv run -m examples.evaluate_support --simulate --output .benchmarks/support-demo
Measure: uv run -m examples.evaluate_support --suite reviewed.json --output .benchmarks/support-test
Service: KAYAK_BASE_URL and optional KAYAK_API_KEY; calls are sequential, without retries.
See docs/support-routing.md for the provisional targets and data-review requirements.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

import httpx

from examples.benchmark_classifiers import overlap_predictions
from examples.evaluate_use_cases import simulated_response
from kayak import Client, KayakError
from kayak.eval import (
    Metric,
    MetricInput,
    Report,
    Suite,
    benchmark,
    default_metrics,
    evaluate,
    load_report,
    load_suite,
    write_benchmark,
)

OUTCOMES = ("billing", "shipping", "account", "review")


class Gate(TypedDict):
    observed: float | int | None
    target: float | int
    passed: bool


def minimum_recall(data: MetricInput) -> float | None:
    recalls = []
    for label in data.labels:
        rows = [row for row in data.samples if row.label == label]
        if not rows:
            return None
        recalls.append(sum(row.choice == label for row in rows) / len(rows))
    return min(recalls)


def review_recall(data: MetricInput) -> float | None:
    rows = [row for row in data.samples if row.label == "review"]
    return sum(row.choice == "review" for row in rows) / len(rows) if rows else None


def validate_suite(suite: Suite) -> None:
    if tuple(suite.question.criteria) != OUTCOMES:
        raise ValueError(f"support pilot candidates must have this order: {OUTCOMES}")


def review_outcomes(report: Report) -> list[dict[str, object]]:
    """Failed calls enter manual review but remain failures in quality metrics."""
    observations = {row.id: row for row in report.observations}
    outcomes: list[dict[str, object]] = []
    for example in report.suite.examples:
        row = observations.get(example.id)
        result = row.attempts[0].result if row else None
        suggestion = result.answers["intent"].choice if result else None
        outcomes.append(
            dict(
                id=example.id,
                suggestion=suggestion,
                review_queue="review" if suggestion is None else suggestion,
                reason="inference_failed_or_missing" if suggestion is None else "model_suggestion",
                human_confirmation_required=True,
            )
        )
    return outcomes


def assess(output: Path) -> dict[str, object]:
    """Read verified observations and apply provisional targets, without inference."""
    report = load_report(output)
    validate_suite(report.suite)
    metrics = (
        *default_metrics(),
        Metric("minimum_recall", minimum_recall, "Lowest recall across all four outcomes."),
        Metric("review_recall", review_recall, "Recall of tickets labeled for manual review."),
    )
    comparison = benchmark(
        {"word_overlap": overlap_predictions(report.suite), "kayak": output},
        metrics=metrics,
        allow_recipe_change=True,
    )
    write_benchmark(comparison, output / "baselines")
    measured = comparison["runs"]["kayak"]["metrics"]
    accuracy = measured["accuracy"]["value"]
    pair = comparison["comparisons"][0]
    # Counts preserve an exact five-percentage-point boundary without subtracting floats.
    paired = pair.get("paired_outcomes")
    gain = (paired["fixed"] - paired["regressed"]) / len(report.suite.examples) if paired else None
    # Targets are policy for this example, never probability/confidence thresholds.
    lower_bounds = {
        "accuracy": (accuracy, 0.95),
        "minimum_recall": (measured["minimum_recall"]["value"], 0.90),
        "review_recall": (measured["review_recall"]["value"], 1.0),
        "accuracy_gain_over_word_overlap": (gain, 0.05),
        "distinct_tickets": (len(report.suite.examples), 200),
        "minimum_tickets_per_outcome": (
            min(sum(row.label == label for row in report.suite.examples) for label in OUTCOMES),
            40,
        ),
    }
    gates: dict[str, Gate] = {
        name: {"observed": value, "target": target, "passed": value is not None and value >= target}
        for name, (value, target) in lower_bounds.items()
    }
    timing = report.summary["attempt_latency"]
    if not isinstance(timing, dict):
        raise ValueError("evaluation report is missing attempt latency")
    p95 = timing["p95_seconds"]
    if p95 is not None and not isinstance(p95, (float, int)):
        raise ValueError("evaluation report has invalid p95 latency")
    gates["attempt_p95_seconds"] = {
        "observed": p95,
        "target": 2.0,
        "passed": p95 is not None and p95 <= 2.0,
    }
    failures = sum(attempt.error is not None for attempt in report.warmups)
    failures += sum(
        attempt.error is not None for row in report.observations for attempt in row.attempts
    )
    gates["failed_calls_including_warmups"] = {
        "observed": failures,
        "target": 0,
        "passed": failures == 0 and report.status == "complete",
    }
    simulated = report.config.get("mode") == "simulated"
    return dict(
        criteria_status="provisional",
        evidence_scope="integration_only" if simulated else "measured_supplied_cases",
        suite_sha256=report.suite_sha256,
        model=report.model.model_dump(mode="json") if report.model else None,
        gates=gates,
        provisional_gates_passed=not simulated and all(gate["passed"] for gate in gates.values()),
        deployment_accepted=False,
        remaining_review=[
            "Independent labels and representative held-out data",
            "Owner acceptance of targets and per-case errors",
            "Target hardware, memory, capacity, and recovery evidence",
        ],
        outcomes=review_outcomes(report),
    )


class Arguments(argparse.Namespace):
    suite: Path
    output: Path | None
    validate: bool
    simulate: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("examples/suites/support_pilot.json"))
    parser.add_argument("--output", type=Path, help="new directory for retained evidence")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true")
    mode.add_argument("--simulate", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    if not args.validate and args.output is None:
        parser.error("--output is required unless --validate is used")
    try:
        suite = load_suite(args.suite)
        validate_suite(suite)
        if args.validate:
            print(json.dumps(dict(cases=len(suite.examples), suite_sha256=suite.sha256)))
            return 0
        assert args.output is not None
        client = (
            Client(
                base_url="http://example.test", transport=httpx.MockTransport(simulated_response)
            )
            if args.simulate
            else Client(
                base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
                api_key=os.environ.get("KAYAK_API_KEY"),
                timeout=5.0,
            )
        )
        with client:
            report = evaluate(
                client,
                suite,
                output=args.output,
                config={"mode": "simulated" if args.simulate else "http"},
            )
        assessment = assess(args.output)
        (args.output / "acceptance.json").write_text(
            json.dumps(assessment, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(assessment, allow_nan=False))
        # Simulation success means the integration ran, even when quality gates fail.
        passed = (args.simulate and report.status == "complete") or assessment[
            "provisional_gates_passed"
        ]
        return 0 if passed else 1
    except (OSError, ValueError, KayakError) as exc:
        print(f"Support evaluation could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
