"""Compare inexpensive classifiers and add a custom metric, with no model downloads.

Run: uv run --no-sync -m examples.benchmark_classifiers --output .benchmarks/demo
Optional: --run .benchmarks/eval/clm-dev770-local-001 (reuse its exact labeled suite)
Own data: --suite examples/suites/support.json (same Python Suite contract)
The default constructed examples demonstrate the API, not generalization quality.
"""

import argparse
import re
from pathlib import Path

from kayak import Choice
from kayak.eval import (
    BenchmarkResult,
    Example,
    Metric,
    MetricInput,
    Prediction,
    PredictionSet,
    Suite,
    benchmark,
    default_metrics,
    load_report,
    load_suite,
    write_benchmark,
)


def overlap_predictions(suite: Suite) -> PredictionSet:
    """Rank descriptions by shared words; labels are never read by the classifier."""
    predictions = []
    labels = tuple(suite.question.criteria)
    descriptions = {
        label: set(re.findall(r"\w+", text.lower()))
        for label, text in suite.question.criteria.items()
    }
    for example in suite.examples:
        words = set(re.findall(r"\w+", example.text.lower()))
        scores = {label: len(words & descriptions[label]) for label in labels}
        ranking = sorted(labels, key=scores.__getitem__, reverse=True)
        predictions.append(Prediction(id=example.id, choice=ranking[0], ranking=ranking))
    return PredictionSet(
        system="word overlap",
        method="No training; description word overlap; candidate-order ties.",
        suite=suite,
        predictions=predictions,
        metadata={"probabilities": "unavailable; overlap counts are not probabilities"},
    )


def demo_suite() -> Suite:
    return Suite(
        name="constructed-support-demo",
        split="demo",
        question=Choice(
            instructions="Choose a team.",
            criteria={
                "billing": "Charges invoices refunds",
                "technical": "Bugs outages errors",
            },
        ),
        examples=[
            Example(id="0", text="Please explain these charges", label="billing"),
            Example(id="1", text="Where are my invoices", label="billing"),
            Example(id="2", text="Bugs and errors block my work", label="technical"),
            Example(id="3", text="There are outages again", label="technical"),
        ],
        provenance={"coverage": "constructed demonstration, not a benchmark score"},
    )


def run(
    output: Path, native_run: Path | None = None, *, suite: Suite | None = None
) -> BenchmarkResult:
    """Compare on caller-owned data, a saved run's suite, or the constructed demo."""
    if native_run is not None and suite is not None:
        raise ValueError("choose a saved run or a suite, not both")
    if native_run is not None:
        suite = load_report(native_run).suite
    elif suite is None:
        suite = demo_suite()
    constant = PredictionSet(
        system="constant first label",
        method="No training; always choose the first candidate.",
        suite=suite,
        predictions=[
            Prediction(id=example.id, choice=next(iter(suite.question.criteria)))
            for example in suite.examples
        ],
    )
    overlap = overlap_predictions(suite)
    critical_label = next(iter(suite.question.criteria))

    def error_cost(data: MetricInput) -> float:
        # Illustrative business cost: configure the critical intent before evaluation.
        return sum(
            (3.0 if row.label == critical_label else 1.0)
            for row in data.samples
            if row.choice != row.label
        ) / len(data.samples)

    cost = Metric(
        "example_error_cost",
        error_cost,
        "Illustrative error cost; first intent costs three units.",
        direction="lower",
        parameters=(("critical_label", critical_label), ("critical_cost", 3)),
    )
    inputs: dict[str, Path | PredictionSet] = {"constant": constant, "word_overlap": overlap}
    if native_run is not None:
        inputs["kayak"] = native_run
    result = benchmark(inputs, metrics=(*default_metrics(), cost), allow_recipe_change=True)
    write_benchmark(result, output)
    for name, predictions in (("constant", constant), ("word_overlap", overlap)):
        (output / f"{name}.predictions.json").write_text(
            predictions.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path)
    source.add_argument("--suite", type=Path)
    args = parser.parse_args()
    run(args.output, args.run, suite=load_suite(args.suite) if args.suite else None)
    print(args.output / "benchmark.md")


if __name__ == "__main__":
    main()
