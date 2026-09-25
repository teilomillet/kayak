"""Compare Laya or Jev Choice answers with word overlap on the same labeled cases.

Offline: uv run -m examples.evaluate_provider --output .benchmarks/provider-demo
Laya: uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.evaluate_provider \
    --provider laya --model /path/to/laya --output .benchmarks/laya
Jev: TYPESAFE_API_KEY must be configured; hosted calls use your provider account.
    uv run --with 'typesafe-sdk==0.7.1' -m examples.evaluate_provider \
    --provider jev --model jev-1.13 --output .benchmarks/jev
Own data: --suite examples/suites/support.json (the default fictional support cases).

The default mock always selects the first candidate. It checks integration, not
model quality. Live modes make one sequential call per case and stop on failure.
"""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Literal

from kayak import Choice
from kayak.adapters import Jev, Laya
from kayak.eval import Prediction, PredictionSet, Suite, benchmark, load_suite, write_benchmark

from .benchmark_classifiers import overlap_predictions


class FirstChoiceFixture:
    """A controlled Laya-shaped response with no model, network, or expected labels."""

    def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object:
        question = Choice.model_validate(questions["intent"])
        return {
            "model": "first-choice-fixture",
            "answers": {"intent": {"type": "choice", "choice": next(iter(question.criteria))}},
        }


def run(
    judge: Laya | Jev,
    suite: Suite,
    *,
    output: Path,
    system: str,
    evidence_kind: Literal["simulation", "provider_execution"],
    metadata: dict[str, str] | None = None,
) -> PredictionSet:
    """Borrow a configured adapter; retain partial results and re-raise failures.

    The caller owns model/client lifetime and SDK retry/timeout settings. Only
    case text and the Choice question reach inference. Expected labels stay in
    the saved suite for scoring. Output must be a new directory.
    """
    predictions = PredictionSet(
        system=system,
        method=f"{type(judge).__name__}.judge; original text and Choice; selected IDs only",
        suite=Suite.model_validate(suite.model_dump()),
        predictions=[],
        metadata={
            **(metadata or {}),
            "evidence_kind": evidence_kind,
            "original_status": "incomplete",
            "probabilities": "Omitted from scoring; provider values retained in responses.jsonl",
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "suite.json").write_text(
        predictions.suite.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    try:
        with (output / "responses.jsonl").open("x", encoding="utf-8") as responses:
            for case in predictions.suite.examples:
                try:
                    result = judge.judge(
                        state=case.text,
                        questions={"intent": predictions.suite.question.model_copy(deep=True)},
                    )
                    answer = result.answers["intent"]
                    if answer.type != "choice":
                        raise ValueError("provider must return a Choice answer")
                except BaseException as exc:
                    # Retain failure class, never exception text that could contain credentials.
                    responses.write(json.dumps({"id": case.id, "error": type(exc).__name__}) + "\n")
                    raise
                responses.write(
                    json.dumps({"id": case.id, "result": result.model_dump(mode="json")}) + "\n"
                )
                responses.flush()
                predictions.predictions.append(Prediction(id=case.id, choice=answer.choice))
        predictions.metadata["original_status"] = "complete"
    finally:
        # Missing cases remain in the denominator, including after Ctrl-C. Never
        # invent scores or normalize provider rounding to fit probability metrics.
        (output / "predictions.json").write_text(
            predictions.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        comparison = benchmark(
            {"word_overlap": overlap_predictions(predictions.suite), "provider": predictions},
            allow_recipe_change=True,
        )
        write_benchmark(comparison, output / "comparison")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("mock", "laya", "jev"), default="mock")
    parser.add_argument("--suite", type=Path, default=Path("examples/suites/support.json"))
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory for saved evidence"
    )
    parser.add_argument("--model", help="Laya model directory/ID or requested Jev model name")
    parser.add_argument("--device", default="cpu", help="Laya device (default: cpu)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    if args.provider != "mock" and not args.model:
        parser.error("live providers require an explicit --model")
    suite = load_suite(args.suite)

    if args.provider == "mock":
        run(
            Laya(FirstChoiceFixture()),
            suite,
            output=args.output,
            system="SIMULATION: first candidate, no learned model",
            evidence_kind="simulation",
        )
        print("SIMULATION: controlled responses; no model quality was measured.")
    elif args.provider == "laya":
        laya = importlib.import_module("laya")
        with laya.load(args.model, device=args.device) as agent:
            run(
                Laya(agent),
                suite,
                output=args.output,
                system=f"Laya: {args.model}",
                evidence_kind="provider_execution",
                metadata={
                    "laya_version": importlib.metadata.version("laya"),
                    "requested_model": args.model,
                    "device": args.device,
                    "checkpoint_identity": "Not verified; retain your pinned artifact separately",
                },
            )
    else:
        if not os.environ.get("TYPESAFE_API_KEY"):
            parser.error("set TYPESAFE_API_KEY before selecting the live Jev provider")
        sdk = importlib.import_module("typesafe_sdk")
        with sdk.TypeSafeClient(
            model=args.model, timeout=30, retry=sdk.RetryPolicy(max_retries=0)
        ) as client:
            run(
                Jev(client),
                suite,
                output=args.output,
                system=f"Jev: {args.model}",
                evidence_kind="provider_execution",
                metadata={
                    "typesafe_sdk_version": importlib.metadata.version("typesafe-sdk"),
                    "requested_model": args.model,
                    "timeout_seconds": "30",
                    "sdk_max_retries": "0",
                },
            )
    print(args.output / "comparison" / "benchmark.md")


if __name__ == "__main__":
    main()
