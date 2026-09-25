"""Evaluate editable Choice and ranking cases through a running Kayak service.

Install: uv sync
Check data: uv run -m examples.evaluate_use_cases examples/evaluations/*.json --validate
Try without a model: uv run -m examples.evaluate_use_cases examples/evaluations/*.json --simulate
Measure: uv run -m examples.evaluate_use_cases examples/evaluations/*.json > results.jsonl
Override: KAYAK_BASE_URL=http://127.0.0.1:8000 KAYAK_API_KEY=...
See examples/evaluations/README.md for the use-case map and what each metric establishes.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx
from pydantic import ValidationError

from kayak import (
    ChoiceAnswer,
    Client,
    DecisionRequest,
    DecisionResult,
    KayakError,
    ModelInfo,
    RankingRequest,
    RankingResult,
)
from kayak.eval import CaseReport as CaseReport
from kayak.eval import ChoiceCase as ChoiceCase
from kayak.eval import ChoiceDataset as ChoiceDataset
from kayak.eval import Dataset as Dataset
from kayak.eval import RankingCase as RankingCase
from kayak.eval import RankingDataset as RankingDataset
from kayak.eval import choice_metrics as choice_metrics
from kayak.eval import evaluate_cases, parse_cases, summarize_cases
from kayak.eval import ranking_metrics as ranking_metrics

summarize = summarize_cases


def read_dataset(path: Path) -> tuple[Dataset, str]:
    content = path.read_bytes()
    try:
        dataset = parse_cases(content)
    except ValidationError as exc:
        reason = exc.errors(include_input=False, include_url=False)[0]["msg"]
        raise ValueError(f"{path}: {reason}") from None
    return dataset, hashlib.sha256(content).hexdigest()


def evaluate(client: Client, dataset: Dataset) -> list[CaseReport]:
    """Adapt the real client to the library's request-only callback boundary."""

    def decide(request: DecisionRequest) -> DecisionResult:
        return client.decide(state=request.state, questions=request.questions)

    def rank(request: RankingRequest) -> RankingResult:
        return client.rank(
            state=request.state,
            instructions=request.instructions,
            candidates=request.candidates,
        )

    return evaluate_cases(dataset, decide=decide, rank=rank)


def simulated_response(request: httpx.Request) -> httpx.Response:
    """Choose the first candidate with tied scores, independently of expected labels."""
    decision = DecisionRequest.model_validate_json(request.content)
    result = DecisionResult(
        model=ModelInfo(
            id="example/simulated",
            revision="first-candidate-v1",
            fingerprint="simulated-no-weights",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        answers={
            question_id: ChoiceAnswer(
                choice=next(iter(question.criteria)),
                scores=dict.fromkeys(question.criteria, 0.0),
                probabilities=dict.fromkeys(question.criteria, 1 / len(question.criteria)),
            )
            for question_id, question in decision.questions.items()
        },
        input_tokens=0,
    )
    return httpx.Response(200, content=result.model_dump_json())


class Arguments(argparse.Namespace):
    inputs: list[Path]
    validate: bool
    simulate: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Choice or ranking dataset JSON files")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate", action="store_true", help="check all data without a service")
    mode.add_argument(
        "--simulate", action="store_true", help="try reports without measuring a model"
    )
    args = parser.parse_args(namespace=Arguments())
    try:
        datasets = [read_dataset(path) for path in args.inputs]
        names = [dataset.name for dataset, _ in datasets]
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique across input files")
        if args.validate:
            for dataset, digest in datasets:
                print(
                    json.dumps(
                        dict(
                            name=dataset.name,
                            kind=dataset.kind,
                            cases=len(dataset.cases),
                            dataset_sha256=digest,
                            mode="validate",
                            model_quality="not_measured",
                        )
                    )
                )
            return 0
        client = (
            Client(
                base_url="http://example.test", transport=httpx.MockTransport(simulated_response)
            )
            if args.simulate
            else Client(
                base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
                api_key=os.environ.get("KAYAK_API_KEY"),
            )
        )
        passed = True
        with client:
            for dataset, digest in datasets:
                reports = evaluate(client, dataset)
                summary = summarize(dataset, reports)
                metric = "exact_match" if isinstance(dataset, ChoiceDataset) else "top1"
                passed = (
                    passed and summary["failed"] == 0 and (args.simulate or summary[metric] == 1)
                )
                print(
                    json.dumps(
                        dict(
                            name=dataset.name,
                            kind=dataset.kind,
                            k=dataset.k if isinstance(dataset, RankingDataset) else None,
                            dataset_sha256=digest,
                            mode="simulated" if args.simulate else "http",
                            model_quality="not_measured" if args.simulate else "starter_cases_only",
                            summary=summary,
                            cases=[report.model_dump(mode="json") for report in reports],
                        ),
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        return 0 if passed else 1
    except (OSError, ValueError, KayakError) as exc:
        # Validation diagnostics omit input values; server errors are handled per case above.
        print(f"Evaluation could not complete: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
