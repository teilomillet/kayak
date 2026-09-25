"""Offline RAG validation, schema export, scoring, and report inspection."""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

from ._rag_experiment import (
    RAGAttempt,
    RAGDataset,
    RAGEvalConfig,
    RAGInput,
    RAGOutput,
    RAGReport,
    RAGReview,
    RAGReviewInput,
    RAGReviewRecord,
)
from ._rag_reports import load_rag_report, render_rag_report, save_rag_report, score_rag


class Arguments(argparse.Namespace):
    command: str | None
    dataset: Path
    attempts: Path
    config: Path | None
    output: Path
    path: Path
    record: str
    reviews: Path | None


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kayak eval rag", description=__doc__)
    commands = root.add_subparsers(dest="command")
    schema = commands.add_parser("schema", help="print a portable JSON Schema; no model calls")
    schema.add_argument(
        "record",
        choices=[
            "input",
            "output",
            "dataset",
            "config",
            "attempt",
            "review",
            "review_input",
            "review_record",
            "report",
        ],
    )
    validate = commands.add_parser("validate", help="validate dataset and evaluation settings")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--config", type=Path)
    inputs = commands.add_parser("inputs", help="export planned input/repeat/system JSONL")
    inputs.add_argument("dataset", type=Path)
    inputs.add_argument("--config", type=Path)
    reviews = commands.add_parser(
        "review-inputs", help="export reference material and fingerprints for external reviewers"
    )
    reviews.add_argument("dataset", type=Path)
    reviews.add_argument("attempts", type=Path)
    reviews.add_argument("--config", type=Path)
    score = commands.add_parser("score", help="score recorded attempt JSONL without execution")
    score.add_argument("dataset", type=Path)
    score.add_argument("attempts", type=Path)
    score.add_argument("--config", type=Path)
    score.add_argument("--reviews", type=Path, help="optional external RAGReviewRecord JSONL")
    score.add_argument("--output", type=Path, required=True, help="new report JSON file")
    report = commands.add_parser("report", help="verify a saved report and print Markdown")
    report.add_argument("path", type=Path)
    return root


def _configuration(path: Path | None) -> RAGEvalConfig:
    if path is None:
        return RAGEvalConfig()
    return RAGEvalConfig.model_validate_json(path.read_bytes())


def _attempts(path: Path) -> list[RAGAttempt]:
    records: list[RAGAttempt] = []
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(RAGAttempt.model_validate_json(line))
        except ValidationError:
            raise ValueError(f"invalid attempt record on line {number}") from None
    return records


def _attach_reviews(attempts: list[RAGAttempt], path: Path) -> list[RAGAttempt]:
    positions = {(attempt.case_id, attempt.repeat): index for index, attempt in enumerate(attempts)}
    if len(positions) != len(attempts):
        raise ValueError("duplicate attempt identities")
    seen: set[tuple[str, int]] = set()
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = RAGReviewRecord.model_validate_json(line)
        except ValidationError:
            raise ValueError(f"invalid review record on line {number}") from None
        key = (record.case_id, record.repeat)
        if key not in positions or key in seen:
            raise ValueError("review identity is unknown or duplicated")
        seen.add(key)
        index = positions[key]
        attempt = attempts[index]
        if attempt.review is not None or attempt.review_error is not None:
            raise ValueError("external review cannot replace a recorded review outcome")
        payload = {**attempt.model_dump(), "review": record.review.model_dump()}
        attempts[index] = RAGAttempt.model_validate(payload)
    return attempts


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv, namespace=Arguments())
    try:
        if args.command == "schema":
            schemas: dict[str, type[BaseModel]] = {
                "input": RAGInput,
                "output": RAGOutput,
                "dataset": RAGDataset,
                "config": RAGEvalConfig,
                "attempt": RAGAttempt,
                "review": RAGReview,
                "review_input": RAGReviewInput,
                "review_record": RAGReviewRecord,
                "report": RAGReport,
            }
            print(json.dumps(schemas[args.record].model_json_schema(), indent=2, allow_nan=False))
        elif args.command == "report":
            print(render_rag_report(load_rag_report(args.path)), end="")
        elif args.command in {"validate", "inputs", "review-inputs", "score"}:
            dataset = RAGDataset.model_validate_json(args.dataset.read_bytes())
            config = _configuration(args.config)
            if args.command == "validate":
                print(
                    json.dumps(
                        {
                            "dataset": dataset.name,
                            "cases": len(dataset.cases),
                            "planned_attempts": len(dataset.cases) * config.repeats,
                            "dataset_sha256": dataset.sha256,
                            "config_sha256": config.sha256,
                            "execution": "not_run",
                        }
                    )
                )
            elif args.command == "inputs":
                for case in dataset.cases:
                    for repeat in range(config.repeats):
                        print(
                            json.dumps(
                                {
                                    "input": case.input.model_dump(mode="json"),
                                    "repeat": repeat,
                                    "system": config.system,
                                },
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        )
            elif args.command == "review-inputs":
                report = score_rag(dataset, _attempts(args.attempts), config=config)
                cases = {case.input.id: case for case in dataset.cases}
                for result in report.results:
                    attempt = result.attempt
                    if attempt.output is None:
                        continue
                    request = RAGReviewInput(case=cases[attempt.case_id], output=attempt.output)
                    print(
                        json.dumps(
                            {
                                "case_id": attempt.case_id,
                                "repeat": attempt.repeat,
                                "review_input": request.model_dump(mode="json"),
                                "review_input_sha256": request.sha256,
                            },
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                    )
            else:
                attempts = _attempts(args.attempts)
                if args.reviews is not None:
                    attempts = _attach_reviews(attempts, args.reviews)
                report = score_rag(dataset, attempts, config=config)
                save_rag_report(report, args.output)
                print(
                    json.dumps(
                        {
                            "status": report.status,
                            "passed": report.passed,
                            "planned_attempts": report.summary.planned,
                            "recorded_attempts": report.summary.recorded,
                            "gates": [gate.model_dump(mode="json") for gate in report.gates],
                            "report": str(args.output),
                        },
                        allow_nan=False,
                    )
                )
                return int(report.status != "complete" or report.passed is False)
        else:
            cli.print_help()
    except ValidationError as exc:
        reason = exc.errors(include_input=False, include_url=False)[0]["msg"]
        print(f"RAG evaluation could not complete: {reason}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"RAG evaluation could not complete: {exc}", file=sys.stderr)
        return 2
    return 0
