"""Classify your own JSONL records with a local Laya model and editable categories.

Validate: uv run -m examples.classify_file examples/tickets.jsonl \
    --question examples/department.json --validate
Run: uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.classify_file \
    examples/tickets.jsonl --question examples/department.json \
    --model /path/to/laya --output decisions.jsonl

An online --model ID may download weights. CPU is the default. Each input line
is {"id": "...", "text": "..."}; labels are not accepted. Output preserves each
ID, its selected category, and the full provider result. Existing output files
are refused. Failure stops the run; completed rows remain, without retries or
automatic resume. Keep the input file unchanged during validation and inference.
"""

import argparse
import importlib
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ValidationError

from kayak import Choice, InputError, judgments
from kayak.adapters import Laya, ProviderResult

from .process_jsonl import read_tickets


class ClassifiedRecord(BaseModel):
    id: str
    choice: str
    result: ProviderResult


def validate(stream: TextIO, question: Choice) -> int:
    """Check every record against the inference contract before loading a model."""
    count = 0
    for count, ticket in enumerate(read_tickets(stream), start=1):
        try:
            judgments.request_from(ticket.text, {"category": question})
        except InputError as exc:
            raise ValueError(f"{stream.name}:{count}: invalid request: {exc}") from None
    if not count:
        raise ValueError(f"{stream.name}: input must contain at least one record")
    stream.seek(0)
    return count


def run(judge: Laya, stream: TextIO, question: Choice, output: TextIO) -> int:
    """Borrow a loaded adapter and open streams; flush each completed prediction."""
    count = 0
    for ticket in read_tickets(stream):
        result = judge.judge(state=ticket.text, questions={"category": question})
        answer = result.answers["category"]
        if answer.type != "choice":
            raise ValueError("provider must return a Choice answer")
        record = ClassifiedRecord(id=ticket.id, choice=answer.choice, result=result)
        output.write(record.model_dump_json() + "\n")
        output.flush()
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSONL with id and text on every line")
    parser.add_argument("--question", type=Path, required=True, help="Choice question JSON file")
    parser.add_argument("--validate", action="store_true", help="check inputs without a model")
    parser.add_argument("--model", help="local Laya directory or online model ID (may download)")
    parser.add_argument("--device", default="cpu", help="Laya device (default: cpu)")
    parser.add_argument("--output", type=Path, help="new JSONL file for predictions")
    args = parser.parse_args()
    if not args.validate and (not args.model or args.output is None):
        parser.error("inference requires --model and --output")

    try:
        try:
            question = Choice.model_validate_json(args.question.read_text(encoding="utf-8"))
        except ValidationError:
            raise ValueError(
                f"{args.question}: invalid Choice; see examples/department.json"
            ) from None
        with args.input.open(encoding="utf-8") as stream:
            count = validate(stream, question)
            if args.validate:
                print(f"Validated {count} records and the question. No model loaded.")
                return
            # Exclusive creation checks output access before importing or loading Laya.
            with args.output.open("x", encoding="utf-8") as output:
                try:
                    laya = importlib.import_module("laya")
                    with laya.load(args.model, device=args.device) as agent:
                        count = run(Laya(agent), stream, question, output)
                except (Exception, KeyboardInterrupt) as exc:
                    # Provider exception text can contain input data or credentials.
                    parser.exit(
                        1,
                        f"Stopped ({type(exc).__name__}); {args.output} contains partial results. "
                        "No automatic retry or resume.\n",
                    )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Classified {count} records -> {args.output}")


if __name__ == "__main__":
    main()
