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
Progress goes to stderr; stdout previews up to five saved predictions and names
the output file. Validation and missing dependencies create no output file.
"""

import argparse
import importlib
import json
import shlex
import sys
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ValidationError

from kayak import Choice, InputError, decisions, judgments
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


def preview_value(value: str) -> str:
    """Keep terminal previews short and escape control characters; saved values stay exact."""
    return json.dumps(value if len(value) <= 80 else value[:77] + "...")


def run(judge: Laya, stream: TextIO, question: Choice, output: TextIO, *, total: int) -> int:
    """Borrow resources; flush predictions before showing progress or a bounded preview."""
    count = 0
    preview: list[tuple[str, str]] = []
    try:
        for ticket in read_tickets(stream):
            print(f"Processing record {count + 1}/{total}...", file=sys.stderr, flush=True)
            result = judge.judge(state=ticket.text, questions={"category": question})
            answer = result.answers["category"]
            if answer.type != "choice":
                raise ValueError("provider must return a Choice answer")
            record = ClassifiedRecord(id=ticket.id, choice=answer.choice, result=result)
            output.write(record.model_dump_json() + "\n")
            output.flush()
            count += 1
            if count <= 5:
                preview.append((ticket.id, answer.choice))
    finally:
        # Show saved results after progress, including when a later prediction fails.
        if preview:
            print(f"Preview (first {len(preview)} predictions):", flush=True)
            for identifier, choice in preview:
                print(f"  {preview_value(identifier)} -> {preview_value(choice)}", flush=True)
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
        except ValidationError as exc:
            raise ValueError(
                f"{args.question}: invalid Choice: {decisions.validation_message(exc)}. "
                "Edit the instructions and criteria; see examples/department.json."
            ) from None
        with args.input.open(encoding="utf-8") as stream:
            count = validate(stream, question)
            if args.validate:
                print(f"Validated {count} records and the question. No model loaded.")
                return
            print(f"Validated {count} records and the question.", file=sys.stderr, flush=True)
            if args.output.exists():
                parser.error(f"{args.output} already exists. Choose a new --output path.")
            # Missing dependencies should leave no empty output that blocks the corrected run.
            print(f"Loading Laya on {args.device}...", file=sys.stderr, flush=True)
            try:
                laya = importlib.import_module("laya")
            except (Exception, KeyboardInterrupt) as exc:
                command = shlex.join(
                    [
                        "uv",
                        "run",
                        "--with",
                        "laya==0.3.20",
                        "--with",
                        "transformers<5",
                        "-m",
                        "examples.classify_file",
                        *sys.argv[1:],
                    ]
                )
                parser.exit(
                    1,
                    f"Cannot import Laya or its dependencies ({type(exc).__name__}). "
                    "Rerun with:\n"
                    f"  {command}\nNo output file was created.\n",
                )
            # Exclusive creation still prevents overwrite if another process creates the path.
            with args.output.open("x", encoding="utf-8") as output:
                stage = "model loading"
                try:
                    with laya.load(args.model, device=args.device) as agent:
                        print("Model loaded.", file=sys.stderr, flush=True)
                        stage = "classification"
                        count = run(Laya(agent), stream, question, output, total=count)
                        stage = "model cleanup"
                except (Exception, KeyboardInterrupt) as exc:
                    # Provider exception text can contain input data or credentials.
                    recovery = {
                        "model loading": "Check --model points to a Laya checkpoint or "
                        "accessible model ID, and that --device is available.",
                        "classification": "Check the last record shown in progress "
                        "and the model settings.",
                        "model cleanup": "Predictions were saved before model shutdown failed.",
                    }[stage]
                    if isinstance(exc, KeyboardInterrupt):
                        recovery = "Interrupted; inspect saved predictions before starting again."
                    parser.exit(
                        1,
                        f"Stopped during {stage} ({type(exc).__name__}). {recovery}\n"
                        f"Completed predictions remain in {args.output}; partial results may be "
                        "empty if loading failed. Use a new --output path to rerun. "
                        "No automatic retry or resume.\n",
                    )
    except FileExistsError:
        parser.error(f"{args.output} already exists. Choose a new --output path.")
    except FileNotFoundError as exc:
        parser.error(
            f"{exc.filename}: path not found. Check the input and --question paths, "
            "and create the --output parent directory if needed."
        )
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Classified {count} records -> {args.output}")


if __name__ == "__main__":
    main()
