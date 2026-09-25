"""Process a file sequentially with one loaded model, reading one record at a time.

Install: uv sync --extra local
Run: uv run --extra local -m examples.process_jsonl examples/tickets.jsonl > decisions.jsonl
Override: KAYAK_MODEL=/path/to/bundle KAYAK_DEVICE=auto

Each input line is {"id": "...", "text": "..."}. Each output line preserves its
ID alongside the full decision. On failure, the command exits nonzero; earlier
output lines remain valid. There is no implicit resume, retry, or deduplication.
"""

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import kayak
from kayak import Choice, DecisionResult


class Ticket(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=65_536)


class ClassifiedTicket(BaseModel):
    id: str
    decision: DecisionResult


class Arguments(argparse.Namespace):
    input: Path


def read_tickets(stream: TextIO) -> Iterator[Ticket]:
    for line_number, line in enumerate(stream, start=1):
        try:
            yield Ticket.model_validate_json(line)
        except ValidationError as exc:
            raise ValueError(f"{stream.name}:{line_number}: invalid ticket") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 JSONL file containing id and text")
    args = parser.parse_args(namespace=Arguments())
    question = Choice(
        instructions="Which team should handle this request?",
        criteria={
            "billing": "Charges, invoices, and refunds",
            "technical": "Bugs and service outages",
        },
    )
    with args.input.open(encoding="utf-8") as stream:
        with kayak.load(
            os.environ.get("KAYAK_MODEL", "Contrastive-LM/CLM-v0.1-8B"),
            device=os.environ.get("KAYAK_DEVICE", "auto"),
            cache_dir=os.environ.get("KAYAK_CACHE_DIR"),
        ) as model:
            for ticket in read_tickets(stream):
                result = model.decide(state=ticket.text, questions={"department": question})
                output = ClassifiedTicket(id=ticket.id, decision=result)
                print(output.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
