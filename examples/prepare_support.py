"""Export blind support-review sheets, resolve disagreements, and compile checked suites.

Install: uv sync
Export: uv run -m examples.prepare_support export examples/support_review/tickets.json \
  --reviewers fixture-a fixture-b --output .benchmarks/support-review
See docs/support-review.md for adjudication, import, audit, and the offline walkthrough.
No model, tokenizer, external service, or optional dependency is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from examples.support_data import (
    RESOLUTION_COLUMNS,
    Packet,
    Source,
    audit_support,
    compile_reviews,
    csv_bytes,
    json_bytes,
    load_prepared,
    read_record,
    resolution_rows,
    review_sheet,
)
from kayak.eval import load_suite


def write_directory(output: Path, files: dict[str, bytes]) -> None:
    """All content is validated in memory first; existing paths are never replaced."""
    output.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        with (output / name).open("xb") as stream:
            stream.write(content)


def export_packet(source: Path, question_from: Path, reviewers: list[str], output: Path) -> None:
    packet = Packet(
        source=read_record(source.read_bytes(), Source),
        question=load_suite(question_from).question,
        reviewers=reviewers,
    )
    guide = (
        "# Support review packet\n\n"
        "Work separately. Read only your assigned sheet and the draft rubric in "
        "docs/support-review.md. Do not inspect model predictions or another reviewer's labels.\n\n"
        f"Question: {packet.question.instructions}\n\n"
        + "\n".join(f"- {label}: {text}" for label, text in packet.question.criteria.items())
        + "\n\nEdit only `label` and `notes` in your sheet. Every row needs one of the four "
        "labels. `review` is a label for unclear, unrelated, or multiple-team tickets, "
        "not an omitted annotation. The `text_json` cell shows the exact ticket as a "
        "JSON string; preserve it, the ID, reviewer, and packet hash. "
        "Treat CSV columns as text when using a spreadsheet. Save UTF-8 CSV.\n\n"
        "These files contain ticket text. Keep them in the application's approved storage. "
        "Reviewer names and provenance are declarations, not authenticated identities.\n"
    )
    files = {"packet.json": json_bytes(packet.model_dump(mode="json")), "README.md": guide.encode()}
    files.update(
        {
            f"review-{index}.csv": review_sheet(packet, reviewer)
            for index, reviewer in enumerate(packet.reviewers, start=1)
        }
    )
    write_directory(output, files)


def import_reviews(
    packet_path: Path,
    review_paths: list[Path],
    adjudications: Path | None,
    output: Path,
) -> dict[str, object]:
    packet_raw = packet_path.read_bytes()
    packet = read_record(packet_raw, Packet)
    reviews = [path.read_bytes() for path in review_paths]
    resolutions = adjudications.read_bytes() if adjudications is not None else None
    suites, receipt, audit = compile_reviews(packet, reviews, resolutions)
    files = {
        "packet.json": packet_raw,
        "review-record.json": json_bytes(receipt),
        "audit.json": json_bytes(audit),
        **{f"review-{index}.csv": raw for index, raw in enumerate(reviews, start=1)},
        **{f"{suite.split}.json": json_bytes(suite.model_dump(mode="json")) for suite in suites},
    }
    if resolutions is not None:
        files["adjudications.csv"] = resolutions
    write_directory(output, files)
    return audit


class Arguments(argparse.Namespace):
    command: str
    source: Path
    question_from: Path
    reviewers: list[str]
    output: Path
    packet: Path
    reviews: list[Path]
    adjudications: Path | None
    prepared: Path | None
    suites: list[Path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="freeze inputs and create two empty review sheets")
    export.add_argument("source", type=Path)
    export.add_argument(
        "--question-from", type=Path, default=Path("examples/suites/support_pilot.json")
    )
    export.add_argument("--reviewers", nargs=2, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("adjudicate", "import"):
        command = commands.add_parser(name)
        command.add_argument("--packet", type=Path, required=True, help="frozen packet.json")
        command.add_argument("--reviews", type=Path, nargs=2, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "import":
            command.add_argument("--adjudications", type=Path)
    audit = commands.add_parser("audit", help="check retained reviews or existing suites offline")
    inputs = audit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prepared", type=Path, help="directory created by import")
    inputs.add_argument("--suites", type=Path, nargs="+", default=[])
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        if args.command == "export":
            export_packet(args.source, args.question_from, args.reviewers, args.output)
            print(f"Created two prediction-free review sheets in {args.output}")
            return 0
        if args.command == "adjudicate":
            packet = read_record(args.packet.read_bytes(), Packet)
            reviews = [path.read_bytes() for path in args.reviews]
            rows = resolution_rows(packet, reviews)
            content = csv_bytes(RESOLUTION_COLUMNS, rows)
            with args.output.open("xb") as stream:
                stream.write(content)
            print(f"Exported {len(rows)} disagreements; fill label, adjudicator, and reason.")
            return 0
        if args.command == "import":
            report = import_reviews(
                args.packet,
                args.reviews,
                args.adjudications,
                args.output,
            )
        elif args.prepared is not None:
            _, report = load_prepared(args.prepared)
        else:
            report = audit_support([load_suite(path) for path in args.suites])
        print(json_bytes(report).decode(), end="")
        return 0 if report["status"] == "clear" else 1
    except ValidationError as exc:
        locations = [".".join(map(str, row["loc"])) for row in exc.errors(include_input=False)]
        print("Invalid review fields: " + ", ".join(locations[:10]), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        # Validation errors contain structure/identity descriptions, not ticket contents.
        print(f"Support preparation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
