"""Bind support reviews to frozen inputs and audit their compiled evaluation suites.

Used by `python -m examples.prepare_support`; requires only the base package.
Review identities are declarations. Hashes detect changed inputs, not honest labeling.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from benchmarks.audit_dataset import NORMALIZATIONS, audit_suites, normalize
from kayak import Choice, DecisionRequest
from kayak.eval import Example, Suite, load_suite

OUTCOMES = ("billing", "shipping", "account", "review")
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")]
Label = Literal["billing", "shipping", "account", "review"]
Split = Literal["development", "test"]
REVIEW_COLUMNS = ("packet_sha256", "id", "text_json", "reviewer", "label", "notes")
RESOLUTION_COLUMNS = (
    "packet_sha256",
    "reviews_sha256",
    "id",
    "text_json",
    "first_reviewer",
    "first_label",
    "first_notes_json",
    "second_reviewer",
    "second_label",
    "second_notes_json",
    "label",
    "adjudicator",
    "reason",
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Ticket(Record):
    id: Identifier
    group_id: Identifier
    split: Split
    text: str = Field(min_length=1)


class Source(Record):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    source_kind: Literal["fictional", "application"]
    provenance: dict[str, str]
    tickets: list[Ticket] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_source(self) -> Self:
        if len({row.id for row in self.tickets}) != len(self.tickets):
            raise ValueError("ticket IDs must be unique across all splits")
        required = {"source", "collection_window", "permission", "deidentification", "split_policy"}
        if not required <= self.provenance.keys() or any(
            not self.provenance[key].strip() for key in required
        ):
            raise ValueError(
                "source, collection_window, permission, deidentification and "
                "split_policy provenance are required"
            )
        return self


class Packet(Record):
    schema_version: Literal[1] = 1
    rubric: Literal["support-routing-v1"] = "support-routing-v1"
    source: Source
    question: Choice
    reviewers: list[Identifier] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def valid_packet(self) -> Self:
        if self.reviewers[0] == self.reviewers[1]:
            raise ValueError("two distinct reviewer identities are required")
        if tuple(self.question.criteria) != OUTCOMES:
            raise ValueError(f"support candidates must have this order: {OUTCOMES}")
        for ticket in self.source.tickets:
            DecisionRequest(state=ticket.text, questions={"intent": self.question})
        return self

    @property
    def sha256(self) -> str:
        return sha256(json_bytes(self.model_dump(mode="json")))


class Vote(Record):
    id: Identifier
    reviewer: Identifier
    label: Label
    notes: str


class Decision(Record):
    id: Identifier
    votes: list[Vote]
    label: Label
    adjudicator: Identifier | None = None
    reason: str = "reviewers agree"


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


Model = TypeVar("Model", bound=BaseModel)


def read_record(raw: bytes, model: type[Model]) -> Model:
    try:
        value: object = json.loads(raw, object_pairs_hook=unique_keys)
        return model.model_validate(value)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        details = [f"{'.'.join(map(str, row['loc'])) or 'record'}: {row['msg']}" for row in errors]
        raise ValueError("invalid review data: " + "; ".join(details[:10])) from None


def csv_bytes(columns: tuple[str, ...], rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows([row[column] for column in columns] for row in rows)
    return stream.getvalue().encode()


def csv_rows(raw: bytes, columns: tuple[str, ...]) -> list[dict[str, str]]:
    # This CLI parses sequentially. Escaped valid ticket text can exceed csv's
    # default field cap; bound it by the file already in memory, then restore it.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, len(raw)))
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, None) != list(columns):
            raise ValueError("CSV header does not match the review format")
        rows: list[dict[str, str]] = []
        for values in reader:
            if len(values) != len(columns):
                raise ValueError("CSV row has missing or extra columns")
            rows.append(dict(zip(columns, values, strict=True)))
        return rows
    except csv.Error as exc:
        raise ValueError("invalid review CSV") from exc
    finally:
        csv.field_size_limit(previous_limit)


def review_sheet(packet: Packet, reviewer: str) -> bytes:
    if reviewer not in packet.reviewers:
        raise ValueError("reviewer is not assigned to this packet")
    # JSON-quoted text cells cannot begin with a spreadsheet formula prefix.
    # Reviewers edit only label/notes; JSON quoting also preserves multiline text.
    return csv_bytes(
        REVIEW_COLUMNS,
        [
            dict(
                packet_sha256=packet.sha256,
                id=ticket.id,
                text_json=json.dumps(ticket.text, ensure_ascii=False),
                reviewer=reviewer,
                label="",
                notes="",
            )
            for ticket in packet.source.tickets
        ],
    )


def read_reviews(packet: Packet, files: list[bytes]) -> tuple[dict[str, list[Vote]], str]:
    if len(files) != 2:
        raise ValueError("exactly two completed review files are required")
    tickets = {row.id: row for row in packet.source.tickets}
    by_reviewer: dict[str, dict[str, Vote]] = {}
    hashes: dict[str, str] = {}
    for raw in files:
        rows = csv_rows(raw, REVIEW_COLUMNS)
        if len(rows) != len(tickets):
            raise ValueError("each reviewer must label every ticket exactly once")
        votes: dict[str, Vote] = {}
        reviewer = rows[0]["reviewer"]
        if reviewer not in packet.reviewers or reviewer in by_reviewer:
            raise ValueError("unexpected or repeated reviewer")
        for row in rows:
            ticket = tickets.get(row["id"])
            if ticket is None or ticket.id in votes:
                raise ValueError("unknown or repeated ticket ID in reviews")
            if row["packet_sha256"] != packet.sha256 or row["reviewer"] != reviewer:
                raise ValueError("review packet or reviewer identity changed")
            if json.loads(row["text_json"]) != ticket.text:
                raise ValueError("review text differs from the frozen ticket")
            if row["label"] not in OUTCOMES:
                raise ValueError("every review needs a label from the four outcomes")
            votes[ticket.id] = Vote.model_validate(
                {
                    "id": ticket.id,
                    "reviewer": reviewer,
                    "label": row["label"],
                    "notes": row["notes"],
                }
            )
        by_reviewer[reviewer] = votes
        hashes[reviewer] = sha256(raw)
    ordered_hashes = {reviewer: hashes[reviewer] for reviewer in packet.reviewers}
    paired = {
        ticket.id: [by_reviewer[reviewer][ticket.id] for reviewer in packet.reviewers]
        for ticket in packet.source.tickets
    }
    return paired, sha256(json_bytes(ordered_hashes))


def resolution_rows(packet: Packet, reviews: list[bytes]) -> list[dict[str, str]]:
    paired, digest = read_reviews(packet, reviews)
    rows = []
    for ticket in packet.source.tickets:
        first, second = paired[ticket.id]
        if first.label == second.label:
            continue
        rows.append(
            dict(
                packet_sha256=packet.sha256,
                reviews_sha256=digest,
                id=ticket.id,
                text_json=json.dumps(ticket.text, ensure_ascii=False),
                first_reviewer=first.reviewer,
                first_label=first.label,
                first_notes_json=json.dumps(first.notes, ensure_ascii=False),
                second_reviewer=second.reviewer,
                second_label=second.label,
                second_notes_json=json.dumps(second.notes, ensure_ascii=False),
                label="",
                adjudicator="",
                reason="",
            )
        )
    return rows


def resolve_reviews(
    packet: Packet, reviews: list[bytes], resolutions: bytes | None
) -> list[Decision]:
    paired, _ = read_reviews(packet, reviews)
    expected = {row["id"]: row for row in resolution_rows(packet, reviews)}
    resolved: dict[str, Decision] = {}
    rows = csv_rows(resolutions, RESOLUTION_COLUMNS) if resolutions is not None else []
    for row in rows:
        identifier = row["id"]
        if identifier not in expected or identifier in resolved:
            raise ValueError("unknown, agreed or repeated ticket in adjudications")
        for column in RESOLUTION_COLUMNS[:-3]:
            if row[column] != expected[identifier][column]:
                raise ValueError("adjudication inputs changed; regenerate from the current reviews")
        if row["label"] not in OUTCOMES or not row["reason"].strip():
            raise ValueError("each disagreement needs a final label and a reason")
        resolved[identifier] = Decision.model_validate(
            dict(
                id=identifier,
                votes=paired[identifier],
                label=row["label"],
                adjudicator=row["adjudicator"],
                reason=row["reason"],
            )
        )
    if resolved.keys() != expected.keys():
        raise ValueError("unresolved review disagreements")
    return [
        resolved[ticket.id]
        if ticket.id in resolved
        else Decision(
            id=ticket.id,
            votes=paired[ticket.id],
            label=paired[ticket.id][0].label,
        )
        for ticket in packet.source.tickets
    ]


def audit_support(suites: list[Suite], tickets: list[Ticket] | None = None) -> dict[str, object]:
    """Reuse the existing text auditor and add support-specific preflight findings."""
    report = audit_suites(suites)
    findings: list[dict[str, object]] = []
    for normalization in NORMALIZATIONS:
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for suite in suites:
            for example in suite.examples:
                grouped[normalize(example.text, normalization)].append(
                    dict(
                        split=suite.split,
                        id=example.id,
                        label=example.label,
                    )
                )
        for key in sorted(grouped):
            rows = grouped[key]
            if len(rows) > 1:
                findings.append(
                    dict(
                        kind="text_overlap",
                        normalization=normalization,
                        rows=rows,
                        conflicting_labels=len({row["label"] for row in rows}) > 1,
                        crosses_splits=len({row["split"] for row in rows}) > 1,
                    )
                )
    identifiers: dict[str, list[str]] = defaultdict(list)
    for suite in suites:
        for example in suite.examples:
            identifiers[example.id].append(suite.split)
    for identifier in sorted(identifiers):
        if len(identifiers[identifier]) > 1:
            findings.append(dict(kind="id_overlap", id=identifier, splits=identifiers[identifier]))
    if tickets is not None:
        groups: dict[str, list[Ticket]] = defaultdict(list)
        for ticket in tickets:
            groups[ticket.group_id].append(ticket)
        for group_id in sorted(groups):
            rows_in_group = groups[group_id]
            if len({row.split for row in rows_in_group}) > 1:
                findings.append(
                    dict(
                        kind="group_overlap",
                        group_id=group_id,
                        ids=[row.id for row in rows_in_group],
                    )
                )
    report.update(
        status="needs_review" if findings else "clear",
        findings=findings,
        grouping_checked=tickets is not None,
        cross_split_checked=len(suites) > 1,
        acceptance="Diagnostic checks only; independence and representativeness need human review.",
    )
    return report


def compile_reviews(
    packet: Packet,
    reviews: list[bytes],
    resolutions: bytes | None,
) -> tuple[list[Suite], dict[str, object], dict[str, object]]:
    decisions = resolve_reviews(packet, reviews, resolutions)
    labels = {row.id: row.label for row in decisions}
    _, review_digest = read_reviews(packet, reviews)
    receipt: dict[str, object] = dict(
        schema_version=1,
        packet_sha256=packet.sha256,
        reviews_sha256=review_digest,
        adjudications_sha256=sha256(resolutions) if resolutions is not None else None,
        source_kind=packet.source.source_kind,
        rubric=packet.rubric,
        evidence_scope="declared_reviews; identities and independence are not authenticated",
        decisions=[row.model_dump(mode="json") for row in decisions],
    )
    provenance = {
        **packet.source.provenance,
        "source_kind": packet.source.source_kind,
        "review_packet_sha256": packet.sha256,
        "review_record_sha256": sha256(json_bytes(receipt)),
        "label_review": "Two declared reviewers; every disagreement explicitly adjudicated",
        "rubric": packet.rubric,
    }
    suites = [
        Suite(
            name=packet.source.name,
            split=split,
            question=packet.question,
            examples=[
                Example(id=row.id, text=row.text, label=labels[row.id])
                for row in packet.source.tickets
                if row.split == split
            ],
            provenance=provenance,
        )
        for split in ("development", "test")
        if any(row.split == split for row in packet.source.tickets)
    ]
    return suites, receipt, audit_support(suites, packet.source.tickets)


def load_prepared(directory: Path) -> tuple[list[Suite], dict[str, object]]:
    """Recompile from retained reviews before trusting derived suites or audit results."""
    packet = read_record((directory / "packet.json").read_bytes(), Packet)
    reviews = [(directory / f"review-{index}.csv").read_bytes() for index in (1, 2)]
    path = directory / "adjudications.csv"
    resolutions = path.read_bytes() if path.exists() else None
    suites, receipt, audit = compile_reviews(packet, reviews, resolutions)
    for name, expected in (("review-record.json", receipt), ("audit.json", audit)):
        if (directory / name).read_bytes() != json_bytes(expected):
            raise ValueError(f"{name} differs from the retained review inputs")
    for suite in suites:
        if load_suite(directory / f"{suite.split}.json").sha256 != suite.sha256:
            raise ValueError("compiled suite differs from the retained review inputs")
    return suites, audit
