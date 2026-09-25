"""Freeze an encoder diagnostic's texts, identities, tolerances, and supplied token rows.

Install: uv sync
Prepare: uv run -m benchmarks.encoder_probe prepare --suite examples/suites/support_pilot.json \
  --embedding-atol 0.00001 --score-atol 0.00001 --output .benchmarks/probe-texts.json
Bind previously captured tokens with `bind`; see docs/encoder-comparison.md.
These commands operate on files only. They never load a model or a tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from kayak import DecisionRequest
from kayak.bundle import ModelSpec
from kayak.eval import Suite, load_suite
from kayak.runtime._preparation import prepare_texts, validate_token_lengths

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[a-f0-9]{40}$")]
Name = Annotated[str, Field(min_length=1, pattern=r"\S")]
Version = Annotated[str, Field(pattern=r"^[0-9]+(?:\.[0-9]+)+[a-zA-Z0-9.+_-]*$")]
Token = Annotated[int, Field(ge=0)]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Sequence(Record):
    id: Name
    text: str = Field(min_length=1)


class Case(Record):
    id: Name
    state_id: Name
    candidates: dict[str, str] = Field(min_length=2)
    gold: str | None = None

    @model_validator(mode="after")
    def valid_gold(self) -> Self:
        if self.gold is not None and self.gold not in self.candidates:
            raise ValueError("gold label must name a candidate")
        if any(not label.strip() for label in self.candidates):
            raise ValueError("candidate IDs must be nonblank")
        return self


class TokenRow(Record):
    id: Name
    text_sha256: Digest
    token_ids: list[Token] = Field(min_length=1)


class TokenCapture(Record):
    schema_version: Literal[1] = 1
    text_spec_sha256: Digest
    tokenizer_id: Name
    tokenizer_revision: Revision
    tokenizer_files_sha256: dict[str, Digest]
    tokenizer_library: Name
    tokenizer_library_version: Version
    collector_sha256: Digest
    add_special_tokens: Literal[True] = True
    chat_template: Literal[False] = False
    truncation: Literal[False] = False
    sequences: list[TokenRow] = Field(min_length=1)

    @model_validator(mode="after")
    def required_files(self) -> Self:
        if not {"tokenizer.json", "tokenizer_config.json"} <= self.tokenizer_files_sha256.keys():
            raise ValueError("hash tokenizer.json and tokenizer_config.json")
        return self


class Tolerances(Record):
    embedding_atol: Finite = Field(ge=0)
    score_atol: Finite = Field(ge=0)
    # Verifies each saved normalized vector against its raw pooled vector.
    normalization_atol: Finite = Field(default=1e-6, ge=0, le=1e-4)


class Probe(Record):
    schema_version: Literal[1] = 1
    purpose: Literal["encoder_diagnostic_not_quality"] = "encoder_diagnostic_not_quality"
    evidence_kind: Literal["encoder_diagnostic", "synthetic_fixture"]
    model: ModelSpec
    source_suite_sha256: Digest
    source_split: Name
    selection: Name
    pooling: Literal["last_non_padding_token"] = "last_non_padding_token"
    normalization: Literal["fp32_l2_plus_1e-12"] = "fp32_l2_plus_1e-12"
    tolerances: Tolerances
    sequences: list[Sequence] = Field(min_length=1)
    cases: list[Case] = Field(min_length=1)
    token_capture: TokenCapture | None = None

    @property
    def text_spec_sha256(self) -> str:
        return digest(json_bytes(self.model_dump(mode="json", exclude={"token_capture"})))

    @property
    def sha256(self) -> str:
        return digest(json_bytes(self.model_dump(mode="json")))

    @model_validator(mode="after")
    def valid_inputs(self) -> Self:
        identifiers = [row.id for row in self.sequences]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("sequence IDs must be unique")
        if len({row.text for row in self.sequences}) != len(self.sequences):
            raise ValueError("deduplicate exact texts before freezing a probe")
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("case IDs must be unique")
        used = {
            identifier
            for case in self.cases
            for identifier in [case.state_id, *case.candidates.values()]
        }
        if used != set(identifiers):
            raise ValueError("case references must cover exactly the frozen sequences")
        if self.token_capture is not None:
            capture = self.token_capture
            if capture.text_spec_sha256 != self.text_spec_sha256:
                raise ValueError("token capture belongs to a different text specification")
            if (capture.tokenizer_id, capture.tokenizer_revision) != (
                self.model.encoder_id,
                self.model.encoder_revision,
            ):
                raise ValueError("tokenizer identity differs from the pinned encoder snapshot")
            rows = {row.id: row for row in capture.sequences}
            if len(rows) != len(capture.sequences) or rows.keys() != set(identifiers):
                raise ValueError("token capture must contain every sequence exactly once")
            for sequence in self.sequences:
                if rows[sequence.id].text_sha256 != digest(sequence.text.encode()):
                    raise ValueError("token row text hash differs from the frozen sequence")
            for case in self.cases:
                lengths = [
                    len(rows[identifier].token_ids)
                    for identifier in [case.state_id, *case.candidates.values()]
                ]
                validate_token_lengths(lengths, max_length=self.model.max_length)
        return self


def digest(raw: bytes) -> str:
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


def read_record(path: Path, model: type[Model]) -> Model:
    value: object = json.loads(path.read_bytes(), object_pairs_hook=unique_keys)
    try:
        return model.model_validate(value)
    except ValidationError as exc:
        raise ValueError(validation_message(exc)) from None


def validation_message(exc: ValidationError) -> str:
    errors = exc.errors(include_input=False, include_context=False, include_url=False)
    details = [f"{'.'.join(map(str, row['loc'])) or 'record'}: {row['msg']}" for row in errors]
    return "invalid diagnostic data: " + "; ".join(details[:10])


def prepare_probe(
    suite: Suite,
    model: ModelSpec,
    tolerances: Tolerances,
    identifiers: list[str] | None = None,
    *,
    limit: int = 6,
    synthetic: bool = False,
) -> Probe:
    if suite.split == "test":
        raise ValueError("prepare diagnostic probes from development data; reserve the test split")
    if identifiers is not None:
        known = {row.id for row in suite.examples}
        if (
            not identifiers
            or len(set(identifiers)) != len(identifiers)
            or not set(identifiers) <= known
        ):
            raise ValueError("selected IDs must be distinct known development cases")
        selected = [row for row in suite.examples if row.id in identifiers]
        selection = "Explicit IDs, retained in source suite order"
    else:
        if limit < 1:
            raise ValueError("limit must be positive")
        selected = suite.examples[:limit]
        selection = f"First {len(selected)} cases in source suite order; no outcome-based tuning"
    by_text: dict[str, str] = {}
    cases = []
    for example in selected:
        texts = prepare_texts(
            DecisionRequest(state=example.text, questions={"intent": suite.question})
        )
        for text in texts:
            if text not in by_text:
                by_text[text] = f"sequence-{len(by_text) + 1:04d}"
        cases.append(
            Case(
                id=example.id,
                state_id=by_text[texts[0]],
                gold=example.label,
                candidates={
                    label: by_text[text]
                    for label, text in zip(suite.question.criteria, texts[1:], strict=True)
                },
            )
        )
    return Probe(
        evidence_kind="synthetic_fixture" if synthetic else "encoder_diagnostic",
        model=model,
        source_suite_sha256=suite.sha256,
        source_split=suite.split,
        selection=selection,
        tolerances=tolerances,
        sequences=[Sequence(id=identifier, text=text) for text, identifier in by_text.items()],
        cases=cases,
    )


def bind_tokens(probe: Probe, capture: TokenCapture) -> Probe:
    if probe.token_capture is not None:
        raise ValueError("this probe is already bound; preserve it and prepare a new experiment")
    return Probe.model_validate(
        {**probe.model_dump(mode="json"), "token_capture": capture.model_dump(mode="json")}
    )


class Arguments(argparse.Namespace):
    command: str
    suite: Path
    manifest: Path
    ids: list[str] | None
    limit: int
    embedding_atol: float
    score_atol: float
    synthetic_fixture: bool
    probe: Path
    tokens: Path
    output: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="freeze a diagnostic text specification")
    prepare.add_argument("--suite", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, default=Path("kayak/models/clm-v0.1-8b.json"))
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument("--ids", nargs="+")
    selection.add_argument("--limit", type=int, default=6)
    prepare.add_argument("--embedding-atol", type=float, required=True)
    prepare.add_argument("--score-atol", type=float, required=True)
    prepare.add_argument("--synthetic-fixture", action="store_true", help="mark non-model fixtures")
    prepare.add_argument("--output", type=Path, required=True)
    bind = commands.add_parser("bind", help="bind supplied saved token rows to frozen texts")
    bind.add_argument("--probe", type=Path, required=True)
    bind.add_argument("--tokens", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        if args.command == "prepare":
            probe = prepare_probe(
                load_suite(args.suite),
                read_record(args.manifest, ModelSpec),
                Tolerances(embedding_atol=args.embedding_atol, score_atol=args.score_atol),
                args.ids,
                limit=args.limit,
                synthetic=args.synthetic_fixture,
            )
        else:
            probe = bind_tokens(
                read_record(args.probe, Probe), read_record(args.tokens, TokenCapture)
            )
        content = json_bytes(probe.model_dump(mode="json"))
        with args.output.open("xb") as stream:
            stream.write(content)
        print(
            json.dumps(
                dict(
                    probe_sha256=probe.sha256,
                    text_spec_sha256=probe.text_spec_sha256,
                    sequences=len(probe.sequences),
                    cases=len(probe.cases),
                    status="bound"
                    if probe.token_capture is not None
                    else "awaiting_saved_token_rows",
                )
            )
        )
        return 0
    except ValidationError as exc:
        print(validation_message(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Encoder probe could not be prepared: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
