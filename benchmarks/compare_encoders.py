"""Compare saved encoder vectors and Choice scores without loading a model.

Install: uv sync
Run: uv run -m benchmarks.compare_encoders --probe probe.json \
  --reference reference.json --candidate candidate.json --output comparison.json
See docs/encoder-comparison.md for export identities and interpretation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from benchmarks.encoder_probe import (
    Digest,
    Finite,
    Name,
    Probe,
    Record,
    Token,
    Version,
    digest,
    json_bytes,
    read_record,
    validation_message,
)


class Runtime(Record):
    name: Name
    version: Version
    collector_sha256: Digest
    device: Name
    dtype: Name
    attention_implementation: Name
    batch_size: int = Field(gt=0)
    packages: dict[Name, Name] = Field(min_length=1)


class Embedding(Record):
    id: Name
    token_ids: list[Token] = Field(min_length=1)
    pooled: list[Finite] = Field(min_length=1)
    normalized: list[Finite] = Field(min_length=1)


class Scores(Record):
    id: Name
    scores: dict[str, Finite] = Field(min_length=2)


class Snapshot(Record):
    schema_version: Literal[1] = 1
    evidence_kind: Literal["encoder_diagnostic", "synthetic_fixture"]
    probe_sha256: Digest
    model_fingerprint: Digest
    heads_implementation_sha256: Digest
    score_scale: Finite = Field(ge=0, le=100)
    input_mode: Literal["token_ids", "text"]
    pooling: Literal["last_non_padding_token"] = "last_non_padding_token"
    normalization: Literal["fp32_l2_plus_1e-12"] = "fp32_l2_plus_1e-12"
    runtime: Runtime
    embeddings: list[Embedding] = Field(min_length=1)
    cases: list[Scores] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_rows(self) -> Self:
        if len({row.id for row in self.embeddings}) != len(self.embeddings):
            raise ValueError("duplicate embedding sequence ID")
        if len({row.id for row in self.cases}) != len(self.cases):
            raise ValueError("duplicate score case ID")
        return self


def validate_snapshot(probe: Probe, snapshot: Snapshot) -> None:
    if probe.token_capture is None:
        raise ValueError("bind genuine saved token rows before comparing encoder snapshots")
    if snapshot.evidence_kind != probe.evidence_kind:
        raise ValueError("synthetic and encoder diagnostic evidence cannot be mixed")
    if snapshot.probe_sha256 != probe.sha256:
        raise ValueError("snapshot is not bound to this probe")
    if snapshot.model_fingerprint != probe.model.fingerprint:
        raise ValueError("snapshot declares a different model manifest")
    vectors = {row.id: row for row in snapshot.embeddings}
    expected = {row.id: row for row in probe.token_capture.sequences}
    if vectors.keys() != expected.keys():
        raise ValueError("snapshot must contain every frozen sequence exactly once")
    for identifier, row in vectors.items():
        if row.token_ids != expected[identifier].token_ids:
            raise ValueError("snapshot token IDs differ from the frozen tokens")
        if (
            len(row.pooled) != probe.model.hidden_size
            or len(row.normalized) != probe.model.hidden_size
        ):
            raise ValueError("embedding width differs from the model manifest")
        norm = math.hypot(*row.pooled)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("raw pooled vector has zero or nonfinite norm")
        normalized = [value / (norm + 1e-12) for value in row.pooled]
        normalized_norm = math.hypot(*row.normalized)
        if normalized_norm == 0 or not math.isfinite(normalized_norm):
            raise ValueError("saved normalized vector has zero or nonfinite norm")
        if any(
            abs(actual - expected_value) > probe.tolerances.normalization_atol
            for actual, expected_value in zip(row.normalized, normalized, strict=True)
        ):
            raise ValueError("saved normalization differs from the recorded pooled vector")
    scores = {row.id: row for row in snapshot.cases}
    if scores.keys() != {case.id for case in probe.cases}:
        raise ValueError("snapshot must contain every frozen case exactly once")
    for case in probe.cases:
        if scores[case.id].scores.keys() != case.candidates.keys():
            raise ValueError("scores must name every candidate without extras")


def vector_difference(left: list[float], right: list[float]) -> dict[str, float]:
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("embedding differences exceed finite numeric range")
    left_norm, right_norm = math.hypot(*left), math.hypot(*right)
    cosine = math.fsum((a / left_norm) * (b / right_norm) for a, b in zip(left, right, strict=True))
    maximum = max(differences)
    rms = (
        maximum
        * math.sqrt(math.fsum((value / maximum) ** 2 for value in differences) / len(differences))
        if maximum
        else 0.0
    )
    return {
        "maximum_absolute_error": maximum,
        "root_mean_square_error": rms,
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "reference_norm": left_norm,
        "candidate_norm": right_norm,
    }


def ranking_summary(
    scores: dict[str, float],
    candidates: list[str],
    gold: str | None,
) -> dict[str, object]:
    # Stable sorting preserves the production first-in-input-order tie rule.
    ranking = sorted(candidates, key=scores.__getitem__, reverse=True)
    top_margin = scores[ranking[0]] - scores[ranking[1]]
    gold_margin = (
        scores[gold] - max(scores[label] for label in candidates if label != gold)
        if gold is not None
        else None
    )
    if not math.isfinite(top_margin) or (
        gold_margin is not None and not math.isfinite(gold_margin)
    ):
        raise ValueError("score margins exceed finite numeric range")
    return dict(
        scores={label: scores[label] for label in candidates},
        ranking=ranking,
        choice=ranking[0],
        top_two_margin=top_margin,
        gold_rank=ranking.index(gold) + 1 if gold is not None else None,
        gold_margin=gold_margin,
    )


def compare(probe: Probe, reference: Snapshot, candidate: Snapshot) -> dict[str, object]:
    # Revalidate mutable in-memory models as well as file inputs.
    probe = Probe.model_validate(probe.model_dump(mode="json"))
    reference = Snapshot.model_validate(reference.model_dump(mode="json"))
    candidate = Snapshot.model_validate(candidate.model_dump(mode="json"))
    validate_snapshot(probe, reference)
    validate_snapshot(probe, candidate)
    if reference.heads_implementation_sha256 != candidate.heads_implementation_sha256:
        raise ValueError("use the same scoring implementation to isolate the encoder comparison")
    if reference.score_scale != candidate.score_scale:
        raise ValueError("score scales differ; encoder effects are confounded")
    left_vectors = {row.id: row for row in reference.embeddings}
    right_vectors = {row.id: row for row in candidate.embeddings}
    vectors: list[dict[str, object]] = []
    embedding_errors = []
    for sequence in probe.sequences:
        left, right = left_vectors[sequence.id], right_vectors[sequence.id]
        pooled = vector_difference(left.pooled, right.pooled)
        normalized = vector_difference(left.normalized, right.normalized)
        embedding_errors.append(
            max(pooled["maximum_absolute_error"], normalized["maximum_absolute_error"])
        )
        vectors.append(dict(id=sequence.id, pooled=pooled, normalized=normalized))
    left_scores = {row.id: row.scores for row in reference.cases}
    right_scores = {row.id: row.scores for row in candidate.cases}
    cases: list[dict[str, object]] = []
    score_errors = []
    changed = []
    for case in probe.cases:
        labels = list(case.candidates)
        left_summary = ranking_summary(left_scores[case.id], labels, case.gold)
        right_summary = ranking_summary(right_scores[case.id], labels, case.gold)
        maximum_error = max(
            abs(left_scores[case.id][label] - right_scores[case.id][label]) for label in labels
        )
        score_errors.append(maximum_error)
        if not math.isfinite(maximum_error):
            raise ValueError("score differences exceed finite numeric range")
        choice_changed = left_summary["choice"] != right_summary["choice"]
        if choice_changed:
            changed.append(case.id)
        cases.append(
            dict(
                id=case.id,
                reference=left_summary,
                candidate=right_summary,
                maximum_absolute_score_error=maximum_error,
                choice_changed=choice_changed,
            )
        )
    within_tolerance = (
        max(embedding_errors) <= probe.tolerances.embedding_atol
        and max(score_errors) <= probe.tolerances.score_atol
    )
    return dict(
        schema_version=1,
        evidence_kind=probe.evidence_kind,
        probe_sha256=probe.sha256,
        model=probe.model.model_dump(mode="json"),
        tolerances=probe.tolerances.model_dump(mode="json"),
        reference_runtime=reference.runtime.model_dump(mode="json"),
        candidate_runtime=candidate.runtime.model_dump(mode="json"),
        input_modes=dict(reference=reference.input_mode, candidate=candidate.input_mode),
        heads_implementation_sha256=reference.heads_implementation_sha256,
        score_scale=reference.score_scale,
        identity_checks="matched declarations, token rows, dimensions, and normalization",
        maximum_embedding_error=max(embedding_errors),
        maximum_score_error=max(score_errors),
        within_declared_numeric_tolerances=within_tolerance,
        changed_choices=changed,
        comparison_passed=within_tolerance and not changed,
        vectors=vectors,
        cases=cases,
        interpretation=(
            "Diagnostic comparison of supplied saved outputs only. No model was executed, "
            "identities are declarations, and checkpoint execution is not authenticated. "
            "This does not establish model quality, calibration, historical training revision "
            "equivalence, or equivalence outside these cases and declared tolerances."
        ),
    )


class Arguments(argparse.Namespace):
    probe: Path
    reference: Path
    candidate: Path
    output: Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        # Hash the normalized snapshots actually compared, without rereading mutable inputs.
        probe = read_record(args.probe, Probe)
        reference = read_record(args.reference, Snapshot)
        candidate = read_record(args.candidate, Snapshot)
        report = compare(probe, reference, candidate)
        report["normalized_artifact_sha256"] = {
            "reference": digest(json_bytes(reference.model_dump(mode="json"))),
            "candidate": digest(json_bytes(candidate.model_dump(mode="json"))),
        }
        with args.output.open("xb") as stream:
            stream.write(json_bytes(report))
        print("Saved encoder comparison; inspect numeric differences and changed_choices.")
        return 0 if report["comparison_passed"] else 1
    except ValidationError as exc:
        print(validation_message(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, OverflowError) as exc:
        print(f"Encoder comparison rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
