"""Inspect pinned CLM Noul/Score recipes and optionally probe a real Kayak service.

This checks the public judgment adapters against pinned upstream semantics. Labels were
authored independently of model outputs. Unknown/ambiguous cases have no label.
The HTTP path uses explicit Choice requests; it does not alter /v1 semantics.
"""

import argparse
import hashlib
import json
import os
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, model_validator

from kayak import (
    Choice,
    ChoiceAnswer,
    Client,
    DecisionRequest,
    DecisionResult,
    JudgmentRequest,
    KayakError,
    ModelInfo,
)
from kayak.decisions import answer_from_scores
from kayak.runtime._preparation import prepare_texts

SUITE = Path(__file__).parent / "data/typed-judgments-v1.jsonl"
REFERENCE_REVISION = "7956937c58ed5839c06ddc4dc6b6b61c3a3e4094"
REFERENCE_SHA256 = "52cec58afbf49ad7b7aa6bdb7e7476ee42bf3fd7a2703d44319dc4b565987335"


class Probe(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    id: str
    kind: Literal["noul", "score"]
    state: str
    instructions: str
    criteria: dict[Literal["false", "true"], str] | list[str] | None
    expected: str | None
    note: str

    @model_validator(mode="after")
    def valid_probe(self) -> Self:
        request = self.request()
        if not self.id.strip() or not self.note.strip():
            raise ValueError("probe ID and note must be nonblank")
        if self.expected is not None and self.expected not in request.questions["probe"].criteria:
            raise ValueError("expected label must be a candidate ID or null for ambiguity")
        return self

    def question(self) -> dict[str, object]:
        return {"type": self.kind, "instructions": self.instructions, "criteria": self.criteria}

    def judgment(self) -> JudgmentRequest:
        """The public adapter input; the legacy probe below is an independent comparison."""
        return JudgmentRequest.model_validate(
            {"state": self.state, "questions": {"probe": self.question()}}
        )

    def request(self) -> DecisionRequest:
        """Compile exactly the upstream text-only recipe, within Kayak's stricter limits."""
        if self.kind == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError("Score needs at least two ordered text levels")
            candidates = {str(index): text for index, text in enumerate(self.criteria)}
        else:
            if self.criteria is not None and not isinstance(self.criteria, dict):
                raise ValueError("Noul criteria must be an optional false/true description map")
            descriptions = self.criteria or {}
            instructions = self.instructions.strip()
            candidates = {}
            for key in ("false", "true"):
                description = descriptions.get(key)
                if not description:
                    description = (
                        f"Yes. This is true: {instructions}"
                        if key == "true"
                        else f"No. This is false: {instructions}"
                    )
                candidates[key] = f"{key}: {description}"
        return DecisionRequest(
            state=self.state,
            questions={"probe": Choice(instructions=self.instructions, criteria=candidates)},
        )


def decode(probe: Probe, answer: ChoiceAnswer) -> dict[str, object]:
    """Match upstream output meaning, including its uncalibrated distribution statistic."""
    if probe.kind == "noul":
        return {"type": "noul", "noul": answer.probabilities["true"]}
    legend = probe.request().questions["probe"].criteria
    # /v1 associates probabilities by ID, even if a server reorders that mapping.
    probabilities = [answer.probabilities[key] for key in legend]
    peak = max(probabilities)
    winner = probabilities.index(peak)
    rest = [value for index, value in enumerate(probabilities) if index != winner]
    return {
        "type": "score",
        "score": sum(index * probability for index, probability in enumerate(probabilities)),
        "confidence": max(0.0, min(1.0, peak - sum(rest) / len(rest))),
        "legend": legend,
        "probabilities": {key: answer.probabilities[key] for key in legend},
    }


def check_reference(probes: list[Probe], reference: Path) -> dict[str, object]:
    """Execute the user-supplied pinned upstream schema after checking its exact hash."""
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    # Only the exact pinned source is executed, even if a different local file is supplied.
    if digest != REFERENCE_SHA256:
        raise ValueError("reference must be the exact pinned CLM schema.py; SHA-256 differs")
    module = runpy.run_path(str(reference))
    build_pairs = cast(Callable[[str, dict[str, object]], object], module["build_pairs"])
    answer_from_logits = cast(
        Callable[[dict[str, object], list[str], list[float]], object], module["answer_from_logits"]
    )
    fixture_model = ModelInfo(
        id="schema-conformance",
        revision="fixture",
        fingerprint="no-inference",
        encoder="none",
        encoder_revision="none",
        device="none",
        dtype="none",
    )
    for probe in probes:
        request = probe.request()
        texts = prepare_texts(request)
        keys = list(request.questions["probe"].criteria)
        expected = {"probe": (texts[0], keys, texts[1:])}
        if build_pairs(probe.state, {"probe": probe.question()}) != expected:
            raise ValueError(f"{probe.id}: prepared inputs differ from upstream")
        if probe.judgment().as_decision() != request:
            raise ValueError(f"{probe.id}: public adapter inputs differ from the checked recipe")
        for scores in ([0.0] * len(keys), [float(index - 1) for index in range(len(keys))]):
            native = decode(probe, answer_from_scores(keys, scores))
            if answer_from_logits(probe.question(), keys, scores) != native:
                raise ValueError(f"{probe.id}: decoding differs from upstream")
            result = DecisionResult(
                model=fixture_model,
                input_tokens=0,
                answers={"probe": answer_from_scores(keys, scores)},
            )
            public = probe.judgment().decode(result).answers["probe"].model_dump()
            fields = (
                ("type", "noul")
                if probe.kind == "noul"
                else ("type", "score", "legend", "probabilities")
            )
            if any(public[key] != native[key] for key in fields):
                raise ValueError(f"{probe.id}: public decoding differs from upstream")
    return {"status": "passed", "revision": REFERENCE_REVISION, "sha256": digest}


def evaluate(probes: list[Probe], client: Client) -> list[dict[str, object]]:
    """Keep every case and failure. Ambiguous cases are inspected, never scored as correct."""
    observations: list[dict[str, object]] = []
    for probe in probes:
        observation: dict[str, object] = {"id": probe.id, "expected": probe.expected}
        request = probe.request()
        try:
            result = client.decide(state=request.state, questions=request.questions)
            answer = result.answers["probe"]
            label = answer.choice
            if probe.kind == "noul":
                # Upstream's diagnostic label rule; not an application threshold.
                label = "true" if answer.probabilities["true"] >= 0.5 else "false"
            observation.update(
                result=result.model_dump(),
                decoded=decode(probe, answer),
                label=label,
                correct=None if probe.expected is None else label == probe.expected,
                judgment=probe.judgment().decode(result).model_dump(),
            )
        except KayakError as exc:
            observation.update(
                error=type(exc).__name__,
                correct=None if probe.expected is None else False,
            )
        observations.append(observation)
    return observations


class Arguments(argparse.Namespace):
    suite: Path
    reference: Path | None
    base_url: str | None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--reference", type=Path, help="pinned upstream schema.py for conformance")
    parser.add_argument(
        "--base-url", help="optional running real-model service; otherwise prepare only"
    )
    args = parser.parse_args(namespace=Arguments())
    try:
        raw = args.suite.read_bytes()
        probes = [Probe.model_validate_json(line) for line in raw.splitlines()]
        if not probes or len({probe.id for probe in probes}) != len(probes):
            raise ValueError("suite must be nonempty with unique IDs")
        report: dict[str, object] = {
            "suite_sha256": hashlib.sha256(raw).hexdigest(),
            "application_quality": "not_evaluated",
            "calibration": "not_evaluated",
            "prepared": [
                {"id": probe.id, "request": probe.request().model_dump()} for probe in probes
            ],
        }
        if args.reference is not None:
            report["conformance"] = check_reference(probes, args.reference)
        failed = False
        if args.base_url is not None:
            with Client(base_url=args.base_url, api_key=os.environ.get("KAYAK_API_KEY")) as client:
                observations = evaluate(probes, client)
            labeled = [item for item in observations if item["expected"] is not None]
            failures = sum("error" in item for item in observations)
            correct = sum(item["correct"] is True for item in labeled)
            report.update(
                application_quality="observed_on_this_suite_only",
                observations=observations,
                labeled=len(labeled),
                correct=correct,
                failures=failures,
                accuracy=correct / len(labeled) if labeled else None,
            )
            failed = failures > 0
        print(json.dumps(report, indent=2, allow_nan=False))
        raise SystemExit(1 if failed else 0)
    except (ValueError, OSError, KayakError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
