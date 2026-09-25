"""Exercise report creation with controlled outcomes; no model or network.

Run: uv run -m benchmarks.mock_evaluation --output .benchmarks/mock-audit

Creates complete, failed, and interrupted runs, plus Markdown reports. The
output directory must be new. These artifacts measure evaluator behavior only.
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

from kayak import Choice, ChoiceAnswer, DecisionRequest, DecisionResult, InferenceError, ModelInfo
from kayak.eval import Example, Suite, evaluate, load_report
from kayak.eval._report import render_report


class MockInterruption(KeyboardInterrupt):
    """A deliberate fixture interruption; a real user interrupt still propagates."""


class MockBackend:
    """Always choose the first candidate; optionally fail or interrupt a known call."""

    def __init__(self, *, fail: bool = False, interrupt_after: int | None = None) -> None:
        self.fail = fail
        self.interrupt_after = interrupt_after
        self.calls = 0

    def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        if self.calls == self.interrupt_after:
            raise MockInterruption
        self.calls += 1
        if self.fail and state == "fourth":
            raise InferenceError("intentional mock failure")
        request = DecisionRequest.model_validate({"state": state, "questions": dict(questions)})
        answers = {}
        for identifier, question in request.questions.items():
            candidates = list(question.criteria)
            answers[identifier] = ChoiceAnswer(
                choice=candidates[0],
                scores=dict.fromkeys(candidates, 0.0),
                probabilities=dict.fromkeys(candidates, 1.0 / len(candidates)),
            )
        return DecisionResult(
            model=ModelInfo(
                id="mock/first-candidate",
                revision="fixture-v1",
                fingerprint="mock-no-weights",
                encoder="none",
                encoder_revision="none",
                device="none",
                dtype="none",
            ),
            answers=answers,
            input_tokens=0,
        )


def mock_suite() -> Suite:
    return Suite(
        name="mock-arithmetic",
        split="synthetic",
        question=Choice(instructions="Select intent", criteria={"a": "Alpha", "b": "Beta"}),
        examples=[
            Example(id="0", text="first", label="a"),
            Example(id="1", text="second", label="a"),
            Example(id="2", text="third", label="b"),
            Example(id="3", text="fourth", label="b"),
        ],
        provenance={
            "coverage": "synthetic_fixture",
            "purpose": "evaluator checks, no model quality",
        },
    )


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    scenarios = {
        "complete": MockBackend(),
        "failed": MockBackend(fail=True),
        "interrupted": MockBackend(interrupt_after=1),
    }
    for name, backend in scenarios.items():
        destination = output / name
        try:
            evaluate(
                backend,
                mock_suite(),
                output=destination,
                warmups=0,
                repeats=2,
                config={"evidence_kind": "mock", "scenario": name},
            )
        except MockInterruption:
            if name != "interrupted":
                raise
        report = load_report(destination)
        if report.status != name:
            raise RuntimeError(f"expected {name}, observed {report.status}")
        (destination / "report.md").write_text(render_report(destination), encoding="utf-8")
        print(f"{name}: verified artifacts and Markdown in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.output)


if __name__ == "__main__":
    main()
