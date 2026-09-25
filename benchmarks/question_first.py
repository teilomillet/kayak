"""Test question-first input ordering against a saved BANKING77 dev baseline.

The resident model and its state-first preparation remain unchanged. For this
single-question task, exchanging the API's state/instructions fields produces
exactly question + two newlines + customer text. Record that request transform
separately from the model's artifact/implementation identity. No test split is
accepted and no production input recipe is promoted by this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from kayak import Choice, DecisionRequest, DecisionResult, InputError, load
from kayak.bundle import DEFAULT_MODEL
from kayak.decisions import request_from
from kayak.eval import DecisionBackend, evaluate, load_report

TRANSFORM = "question-first-v1"


def question_first_request(request: DecisionRequest) -> DecisionRequest:
    if len(request.questions) != 1:
        raise InputError("the question-first experiment requires exactly one question")
    question_id, question = next(iter(request.questions.items()))
    return DecisionRequest(
        state=question.instructions,
        questions={question_id: Choice(instructions=request.state, criteria=question.criteria)},
    )


@contextmanager
def question_first(model: DecisionBackend) -> Iterator[None]:
    """Temporarily transform calls in this serial experiment; restore on any exit."""
    original = model.decide

    def decide(
        *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        request = question_first_request(request_from(state, questions))
        return original(state=request.state, questions=request.questions)

    # Keeping the Model instance lets evaluate retain native synchronization and
    # accelerator memory measurement. No encoder method or scoring code changes.
    with patch.object(model, "decide", new=decide):
        yield


class Arguments(argparse.Namespace):
    baseline: Path
    output: Path
    model: str
    model_cache_dir: Path | None
    local_files_only: bool


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(namespace=Arguments())
    baseline = load_report(args.baseline)
    if (
        baseline.status != "complete"
        or baseline.suite.name != "banking77"
        or baseline.suite.split != "dev"
        or baseline.transport != "local"
        or baseline.model is None
        or baseline.model.input_recipe != "clm-choice-v1"
        or baseline.config.get("request_transform", "identity") != "identity"
    ):
        parser.error("a complete, unchanged local BANKING77 development baseline is required")
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    batch_size = baseline.config.get("batch_size")
    budget = baseline.config.get("max_memory_gib")
    if type(batch_size) is not int or batch_size < 1:
        parser.error("baseline must record its positive integer batch_size")
    if budget is not None and type(budget) is not float:
        parser.error("baseline max_memory_gib must be a float or null")
    source = Path(__file__).read_bytes()
    started = perf_counter()
    with load(
        args.model,
        device=baseline.model.device.split(":", 1)[0],
        dtype=baseline.model.dtype,
        batch_size=batch_size,
        cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    ) as model:
        setup_seconds = perf_counter() - started
        if model.info != baseline.model:
            parser.error("model identity, device and precision must match the saved baseline")
        with question_first(model):
            try:
                report = evaluate(
                    model,
                    baseline.suite,
                    output=args.output,
                    warmups=baseline.protocol.warmups,
                    repeats=baseline.protocol.repeats,
                    seed=baseline.protocol.seed,
                    max_memory_gib=budget,
                    config={
                        "request_transform": TRANSFORM,
                        "experiment_source_sha256": hashlib.sha256(source).hexdigest(),
                        "baseline_path": str(args.baseline),
                        "baseline_suite_sha256": baseline.suite_sha256,
                        "batch_size": batch_size,
                        "setup_seconds": setup_seconds,
                        "setup_kind": "model loading",
                    },
                )
            finally:
                if args.output.is_dir():
                    (args.output / "experiment-source.py").write_bytes(source)
    print(json.dumps({"status": report.status, "summary": report.summary}, indent=2))
    return 0 if report.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
