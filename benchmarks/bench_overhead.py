"""Measure Kayak's surrounding code with fixed data and no model or network.

Run with `uv run --extra bench -m benchmarks.bench_overhead -o .benchmarks/before.json`.
Use pyperf's --profile or --tracemalloc separately from latency measurements.
"""

from argparse import Namespace
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from io import StringIO
from os import devnull
from pathlib import Path
from typing import TextIO, cast

import httpx
import pyperf
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, Response

import kayak
from kayak import Choice, Client, InferenceError
from kayak._logging import json_logging
from kayak.decisions import (
    DecisionRequest,
    DecisionResult,
    ModelInfo,
    answer_from_scores,
    request_from,
    softmax,
)
from kayak.runtime._preparation import prepare_texts
from kayak.server import create_app


def fixture(
    question_count: int, candidate_count: int, *, padded: bool = False
) -> tuple[DecisionRequest, DecisionResult]:
    # Four large texts fit the character and HTTP byte limits. Their edge spaces
    # reveal temporary copies during blank checks, without tokenization/inference.
    text = " " + "🚣" * 63_998 + " " if padded else None
    questions = {
        f"question-{i}": Choice(
            instructions=text or "Choose the best route for this request.",
            criteria={
                f"option-{j}": text or f"Description of option {j}." for j in range(candidate_count)
            },
        )
        for i in range(question_count)
    }
    request = request_from(text or "I was charged twice. Please help. café 🚣", questions)
    result = DecisionResult(
        model=ModelInfo(
            id="benchmark-stub",
            revision="1",
            fingerprint="fixture",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        answers={
            key: answer_from_scores(list(q.criteria), [float(i) for i in range(candidate_count)])
            for key, q in questions.items()
        },
        input_tokens=0,
    )
    return request, result


class FixedModel:
    """Return a prebuilt value: the HTTP benchmark measures transport overhead."""

    def __init__(self, result: DecisionResult) -> None:
        self.result = result
        self.info = result.model

    def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
        return self.result

    def close(self) -> None:
        pass


def build_result(
    request: DecisionRequest, model: ModelInfo, scores: Mapping[str, list[float]]
) -> DecisionResult:
    """Measure the current runtime's result assembly after tensor-to-list conversion."""
    return DecisionResult(
        model=model,
        answers={
            key: answer_from_scores(list(question.criteria), scores[key])
            for key, question in request.questions.items()
        },
        input_tokens=0,
    )


def build_result_once(
    request: DecisionRequest, model: ModelInfo, scores: Mapping[str, list[float]]
) -> DecisionResult:
    """Experiment: validate freshly assembled fields together, without intermediate models.

    This stays in the benchmark until the real-model path is evaluated. It must
    retain the score checks and full result validation of the current assembly.
    """
    answers: dict[str, object] = {}
    for question_id, question in request.questions.items():
        keys = list(question.criteria)
        values = scores[question_id]
        if len(keys) != len(values) or len(set(keys)) != len(keys):
            raise InferenceError("model score count does not match candidate IDs")
        probabilities = softmax(values)
        answers[question_id] = {
            "choice": keys[max(range(len(values)), key=values.__getitem__)],
            "scores": dict(zip(keys, values, strict=True)),
            "probabilities": dict(zip(keys, probabilities, strict=True)),
        }
    return DecisionResult.model_validate({"model": model, "answers": answers, "input_tokens": 0})


def operations(
    stack: ExitStack,
    request: DecisionRequest,
    result: DecisionResult,
    *,
    diagnostics: bool = False,
) -> dict[str, Callable[[], object]]:
    request_json, result_json = request.model_dump_json(), result.model_dump_json()
    scores = {key: list(answer.scores.values()) for key, answer in result.answers.items()}
    client = stack.enter_context(
        Client(
            base_url="http://benchmark",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, content=result_json, headers={"Content-Type": "application/json"}
                )
            ),
        )
    )
    http = stack.enter_context(
        TestClient(create_app(lambda: FixedModel(result), diagnostics=diagnostics))
    )
    post = partial(
        http.post,
        "/v1/decide",
        content=request_json,
        headers={"Content-Type": "application/json"},
    )
    runs: dict[str, Callable[[], object]] = {
        "validate_python": partial(request_from, request.state, request.questions),
        "validate_json": partial(DecisionRequest.model_validate_json, request_json),
        "prepare_texts": partial(prepare_texts, request),
        "build_result": partial(build_result, request, result.model, scores),
        "build_result_once": partial(build_result_once, request, result.model, scores),
        "decode_result": partial(DecisionResult.model_validate_json, result_json),
        # Side-by-side candidates preserve a reproducible serialization experiment.
        "encode_dict": lambda: JSONResponse(result.model_dump()),
        "encode_json": lambda: Response(result.model_dump_json(), media_type="application/json"),
        "client": partial(client.decide, state=request.state, questions=request.questions),
        "http": post,
    }
    # A fast error response is not a successful optimization. Check the measured
    # transport paths against the fixture outside the timed region in every worker.
    response = post()
    response.raise_for_status()
    assert DecisionResult.model_validate_json(response.content) == result
    assert runs["client"]() == result
    assert runs["build_result"]() == runs["build_result_once"]() == result
    return runs


class Arguments(Namespace):
    workload: str
    diagnostics: str


class UnavailableLogSink(StringIO):
    """Exercise failed writes without retaining event data or needing an exporter."""

    def write(self, text: str) -> int:
        raise OSError("benchmark sink unavailable")


def forward_arguments(command: list[str], args: Namespace) -> None:
    options = cast(Arguments, args)
    command.extend(["--workload", options.workload, "--diagnostics", options.diagnostics])


def main() -> None:
    # Workers must retain the source-root import path, including in other checkouts.
    runner = pyperf.Runner(
        program_args=("-m", "benchmarks.bench_overhead"),
        add_cmdline_args=forward_arguments,
    )
    runner.argparser.add_argument(
        "--workload", default="all", help="One benchmark name (e.g. wide.client), or all"
    )
    runner.argparser.add_argument(
        "--diagnostics",
        choices=["off", "json", "unavailable"],
        default="off",
        help="HTTP events: off, buffered JSON to the null device, or a failing log sink",
    )
    args = cast(Arguments, runner.parse_args())
    package_directory = Path(kayak.__file__).resolve().parent
    if package_directory != Path(__file__).resolve().parents[1] / "kayak":
        runner.argparser.error(
            f"imported Kayak from {package_directory}; run the benchmark from its source root"
        )
    runner.metadata.update(
        scope="No inference or network; HTTP includes TestClient scheduling and a fixed model stub",
        fixture_version=2,
        kayak_directory=str(package_directory),
        diagnostics=args.diagnostics,
        sha256_benchmark=sha256(Path(__file__).read_bytes()).hexdigest(),
        **{f"version_{name}": version(name) for name in ("pydantic", "httpx", "fastapi", "pyperf")},
        **{
            f"sha256_{name}": sha256((package_directory / name).read_bytes()).hexdigest()
            for name in (
                "decisions.py",
                "client.py",
                "server.py",
                "runtime/_preparation.py",
                "_diagnostics.py",
                "_http_diagnostics.py",
                "_logging.py",
            )
        },
    )
    found = False
    for size, counts in {"small": (1, 2), "wide": (32, 8), "padded": (1, 2)}.items():
        request, result = fixture(*counts, padded=size == "padded")
        with ExitStack() as stack:
            if args.diagnostics != "off":
                if args.diagnostics == "unavailable":
                    sink: TextIO = stack.enter_context(UnavailableLogSink())
                else:
                    sink = stack.enter_context(open(devnull, "w"))
                stack.enter_context(json_logging(True, stream=sink))
            for operation, run in operations(
                stack, request, result, diagnostics=args.diagnostics != "off"
            ).items():
                name = f"{size}.{operation}"
                if args.workload in {"all", name}:
                    found = True
                    runner.bench_func(
                        name,
                        run,
                        metadata={
                            "questions": counts[0],
                            "candidates": counts[0] * counts[1],
                            "request_bytes": len(request.model_dump_json().encode()),
                            "response_bytes": len(result.model_dump_json().encode()),
                        },
                    )
    if not found:
        runner.argparser.error(f"unknown workload: {args.workload}")


if __name__ == "__main__":
    main()
