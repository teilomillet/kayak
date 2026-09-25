"""Exercise an installed wheel in a fresh, base-only environment.

Run with `python -I scripts/check_wheel.py`; isolation prevents the checkout
from hiding missing wheel files. This check never loads or downloads a model.
"""

import asyncio
import subprocess
import sys
import tempfile
import tomllib
from importlib.metadata import version
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path

import httpx

import kayak
from kayak import Choice, ChoiceAnswer, DecisionRequest, DecisionResult, InputError, ModelInfo
from kayak.adapters import Laya
from kayak.decisions import answer_from_scores
from kayak.eval import (
    Example,
    RAGCase,
    RAGContext,
    RAGDataset,
    RAGEvalConfig,
    RAGGate,
    RAGInput,
    RAGReview,
    RAGReviewInput,
    RAGTrace,
    RankedOutput,
    Suite,
    benchmark,
    compare,
    evaluate,
    evaluate_rag,
    load_rag_report,
    load_report,
    load_suite,
    render_comparison,
    save_rag_report,
)


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    assert sys.flags.isolated, "use python -I to exclude checkout imports"
    assert kayak.__file__ is not None
    assert not Path(kayak.__file__).resolve().is_relative_to(source / "kayak")
    assert not hasattr(kayak, "rag") and "rag" not in kayak.__all__
    assert find_spec("kayak._rag_pipeline") is None
    expected: object = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    assert version("kayak") == expected
    for dependency in ("torch", "transformers", "fastapi", "huggingface_hub"):
        assert find_spec(dependency) is None, f"{dependency} defeats the base-only check"
    assert files("kayak").joinpath("py.typed").is_file()
    from kayak.bundle import ModelSpec

    manifest = files("kayak").joinpath("models/clm-v0.1-8b.json").read_text()
    assert ModelSpec.model_validate_json(manifest).model_id == "Contrastive-LM/CLM-v0.1-8B"
    result = DecisionResult(
        model=ModelInfo(
            id="wheel-check",
            revision="fixture",
            fingerprint="fixture",
            encoder="none",
            encoder_revision="none",
            device="none",
            dtype="none",
        ),
        answers={"route": ChoiceAnswer(choice="a", scores={"a": 0.0}, probabilities={"a": 1.0})},
        input_tokens=0,
    )

    def respond(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, content=result.model.model_dump_json())
        incoming = DecisionRequest.model_validate_json(request.content)
        value = DecisionResult(
            model=result.model,
            input_tokens=0,
            answers={
                name: answer_from_scores(list(question.criteria), [0.0] * len(question.criteria))
                for name, question in incoming.questions.items()
            },
        )
        return httpx.Response(200, content=value.model_dump_json())

    request = DecisionRequest(
        state="wheel check",
        questions={"route": Choice(instructions="select", criteria={"a": "option"})},
    )
    with kayak.Client(
        base_url="http://wheel.test",
        transport=httpx.MockTransport(respond),
    ) as client:
        actual = client.decide(
            state=request.state,
            questions=request.questions,
        )
        assert client.model_info() == result.model
        ranked = client.rank(state="wheel check", instructions="select", candidates={"a": "option"})
        assert [(item.id, item.score, item.probability) for item in ranked.ranked] == [
            ("a", 0.0, 1.0)
        ]
        questions: dict[str, kayak.JudgmentQuestion] = {
            "n": kayak.Noul(instructions="True?"),
            "s": kayak.Score(instructions="Assess", criteria=["Low", "High"]),
        }
        judged = client.judge(state="wheel check", questions=questions)
        binary, score = judged.answers["n"], judged.answers["s"]
        assert isinstance(binary, kayak.NoulAnswer) and binary.noul == 0.5
        assert isinstance(score, kayak.ScoreAnswer) and score.score == 0.5
    assert actual == result

    async def async_check() -> None:
        async with kayak.AsyncClient(
            base_url="http://wheel.test", transport=httpx.MockTransport(respond)
        ) as client:
            assert await client.judge(state="wheel check", questions=questions) == judged
            assert await client.model_info() == result.model
            assert await client.decide(state=request.state, questions=request.questions) == result
            assert (
                await client.rank(
                    state="wheel check", instructions="select", candidates={"a": "option"}
                )
                == ranked
            )

    asyncio.run(async_check())
    suite = Suite(
        name="wheel-check",
        split="dev",
        question=request.questions["route"],
        examples=[Example(id="one", text=request.state, label="a")],
    )

    def eval_response(request: httpx.Request) -> httpx.Response:
        prediction = result.model_copy(update={"answers": {"intent": result.answers["route"]}})
        return httpx.Response(200, content=prediction.model_dump_json())

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "evaluation"
        suite_path = Path(directory) / "suite.json"
        suite_path.write_text(suite.model_dump_json(), encoding="utf-8")
        with kayak.Client(
            base_url="http://wheel.test", transport=httpx.MockTransport(eval_response)
        ) as client:
            report = evaluate(client, load_suite(suite_path), output=output, warmups=0)
        assert report.status == "complete" and report.summary["accuracy"] == 1.0
        assert load_report(output) == report
        comparison = compare(output, output)
        assert comparison["cases"][0]["outcome"] == "both_correct"
        assert "Regressions (0)" in render_comparison(comparison)
        analysis = benchmark({"before": output, "after": output})
        assert analysis["comparisons"][0]["cases"] == comparison["cases"]

        def pipeline(request: RAGInput) -> RAGTrace:
            trace = RAGTrace(
                id=request.id,
                query=request.query,
                documents={"fixture": "fixture answer"},
                retrieval=RankedOutput(ids=["fixture"]),
                context=RAGContext(ids=["fixture"], text="fixture answer"),
                answer="fixture answer",
            )
            checks = trace.evaluate(expected_answer="fixture answer", expected_sources=["fixture"])
            assert checks.answer_correct is True and checks.source_coverage["context"] is True
            assert checks.answer_grounded is None
            return trace

        def review(request: RAGReviewInput) -> RAGReview:
            return RAGReview(
                review_input_sha256=request.sha256,
                correct=request.output.final.answer == request.case.reference_answer,
                provenance={"rubric": "wheel-fixture-exact-match"},
            )

        rag_data = RAGDataset(
            name="wheel-rag-check",
            cases=[
                RAGCase(
                    input=RAGInput(id="one", query="fixture query"),
                    reference_answer="fixture answer",
                )
            ],
        )
        rag_report = evaluate_rag(
            rag_data,
            pipeline,
            review=review,
            config=RAGEvalConfig(gates=[RAGGate(metric="answer.correct", minimum=1.0)]),
        )
        assert rag_report.passed is True
        assert rag_report.summary.metrics["answer.grounded"].mean is None
        rag_path = Path(directory) / "rag.json"
        save_rag_report(rag_report, rag_path)
        assert load_rag_report(rag_path) == rag_report
    try:
        kayak.load(device="invalid")
    except InputError:
        pass
    else:
        raise AssertionError("invalid device was accepted")
    for arguments in (
        (),
        ("--help",),
        ("--version",),
        ("serve", "--help"),
        ("info", "--help"),
        ("rank", "--help"),
        ("eval", "--help"),
        ("eval", "run", "--help"),
        ("eval", "rag", "--help"),
        ("validate", "-"),
    ):
        command = [sys.executable, "-I", "-m", "kayak", *arguments]
        completed = subprocess.run(
            command, input=request.model_dump_json(), check=True, capture_output=True, text=True
        )
        if arguments == ("validate", "-"):
            assert DecisionRequest.model_validate_json(completed.stdout) == request
        else:
            assert "kayak" in completed.stdout
        if arguments == ("--version",):
            assert completed.stdout.strip() == f"kayak {expected}"

    class Provider:
        def predict(self, state: str, questions: dict[str, dict[str, object]]) -> object:
            assert state == "wheel check" and questions["n"]["type"] == "noul"
            return {"answers": {"n": {"type": "noul", "noul": 0.5}}}

    foreign = Laya(Provider()).judge(
        state="wheel check", questions={"n": kayak.Noul(instructions="True?")}
    )
    assert foreign.provider == "laya" and foreign.calibration == "unknown"
    assert foreign.answers["n"].type == "noul" and foreign.answers["n"].noul == 0.5
    assert (
        not {"torch", "transformers", "fastapi", "huggingface_hub", "laya", "typesafe_sdk"}
        & sys.modules.keys()
    )
    print(f"Installed wheel passed: kayak {expected}")


if __name__ == "__main__":
    main()
