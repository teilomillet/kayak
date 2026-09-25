"""Check runnable application adapters without models or external services."""

import asyncio
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from examples import evaluate_rag_async, evaluate_rag_http, evaluate_rag_pipeline
from kayak.eval import (
    RAGDataset,
    RAGEvalConfig,
    RAGInput,
    RAGOutput,
    RAGReport,
    aevaluate_rag,
    evaluate_rag,
    load_rag_report,
    save_rag_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[RAGDataset, RAGEvalConfig]:
    directory = ROOT / "examples" / "rag"
    return (
        RAGDataset.model_validate_json((directory / "dataset.json").read_bytes()),
        RAGEvalConfig.model_validate_json((directory / "config.json").read_bytes()),
    )


def test_pipeline_uses_retrieval_and_context_limits_with_stable_source_ids() -> None:
    request = RAGInput(id="two-topics", query="invoice password")
    one = evaluate_rag_pipeline.Pipeline(
        evaluate_rag_pipeline.PipelineSettings(retrieval_limit=1, context_limit=1)
    ).run(request)
    two = evaluate_rag_pipeline.Pipeline(
        evaluate_rag_pipeline.PipelineSettings(retrieval_limit=2, context_limit=2)
    ).run(request)
    assert one.retrieval is not None and one.retrieval.ids == ["invoice"]
    assert two.retrieval is not None and two.retrieval.ids == ["invoice", "password"]
    assert two.retrieval.scores == {"invoice": 1.0, "password": 1.0}
    assert one.context is not None and one.context.ids == ["invoice"]
    assert two.context is not None and two.context.ids == ["invoice", "password"]
    assert one.answer == "Download your invoice from Settings > Billing > Invoices."
    assert "Security > Password" not in one.context.text
    assert "Security > Password" in two.context.text


@pytest.mark.parametrize(
    ("retrieval_limit", "context_limit", "correct", "recall"),
    [(2, 1, 1.0, 1.0), (2, 0, 0.0, 1.0), (0, 1, 0.0, 0.0)],
)
def test_configured_fixture_quality_changes_when_evidence_is_removed(
    retrieval_limit: int, context_limit: int, correct: float, recall: float
) -> None:
    dataset, config = _fixture()
    settings = evaluate_rag_pipeline.PipelineSettings(
        retrieval_limit=retrieval_limit, context_limit=context_limit
    )
    effective = RAGEvalConfig.model_validate(
        {**config.model_dump(), "system": settings.model_dump()}
    )
    pipeline = evaluate_rag_pipeline.Pipeline(settings)
    report = evaluate_rag(
        dataset, pipeline.run, config=effective, review=evaluate_rag_pipeline.review
    )
    assert report.summary.metrics["answer.correct"].mean == correct
    assert report.summary.metrics["answer.correct"].coverage == 1.0
    assert report.summary.metrics["retrieval.known_recall_at_k"].mean == recall
    assert report.passed == bool(correct)
    assert report.status == "complete"
    assert report.summary.metrics["custom.answer_words"].mean is not None


def test_application_configuration_rejects_unknown_or_wrongly_typed_settings() -> None:
    for settings in ({"context_limit": -1}, {"retrieval_limit": True}, {"model_name": "unused"}):
        with pytest.raises(ValueError):
            evaluate_rag_pipeline.PipelineSettings.model_validate(settings)


def test_independent_reference_changes_evaluation_without_changing_pipeline_output() -> None:
    original, config = _fixture()
    first = original.cases[0]
    dataset = original.model_copy(update={"cases": [first]})
    changed = dataset.model_copy(
        update={"cases": [first.model_copy(update={"reference_answer": "Another reference"})]}
    )
    pipeline = evaluate_rag_pipeline.Pipeline(evaluate_rag_pipeline.PipelineSettings())
    correct = evaluate_rag(
        dataset, pipeline.run, config=config, review=evaluate_rag_pipeline.review
    )
    incorrect = evaluate_rag(
        changed, pipeline.run, config=config, review=evaluate_rag_pipeline.review
    )
    assert correct.results[0].attempt.output == incorrect.results[0].attempt.output
    assert correct.summary.metrics["answer.correct"].mean == 1.0
    assert incorrect.summary.metrics["answer.correct"].mean == 0.0


def test_async_example_retains_original_and_rewritten_queries_without_extra_stage_claims() -> None:
    dataset = evaluate_rag_async.dataset()
    pipeline = evaluate_rag_async.AsyncPipeline("Aurora", "Kayak")
    report = asyncio.run(
        aevaluate_rag(
            dataset,
            pipeline.run,
            config=RAGEvalConfig(max_concurrency=2, repeats=2),
            review=evaluate_rag_async.review,
        )
    )
    assert report.summary.recorded == 4
    assert report.summary.unstable_cases == []
    assert report.summary.metrics["answer.correct"].mean == 1.0
    assert report.summary.metrics["retrieval.known_recall_at_k"].mean is None
    for result in report.results:
        output = result.attempt.output
        assert output is not None and len(output.steps) == 1
        assert "Aurora" in output.final.query and "Kayak" in output.steps[0].query
        assert output.steps[0].retrieval is not None
        assert output.final.retrieval is None
        assert result.assessment is not None
        assert result.assessment.stage_status["retrieval"] == "not_observed"


def test_http_adapter_sends_only_inputs_and_keeps_client_lifetime_with_caller() -> None:
    dataset = evaluate_rag_http.dataset()
    payloads: list[RAGInput] = []

    def service(request: httpx.Request) -> httpx.Response:
        payloads.append(RAGInput.model_validate_json(request.content))
        assert b"reference_answer" not in request.content and b"judgments" not in request.content
        return evaluate_rag_http.mock_service(request)

    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(service)) as client:
        pipeline = evaluate_rag_http.HTTPPipeline(client)
        report = evaluate_rag(dataset, pipeline.run, review=evaluate_rag_http.review)
        assert not client.is_closed
        assert pipeline.run(dataset.cases[0].input).final.id == "retention"
    assert client.is_closed
    assert payloads == [dataset.cases[0].input, dataset.cases[0].input]
    assert report.summary.metrics["answer.correct"].mean == 1.0


@pytest.mark.parametrize("problem", ["unavailable", "bad_json", "wrong_identity"])
def test_http_adapter_error_boundaries_are_retained(problem: str) -> None:
    def service(request: httpx.Request) -> httpx.Response:
        if problem == "unavailable":
            return httpx.Response(503, text="private backend detail")
        if problem == "bad_json":
            return httpx.Response(200, content=b'{"unexpected": true}')
        output = RAGOutput.model_validate_json(evaluate_rag_http.mock_service(request).content)
        changed = output.model_copy(
            update={"final": output.final.model_copy(update={"id": "other"})}
        )
        return httpx.Response(200, json=changed.model_dump(mode="json"))

    with httpx.Client(base_url="http://fixture", transport=httpx.MockTransport(service)) as client:
        report = evaluate_rag(
            evaluate_rag_http.dataset(), evaluate_rag_http.HTTPPipeline(client).run
        )
    attempt = report.results[0].attempt
    assert attempt.output is None and attempt.error is not None
    assert attempt.error.phase == ("output" if problem == "wrong_identity" else "pipeline")
    assert "private backend detail" not in report.model_dump_json()
    assert report.status == "failed"


@pytest.mark.parametrize(
    ("module", "attempts"),
    [
        ("examples.evaluate_rag_pipeline", 3),
        ("examples.evaluate_rag_async", 4),
        ("examples.evaluate_rag_http", 1),
    ],
)
def test_standalone_entrypoints_need_no_optional_inference_stack(
    module: str, attempts: int
) -> None:
    script = (
        "import runpy, sys; module = sys.argv[1]; sys.argv = [module]; "
        "runpy.run_module(module, run_name='__main__'); "
        "assert not {'torch', 'transformers', 'huggingface_hub', 'langchain'} & sys.modules.keys()"
    )
    process = subprocess.run(
        [sys.executable, "-c", script, module],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = RAGReport.model_validate_json(process.stdout)
    assert report.status == "complete" and report.summary.recorded == attempts
    assert report.summary.metrics["answer.correct"].mean == 1.0


def test_sync_cli_records_effective_defaults_overrides_and_checked_output(tmp_path: Path) -> None:
    _, config = _fixture()
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_copy(update={"system": {}}).model_dump_json())
    output = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.evaluate_rag_pipeline",
            "--config",
            str(config_path),
            "--context-limit",
            "0",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = load_rag_report(output)
    assert report == RAGReport.model_validate_json(process.stdout)
    assert report.config.system == {"retrieval_limit": 2, "context_limit": 0}
    assert report.summary.metrics["answer.correct"].mean == 0.0
    assert report.passed is False
    assert all(result.attempt.system == report.config.system for result in report.results)
    preserved = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_rag_report(report, output)
    assert output.read_bytes() == preserved
