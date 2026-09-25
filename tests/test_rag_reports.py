"""Independent checks of report accounting, review identity, and saved evidence."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from kayak.eval._rag import (
    RAGAnswerReview,
    RAGContext,
    RAGJudgments,
    RAGReplay,
    RAGTrace,
)
from kayak.eval._rag_experiment import (
    RAGAttempt,
    RAGCase,
    RAGDataset,
    RAGEvalConfig,
    RAGFailure,
    RAGGate,
    RAGInput,
    RAGOutput,
    RAGReport,
    RAGReview,
    RAGReviewInput,
    RAGScore,
)
from kayak.eval._rag_reports import (
    load_rag_report,
    render_rag_report,
    save_rag_report,
    score_rag,
)
from kayak.eval._ranking import RankedOutput


def dataset(*identifiers: str) -> RAGDataset:
    return RAGDataset(
        name="reviewed-invoice-questions",
        cases=[
            RAGCase(
                input=RAGInput(id=identifier, query=f"Invoice question {identifier}?"),
                judgments=RAGJudgments(relevance={"invoice": 1, "irrelevant": 0}),
                reference_answer="Open billing settings.",
            )
            for identifier in identifiers
        ],
    )


def attempt(identifier: str, *, repeat: int = 0, hit: bool = True) -> RAGAttempt:
    identifiers = ["invoice", "irrelevant"] if hit else ["irrelevant", "invoice"]
    trace = RAGTrace(
        id=identifier,
        query=f"Invoice question {identifier}?",
        retrieval=RankedOutput(ids=identifiers),
        context=RAGContext(ids=["invoice"], text="Open billing settings."),
        answer="Open billing settings.",
    )
    return RAGAttempt(
        input=RAGInput(id=identifier, query=trace.query),
        repeat=repeat,
        output=RAGOutput(final=trace),
    )


def reviewed(
    row: RAGAttempt, *, correct: bool = True, value: float = 4.0, case: RAGCase | None = None
) -> RAGAttempt:
    assert row.output is not None
    reference = case or dataset(row.case_id).cases[0]
    return row.model_copy(
        update={
            "review": RAGReview(
                review_input_sha256=RAGReviewInput(case=reference, output=row.output).sha256,
                correct=correct,
                grounded=True,
                scores={"rubric": RAGScore(value=value)},
                provenance={"reviewer": "independent-fixture-rubric"},
            )
        }
    )


def test_missing_and_failed_attempts_remain_in_unknown_coverage() -> None:
    data = dataset("one", "two", "three")
    failure = RAGAttempt(
        input=data.cases[1].input, error=RAGFailure(phase="pipeline", type="TimeoutError")
    )
    report = score_rag(data, [reviewed(attempt("one")), failure], config=RAGEvalConfig(k=1))
    assert report.status == "incomplete"
    assert report.passed is None
    assert report.summary.planned == 3
    assert report.summary.recorded == 2
    assert report.summary.missing == 1
    assert report.summary.pipeline_failures == 1
    metric = report.summary.metrics["retrieval.top1"]
    assert metric.mean == 1.0
    assert (metric.observed, metric.unknown) == (1, 2)
    assert metric.coverage == pytest.approx(1 / 3)


def test_unknown_quality_is_not_a_measured_failure_or_an_implicit_pass() -> None:
    config = RAGEvalConfig(gates=[RAGGate(metric="answer.correct", minimum=0.0, min_coverage=0.0)])
    report = score_rag(dataset("one"), [attempt("one")], config=config)
    assert report.status == "complete"
    assert report.summary.pipeline_failures == 0
    assert report.summary.metrics["answer.correct"].mean is None
    assert report.gates[0].status == "unknown"
    assert report.passed is False


def test_gates_check_native_scales_both_bounds_and_coverage() -> None:
    data = dataset("one", "two")
    rows = [reviewed(attempt("one"), value=-2.0), reviewed(attempt("two"), value=8.0)]
    config = RAGEvalConfig(
        gates=[
            RAGGate(metric="custom.rubric", minimum=2.5, maximum=3.5),
            RAGGate(metric="answer.correct", minimum=1.0),
        ]
    )
    report = score_rag(data, rows, config=config)
    metric = report.summary.metrics["custom.rubric"]
    assert (metric.mean, metric.minimum, metric.maximum, metric.stddev) == (3.0, -2.0, 8.0, 5.0)
    assert report.passed is True
    assert [gate.status for gate in report.gates] == ["passed", "passed"]
    too_strict = RAGEvalConfig(gates=[RAGGate(metric="custom.rubric", maximum=2.9)])
    assert score_rag(data, rows, config=too_strict).gates[0].status == "failed"
    partial = score_rag(data, [rows[0], attempt("two")], config=config)
    assert partial.gates[0].status == "unknown"
    permitted = RAGEvalConfig(
        gates=[RAGGate(metric="custom.rubric", minimum=-3.0, min_coverage=0.5)]
    )
    assert score_rag(data, [rows[0], attempt("two")], config=permitted).passed is True


def test_unobserved_custom_gate_and_explicit_unavailable_score_remain_unknown() -> None:
    row = attempt("one")
    assert row.output is not None
    row = row.model_copy(
        update={
            "review": RAGReview(
                review_input_sha256=RAGReviewInput(
                    case=dataset("one").cases[0], output=row.output
                ).sha256,
                scores={"rubric": RAGScore(value=None, reason="No applicable reference.")},
            )
        }
    )
    report = score_rag(
        dataset("one"),
        [row],
        config=RAGEvalConfig(gates=[RAGGate(metric="custom.optional", minimum=0.0)]),
    )
    assert report.summary.metrics["custom.rubric"].unknown == 1
    assert report.summary.metrics["custom.optional"].unknown == 1
    assert report.gates[0].status == "unknown"
    assert report.passed is False


def test_repeats_retain_variation_and_canonical_order() -> None:
    data = dataset("two", "one")
    rows = [attempt("one", repeat=1), attempt("two", repeat=1, hit=False), attempt("two")]
    report = score_rag(data, rows, config=RAGEvalConfig(repeats=2, k=1))
    assert [(row.attempt.case_id, row.attempt.repeat) for row in report.results] == [
        ("two", 0),
        ("two", 1),
        ("one", 1),
    ]
    assert report.summary.unstable_cases == ["two"]
    assert report.summary.planned == 4
    assert report.summary.missing == 1
    metric = report.summary.metrics["retrieval.top1"]
    assert metric.mean == pytest.approx(2 / 3)
    assert metric.stddev == pytest.approx((2 / 9) ** 0.5)
    assert metric.unknown == 1
    reordered = score_rag(data, list(reversed(rows)), config=report.config)
    assert reordered == report


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "repeat", "query", "id"])
def test_invalid_observation_bindings_are_rejected(kind: str) -> None:
    row = attempt("one")
    rows = [row]
    if kind == "duplicate":
        rows.append(row)
    elif kind == "unknown":
        rows = [attempt("other")]
    elif kind == "repeat":
        rows = [attempt("one", repeat=1)]
    else:
        assert row.output is not None
        trace = row.output.final.model_copy(update={kind: "different"})
        rows = [row.model_copy(update={"output": RAGOutput(final=trace)})]
    with pytest.raises(ValueError):
        score_rag(dataset("one"), rows)


def test_history_is_retained_without_applying_final_labels_to_rewritten_queries() -> None:
    row = attempt("one")
    assert row.output is not None
    step = RAGTrace(
        id="rewrite-1", query="Which account is involved?", retrieval=RankedOutput(ids=["account"])
    )
    row = row.model_copy(update={"output": RAGOutput(final=row.output.final, steps=[step])})
    report = score_rag(dataset("one"), [row], config=RAGEvalConfig(k=1))
    result = report.results[0]
    assert result.attempt.output is not None
    assert result.attempt.output.steps == [step]
    assert result.assessment is not None
    assert result.assessment.retrieval is not None
    assert result.assessment.retrieval.ranked_ids == ["invoice", "irrelevant"]
    assert report.summary.metrics["retrieval.top1"].mean == 1.0


def test_intermediate_diagnostic_cannot_be_laundered_into_ordinary_scores() -> None:
    row = reviewed(attempt("one"))
    assert row.output is not None
    intervention = RAGTrace(
        id="injected-evidence",
        query="Find the exact invoice source.",
        retrieval=RankedOutput(ids=["invoice"]),
        replay=RAGReplay(
            original_trace_sha256="a" * 64,
            changed_boundary="retrieval",
            purpose="Caller supplied a reviewed source.",
        ),
    )
    output = RAGOutput(final=row.output.final, steps=[intervention])
    row = row.model_copy(update={"output": output, "review": None})
    report = score_rag(
        dataset("one"),
        [row],
        config=RAGEvalConfig(gates=[RAGGate(metric="retrieval.top1", minimum=0.0)]),
    )
    assert report.summary.diagnostic_excluded == 1
    assert report.summary.metrics["retrieval.top1"].mean is None
    assert report.summary.metrics["retrieval.top1"].observed == 0
    assert report.summary.metrics["retrieval.top1"].unknown == 0
    assert report.passed is False
    assert report.results[0].assessment is not None
    assert report.results[0].assessment.ordinary_aggregate_eligible is False


def test_mixed_diagnostic_experiment_cannot_pass_despite_a_perfect_ordinary_subset() -> None:
    ordinary = attempt("one")
    diagnostic = attempt("two", hit=False)
    assert diagnostic.output is not None
    replay = RAGReplay(
        original_trace_sha256="a" * 64,
        changed_boundary="retrieval",
        purpose="Caller tested an alternative retrieval output.",
    )
    output = RAGOutput(final=diagnostic.output.final.model_copy(update={"replay": replay}))
    diagnostic = diagnostic.model_copy(update={"output": output})
    config = RAGEvalConfig(k=1, gates=[RAGGate(metric="retrieval.top1", minimum=1.0)])
    report = score_rag(dataset("one", "two"), [ordinary, diagnostic], config=config)
    metric = report.summary.metrics["retrieval.top1"]
    assert (metric.mean, metric.observed, metric.unknown, metric.coverage) == (1.0, 1, 0, 1.0)
    assert report.summary.diagnostic_excluded == 1
    assert report.gates[0].status == "passed"
    assert report.status == "complete"
    assert report.passed is False
    assert "mixed diagnostic experiment" in render_rag_report(report)
    unconfigured = score_rag(dataset("one", "two"), [ordinary, diagnostic])
    assert unconfigured.passed is None


def test_stage_and_review_failures_do_not_erase_observed_retrieval() -> None:
    row = attempt("one")
    assert row.output is not None
    final = row.output.final.model_copy(update={"answer": None, "errors": {"answer": "Timeout"}})
    step = RAGTrace(id="first-search", query="Find evidence.", errors={"retrieval": "Timeout"})
    row = row.model_copy(
        update={
            "output": RAGOutput(final=final, steps=[step]),
            "review_error": RAGFailure(phase="review", type="JudgeUnavailable"),
            "duration_seconds": 1.5,
        }
    )
    config = RAGEvalConfig(gates=[RAGGate(metric="retrieval.top1", minimum=1.0)])
    report = score_rag(dataset("one"), [row], config=config)
    assert report.status == "failed"
    assert report.summary.stage_errors == {"retrieval": 1, "answer": 1}
    assert report.summary.review_failures == 1
    assert report.summary.pipeline_failures == 0
    assert report.summary.metrics["retrieval.top1"].mean == 1.0
    assert report.summary.metrics["answer.correct"].mean is None
    assert report.summary.metrics["pipeline.duration_seconds"].mean == 1.5
    assert report.gates[0].status == "passed"
    assert report.passed is False


def test_failure_duration_is_known_and_absent_offline_duration_is_unknown() -> None:
    failure = RAGAttempt(
        input=dataset("two").cases[0].input,
        error=RAGFailure(phase="pipeline", type="TimeoutError"),
        duration_seconds=2.5,
    )
    report = score_rag(dataset("one", "two"), [attempt("one"), failure])
    metric = report.summary.metrics["pipeline.duration_seconds"]
    assert metric.mean == 2.5
    assert (metric.observed, metric.unknown) == (1, 1)
    assert report.status == "failed"


def test_score_only_review_does_not_invent_answer_quality() -> None:
    row = attempt("one")
    assert row.output is not None
    row = row.model_copy(
        update={
            "review": RAGReview(
                review_input_sha256=RAGReviewInput(
                    case=dataset("one").cases[0], output=row.output
                ).sha256,
                scores={"rubric": RAGScore(value=100.0)},
            )
        }
    )
    report = score_rag(dataset("one"), [row])
    assert report.summary.metrics["custom.rubric"].mean == 100.0
    assert report.summary.metrics["answer.correct"].mean is None


@pytest.mark.parametrize("missing", ["answer", "context"])
def test_boolean_review_requires_the_reviewed_observations(missing: str) -> None:
    row = attempt("one")
    assert row.output is not None
    output = RAGOutput(final=row.output.final.model_copy(update={missing: None}))
    row = reviewed(row.model_copy(update={"output": output}))
    with pytest.raises(ValueError, match="requires an observed answer|requires observed context"):
        score_rag(dataset("one"), [row])


def test_saved_review_binds_the_full_history() -> None:
    row = reviewed(attempt("one"))
    assert row.output is not None
    changed = RAGOutput(
        final=row.output.final,
        steps=[RAGTrace(id="new-step", query="New retrieval question.")],
    )
    with pytest.raises(ValueError, match="different inputs, references, or outputs"):
        score_rag(dataset("one"), [row.model_copy(update={"output": changed})])


def test_prebound_review_can_supply_missing_judgment_but_cannot_be_contradicted() -> None:
    row = attempt("one")
    assert row.output is not None
    data = dataset("one")
    original = RAGAnswerReview(
        trace_sha256=row.output.final.sha256,
        correct=True,
        provenance={"reviewer": "original-reviewer"},
    )
    case = data.cases[0].model_copy(
        update={"judgments": data.cases[0].judgments.model_copy(update={"answer": original})}
    )
    data = data.model_copy(update={"cases": [case]})
    review = RAGReview(
        review_input_sha256=RAGReviewInput(case=case, output=row.output).sha256,
        grounded=False,
        provenance={"reviewer": "second-reviewer"},
    )
    report = score_rag(data, [row.model_copy(update={"review": review})])
    assessment = report.results[0].assessment
    assert assessment is not None
    assert assessment.answer_correct is True
    assert assessment.answer_grounded is False
    with pytest.raises(ValueError, match="contradicts"):
        score_rag(data, [reviewed(row, correct=False, case=case)])


def test_tenant_and_reference_changes_cannot_reuse_an_old_observation_or_review() -> None:
    data = dataset("one")
    tenant_input = data.cases[0].input.model_copy(update={"parameters": {"tenant": "A"}})
    case = data.cases[0].model_copy(update={"input": tenant_input, "reference_answer": "Plan A"})
    data = data.model_copy(update={"cases": [case]})
    row = reviewed(attempt("one").model_copy(update={"input": tenant_input}), case=case)
    assert score_rag(data, [row]).summary.metrics["answer.correct"].mean == 1.0
    changed_input = tenant_input.model_copy(update={"parameters": {"tenant": "B"}})
    changed = data.model_copy(update={"cases": [case.model_copy(update={"input": changed_input})]})
    with pytest.raises(ValueError, match="complete dataset input"):
        score_rag(changed, [row])
    changed = data.model_copy(
        update={"cases": [case.model_copy(update={"reference_answer": "Plan B"})]}
    )
    with pytest.raises(ValueError, match="different inputs, references, or outputs"):
        score_rag(changed, [row])
    changed = data.model_copy(
        update={"cases": [case.model_copy(update={"judgments": RAGJudgments(relevance={"x": 1})})]}
    )
    with pytest.raises(ValueError, match="different inputs, references, or outputs"):
        score_rag(changed, [row])


def test_large_finite_native_scores_do_not_overflow_the_summary() -> None:
    rows = [reviewed(attempt("one"), value=1e308), reviewed(attempt("two"), value=1e308)]
    summary = score_rag(dataset("one", "two"), rows).summary.metrics["custom.rubric"]
    assert summary.mean == 1e308
    assert summary.stddev == 0.0
    rows[1] = reviewed(attempt("two"), value=-1e308)
    summary = score_rag(dataset("one", "two"), rows).summary.metrics["custom.rubric"]
    assert summary.mean == 0.0
    assert summary.stddev == 1e308


def test_scoring_detaches_inputs_and_revalidates_nested_mutation() -> None:
    data = dataset("one")
    row = reviewed(attempt("one"))
    config = RAGEvalConfig(system={"index": {"version": "v1"}})
    row = row.model_copy(update={"system": config.system})
    report = score_rag(data, [row], config=config)
    data.cases[0].judgments.relevance["invoice"] = 0
    config.system["index"] = "modified"
    assert row.output is not None and row.output.final.retrieval is not None
    row.output.final.retrieval.ids.append("invoice")
    assert report.dataset.cases[0].judgments.relevance["invoice"] == 1
    assert report.config.system == {"index": {"version": "v1"}}
    with pytest.raises(ValidationError, match="unique"):
        score_rag(dataset("one"), [row])


def test_saved_outputs_cannot_be_redescribed_under_changed_system_configuration() -> None:
    data = dataset("one")
    config = RAGEvalConfig(system={"retriever": "index-A", "top_k": 10})
    row = attempt("one").model_copy(update={"system": config.system})
    assert score_rag(data, [row], config=config).status == "complete"
    changed = config.model_copy(update={"system": {"retriever": "index-B", "top_k": 10}})
    with pytest.raises(ValueError, match="system settings"):
        score_rag(data, [row], config=changed)
    metric_change = config.model_copy(update={"k": 1, "gates": []})
    assert score_rag(data, [row], config=metric_change).status == "complete"


@pytest.mark.parametrize("boundary", ["input", "system"])
@pytest.mark.parametrize("recorded,expected", [(True, 1), (1, True), (False, 0), (1, 1.0)])
def test_json_identity_preserves_scalar_types(
    boundary: str, recorded: bool | int | float, expected: bool | int | float
) -> None:
    data = dataset("one")
    row = attempt("one")
    config = RAGEvalConfig()
    if boundary == "input":
        recorded_input = row.input.model_copy(update={"parameters": {"nested": {"v": recorded}}})
        expected_input = row.input.model_copy(update={"parameters": {"nested": {"v": expected}}})
        row = row.model_copy(update={"input": recorded_input})
        data.cases[0] = data.cases[0].model_copy(update={"input": expected_input})
    else:
        row = row.model_copy(update={"system": {"nested": {"v": recorded}}})
        config = RAGEvalConfig(system={"nested": {"v": expected}})
    with pytest.raises(ValueError, match="input differs|system settings differ"):
        score_rag(data, [row], config=config)


def test_input_matching_ignores_object_key_order_without_reordering_saved_values() -> None:
    data = dataset("one")
    expected = {"a": {"x": 1, "y": [2, 3]}, "b": True}
    recorded = {"b": True, "a": {"y": [2, 3], "x": 1}}
    data.cases[0] = data.cases[0].model_copy(
        update={"input": data.cases[0].input.model_copy(update={"parameters": expected})}
    )
    row = attempt("one")
    row = row.model_copy(
        update={"input": row.input.model_copy(update={"parameters": recorded}), "system": recorded}
    )
    config = RAGEvalConfig.model_validate({"system": expected})
    report = score_rag(data, [row], config=config)
    assert report.status == "complete"
    assert list(report.dataset.cases[0].input.parameters) == ["a", "b"]
    assert list(report.results[0].attempt.input.parameters) == ["b", "a"]
    assert list(report.results[0].attempt.system) == ["b", "a"]
    changed = row.input.model_copy(update={"parameters": {"b": True, "a": {"y": [3, 2], "x": 1}}})
    with pytest.raises(ValueError, match="input differs"):
        score_rag(data, [row.model_copy(update={"input": changed})], config=config)


def test_save_load_roundtrip_has_no_overwrite_or_leftover_temporary_file(tmp_path: Path) -> None:
    report = score_rag(dataset("one"), [reviewed(attempt("one"))])
    path = tmp_path / "report.json"
    save_rag_report(report, path)
    before = path.read_bytes()
    assert load_rag_report(path) == report
    assert RAGReport.model_validate_json(before) == report
    with pytest.raises(FileExistsError):
        save_rag_report(report, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("changed", ["hash", "summary", "assessment", "config", "attempt"])
def test_saved_reports_recompute_hashes_assessments_and_summary(
    tmp_path: Path, changed: str
) -> None:
    report = score_rag(dataset("one"), [reviewed(attempt("one"))])
    if changed == "hash":
        report = report.model_copy(update={"dataset_sha256": "a" * 64})
    elif changed == "summary":
        metric = report.summary.metrics["answer.correct"]
        report.summary.metrics["answer.correct"] = metric.model_copy(update={"mean": 0.0})
    elif changed == "assessment":
        result = report.results[0]
        assert result.assessment is not None
        report.results[0] = result.model_copy(
            update={"assessment": result.assessment.model_copy(update={"answer_correct": False})}
        )
    elif changed == "config":
        report = report.model_copy(update={"config": RAGEvalConfig(k=1)})
    else:
        result = report.results[0]
        changed_attempt = result.attempt.model_copy(update={"duration_seconds": 999.0})
        report.results[0] = result.model_copy(update={"attempt": changed_attempt})
    path = tmp_path / "changed.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from retained records"):
        load_rag_report(path)
    with pytest.raises(ValueError, match="differ from retained records"):
        save_rag_report(report, tmp_path / "new.json")
    assert not (tmp_path / "new.json").exists()


def test_json_roundtrip_preserves_unknowns_and_rendering_names_evidence_limits(
    tmp_path: Path,
) -> None:
    report = score_rag(
        dataset("one", "two"),
        [attempt("one")],
        config=RAGEvalConfig(gates=[RAGGate(metric="answer.correct", minimum=1.0)]),
    )
    path = tmp_path / "report.json"
    save_rag_report(report, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["summary"]["metrics"]["answer.correct"]["mean"] is None
    rendered = render_rag_report(load_rag_report(path))
    assert "caller-declared" in rendered
    assert "independent task samples" in rendered
    assert "Diagnostic attempts excluded" in rendered
    assert "Unknown quality is not zero or success" in rendered
    assert "**Execution status:** `incomplete`" in rendered
