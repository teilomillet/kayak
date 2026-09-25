"""User-owned suites and typed case inspection, independent of model quality."""

import json
import subprocess
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import assert_type

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_eval_runner import MODEL

from kayak import Choice, DecisionRequest, DecisionResult
from kayak.decisions import answer_from_scores
from kayak.eval import (
    BenchmarkResult,
    CaseComparison,
    Comparison,
    Example,
    Prediction,
    PredictionSet,
    Suite,
    benchmark,
    compare,
    evaluate,
    export_predictions,
    load_suite,
    render_benchmark,
    render_comparison,
    write_benchmark,
)


def review_suite() -> Suite:
    return Suite(
        name="case-review",
        split="dev",
        question=Choice(
            instructions="Select team", criteria={"a": "Alpha", "b": "Beta", "c": "Gamma"}
        ),
        examples=[
            Example(id=str(index), text=f"request {index}", label=label)
            for index, label in enumerate("abacc")
        ],
    )


def external(suite: Suite, choices: str) -> PredictionSet:
    return PredictionSet(
        system="controlled fixture",
        method="fixed choices; no model quality measured",
        suite=suite,
        predictions=[
            Prediction(id=case.id, choice=choice)
            for case, choice in zip(suite.examples, choices, strict=True)
        ],
        metadata={"evidence_kind": "mock"},
    )


def test_suite_roundtrip_preserves_order_text_provenance_and_digest(tmp_path: Path) -> None:
    suite = review_suite()
    suite.question = Choice(
        instructions=suite.question.instructions,
        criteria={"b": "  Béta  ", "a": "Alpha", "c": "Gamma"},
    )
    suite.provenance = {"source": "hand-reviewed fixture"}
    suite.examples[0].text = "  Unicode café\nwith whitespace  "
    path = tmp_path / "suite.json"
    path.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    loaded = load_suite(path)
    assert loaded == suite and loaded.sha256 == suite.sha256
    assert list(loaded.question.criteria) == ["b", "a", "c"]
    loaded.question.criteria["a"] = "changed"
    assert suite.question.criteria["a"] == "Alpha"
    assert load_suite(path) == suite


@pytest.mark.parametrize(
    "corruption",
    [
        "unknown_label",
        "duplicate_id",
        "empty",
        "blank_text",
        "invalid_text_type",
        "extra",
        "duplicate_key",
        "syntax",
    ],
)
def test_suite_errors_are_actionable_without_printing_private_input(
    tmp_path: Path, corruption: str
) -> None:
    payload = review_suite().model_dump()
    payload["examples"][0]["text"] = "PRIVATE_CUSTOMER_CONTENT"
    if corruption == "unknown_label":
        payload["examples"][1]["label"] = "missing"
    elif corruption == "duplicate_id":
        payload["examples"][1]["id"] = "0"
    elif corruption == "empty":
        payload["examples"] = []
    elif corruption == "blank_text":
        payload["examples"][1]["text"] = " "
    elif corruption == "invalid_text_type":
        payload["examples"][0]["text"] = ["PRIVATE_CUSTOMER_CONTENT"]
    elif corruption == "extra":
        payload["private_field"] = "PRIVATE_CUSTOMER_CONTENT"
    raw = json.dumps(payload)
    if corruption == "duplicate_key":
        raw = raw.replace('"a": "Alpha"', '"a": "Alpha", "a": "overwritten"')
    elif corruption == "syntax":
        raw += "trailing garbage"
    path = tmp_path / "invalid.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="invalid suite") as error:
        load_suite(path)
    assert "PRIVATE_CUSTOMER_CONTENT" not in str(error.value)
    assert "PRIVATE_CUSTOMER_CONTENT" not in "".join(traceback.format_exception(error.value))
    assert str(path) in str(error.value)
    if corruption == "extra":
        assert "private_field" in str(error.value)
    if corruption == "duplicate_key":
        assert "duplicate JSON" in str(error.value)
    assert list(tmp_path.iterdir()) == [path]


def test_native_and_external_comparisons_agree_by_id_and_first_attempt(tmp_path: Path) -> None:
    suite = review_suite()
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(suite.model_dump_json())
    original = suite_path.read_bytes()
    # Independent intended outcomes: correct, fixed, regressed, wrong/changed, wrong/unchanged.
    baseline_choices, candidate_choices = "aaaab", "abbbb"
    received: list[str] = []

    class Backend:
        def __init__(self, choices: str) -> None:
            self.choices = dict(zip((case.text for case in suite.examples), choices, strict=True))
            self.seen: set[str] = set()

        def decide(
            self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
        ) -> DecisionResult:
            request = DecisionRequest.model_validate({"state": state, "questions": questions})
            received.append(state)
            # Deliberately disagree on repeats; analysis must use the first attempt only.
            choice = self.choices[state] if state not in self.seen else "c"
            self.seen.add(state)
            keys = list(request.questions["intent"].criteria)
            return DecisionResult(
                model=MODEL,
                answers={
                    "intent": answer_from_scores(keys, [float(label == choice) for label in keys])
                },
                input_tokens=1,
            )

    before, after = tmp_path / "before", tmp_path / "after"
    evaluate(
        Backend(baseline_choices),
        load_suite(suite_path),
        output=before,
        warmups=0,
        repeats=2,
        config={"evidence_kind": "mock"},
    )
    evaluate(
        Backend(candidate_choices),
        load_suite(suite_path),
        output=after,
        warmups=0,
        repeats=2,
        config={"evidence_kind": "mock"},
    )
    result = compare(before, after)
    assert_type(result, Comparison)
    assert_type(result["cases"], list[CaseComparison])
    assert [case["id"] for case in result["cases"]] == list("01234")
    assert [case["outcome"] for case in result["cases"]] == [
        "both_correct",
        "fixed",
        "regressed",
        "both_wrong",
        "both_wrong",
    ]
    assert result["paired_outcomes"] == {
        "both_correct": 1,
        "fixed": 1,
        "regressed": 1,
        "both_wrong": 2,
    }
    assert result["cases"][2] == {
        "id": "2",
        "text": "request 2",
        "expected": "a",
        "baseline_choice": "a",
        "candidate_choice": "b",
        "outcome": "regressed",
    }
    assert result["quality_delta"]["accuracy"] == 0

    imported_before = external(suite, baseline_choices)
    imported_after = external(suite, candidate_choices)
    imported_after.predictions.reverse()  # Exchange order must not change pairing by ID.
    imported = benchmark({"before": imported_before, "after": imported_after})
    assert_type(imported, BenchmarkResult)
    assert_type(imported["runs"]["before"]["metrics"]["accuracy"]["value"], float | None)
    assert imported["comparisons"][0]["cases"] == result["cases"]
    assert imported["comparisons"][0]["paired_outcomes"] == result["paired_outcomes"]
    exported_before = export_predictions(
        before, tmp_path / "before.json", system="fixture", method="same"
    )
    exported_after = export_predictions(
        after, tmp_path / "after.json", system="fixture", method="same"
    )
    assert (
        benchmark({"before": exported_before, "after": exported_after})["comparisons"][0]["cases"]
        == result["cases"]
    )
    rendered = render_comparison(result)
    assert "#### Regressions (1)" in rendered and "evidence: mock" in rendered
    write_benchmark(imported, tmp_path / "review")
    assert "| 2 | request 2 | a | a | b |" in (tmp_path / "review/benchmark.md").read_text()
    assert (
        json.loads((tmp_path / "review/benchmark.json").read_text())["comparisons"][0]["cases"]
        == result["cases"]
    )
    assert suite_path.read_bytes() == original
    assert set(received) == {case.text for case in suite.examples}


@given(
    st.lists(
        st.tuples(st.sampled_from("abc"), st.sampled_from("abc"), st.sampled_from("abc")),
        min_size=1,
        max_size=20,
    )
)
def test_paired_case_accounting_preserves_every_example(rows: list[tuple[str, str, str]]) -> None:
    suite = review_suite()
    suite.examples = [
        Example(id=str(i), text=str(i), label=gold) for i, (gold, _, _) in enumerate(rows)
    ]
    result = benchmark(
        {
            "before": external(suite, "".join(row[1] for row in rows)),
            "after": external(suite, "".join(row[2] for row in rows)),
        }
    )
    pair = result["comparisons"][0]
    counts = pair["paired_outcomes"]
    assert sum(counts.values()) == len(rows) == len(pair["cases"])
    correct_before = sum(gold == before for gold, before, _ in rows)
    correct_after = sum(gold == after for gold, _, after in rows)
    assert counts["fixed"] - counts["regressed"] == correct_after - correct_before
    assert pair["accuracy_delta"] == pytest.approx((correct_after - correct_before) / len(rows))


@pytest.mark.parametrize("change", ["labels", "order", "text", "incomplete", "method"])
def test_ineligible_inputs_never_produce_case_comparisons(change: str) -> None:
    before = external(review_suite(), "aaaab")
    after = before.model_copy(deep=True)
    if change == "labels":
        after.suite.examples[0].label = "b"
    elif change == "order":
        after.suite.question = Choice(
            instructions=after.suite.question.instructions,
            criteria=dict(reversed(list(after.suite.question.criteria.items()))),
        )
    elif change == "text":
        after.suite.examples[0].text = "different"
    elif change == "incomplete":
        after.predictions.pop()
    else:
        after.method = "changed"
    pair = benchmark({"before": before, "after": after})["comparisons"][0]
    assert not pair["eligible"] and pair["exclusions"]
    assert "cases" not in pair and "paired_outcomes" not in pair


def test_wording_change_is_explicit_and_does_not_mutate_baseline() -> None:
    before = external(review_suite(), "aaaab")
    original = before.model_dump_json()
    after = before.model_copy(deep=True)
    after.suite.question.criteria["a"] = "A new description"
    assert not benchmark({"before": before, "after": after})["comparisons"][0]["eligible"]
    pair = benchmark({"before": before, "after": after}, allow_recipe_change=True)["comparisons"][0]
    assert pair["eligible"] and pair["recipe_changed"]
    assert before.model_dump_json() == original


def test_reports_bound_display_and_escape_case_text_without_losing_raw_evidence() -> None:
    suite = review_suite()
    text = "<script> ![image](https://example.test) | line\n" + "x" * 300
    suite.examples = [Example(id=str(i), text=text, label="b") for i in range(12)]
    result = benchmark({"before": external(suite, "b" * 12), "after": external(suite, "a" * 12)})
    rendered = render_benchmark(result)
    assert "Regressions (12)" in rendered and "Showing 10 of 12" in rendered
    assert "<script>" not in rendered and "![image]" not in rendered
    assert "\\| line" in rendered and "…" in rendered
    cases = result["comparisons"][0]["cases"]
    assert len(cases) == 12 and all(case["text"] == text for case in cases)


def test_previous_saved_report_without_cases_still_renders_metrics_and_counts() -> None:
    result = benchmark(
        {"before": external(review_suite(), "aaaab"), "after": external(review_suite(), "abbbb")}
    )
    # Schema 1 reports previously retained aggregate evidence but no individual cases.
    del result["comparisons"][0]["cases"]
    rendered = render_benchmark(json.loads(json.dumps(result)))
    assert "Fixed 1; regressed 1; both correct 1; both wrong 2." in rendered
    assert "| accuracy | higher | 0.400000 | 0.400000 |" in rendered
    assert "Case details unavailable in this saved report." in rendered
    assert "Regressions (0)" not in rendered


def test_own_dataset_and_custom_metric_example_needs_only_base_dependencies(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from pathlib import Path; "
            "from examples.benchmark_classifiers import run; "
            "from kayak.eval import load_suite; "
            "run(Path(sys.argv[1]), suite=load_suite('examples/suites/support.json')); "
            "assert not {'torch', 'transformers', 'fastapi', 'huggingface_hub'} "
            "& sys.modules.keys()",
            str(tmp_path / "review"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    result = json.loads((tmp_path / "review/benchmark.json").read_bytes())
    assert result["runs"]["constant"]["examples"] == 8
    assert result["runs"]["word_overlap"]["metrics"]["example_error_cost"]["value"] == 1 / 8
    assert result["comparisons"][0]["paired_outcomes"]["fixed"] == 4
    assert result["comparisons"][0]["paired_outcomes"]["regressed"] == 0
