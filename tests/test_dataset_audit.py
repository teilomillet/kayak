"""Independent text/group/pair examples for the offline dataset audit artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from benchmarks import audit_dataset
from kayak import Choice
from kayak.eval import Example, Suite


def synthetic_suites() -> list[Suite]:
    question = Choice(instructions="Select intent", criteria={"a": "Alpha", "b": "Beta"})
    return [
        Suite(
            name="independent-fixture",
            split="train",
            question=question,
            examples=[
                Example(id="t2", text="CAFÉ  fee?", label="a"),
                Example(id="t1", text="café fee?", label="a"),
                Example(id="t3", text="café fee?", label="b"),
                Example(id="t4", text="unique", label="a"),
            ],
            provenance={"revision": "synthetic-v1", "license": "fixture"},
        ),
        Suite(
            name="independent-fixture",
            split="dev",
            question=question,
            examples=[
                Example(id="d1", text="café fee?", label="a"),
                Example(id="d2", text="café-fee", label="b"),
            ],
        ),
        Suite(
            name="independent-fixture",
            split="test",
            question=question,
            examples=[Example(id="s1", text="unrelated", label="b")],
        ),
    ]


def test_diagnostic_counts_and_rows_survive_json_without_changing_suites() -> None:
    suites = synthetic_suites()
    original = [suite.model_dump() for suite in suites]
    # Serialization is part of the command's public artifact boundary.
    report = json.loads(json.dumps(audit_dataset.audit_suites(suites), allow_nan=False))
    assert [suite.model_dump() for suite in suites] == original
    assert report["suites"]["train"]["suite_sha256"] == suites[0].sha256
    assert report["suites"]["train"]["provenance"] == suites[0].provenance
    exact = report["analyses"]["exact"]
    within = exact["within"]["train"]
    assert within["row_count"] == 4
    assert within["distinct_key_count"] == 3
    assert within["duplicate_group_count"] == 1
    assert within["duplicate_row_count"] == 2
    assert within["duplicate_pair_count"] == 1
    assert within["conflicting_label_group_count"] == 1
    assert within["groups"] == [
        {
            "normalized_text": "café fee?",
            "rows": [
                {"id": "t1", "text": "café fee?", "label": "a"},
                {"id": "t3", "text": "café fee?", "label": "b"},
            ],
            "conflicting_labels": True,
        }
    ]
    exact_overlap = exact["between"][0]
    assert (exact_overlap["left_split"], exact_overlap["right_split"]) == ("train", "dev")
    assert exact_overlap["overlap_group_count"] == 1
    assert exact_overlap["left_row_count"] == 2
    assert exact_overlap["right_row_count"] == 1
    assert exact_overlap["row_pair_count"] == 2
    assert exact_overlap["conflicting_label_group_count"] == 1

    folded = report["analyses"]["casefold_whitespace"]
    assert folded["within"]["train"]["duplicate_row_count"] == 3
    assert folded["within"]["train"]["duplicate_pair_count"] == 3
    assert folded["between"][0]["row_pair_count"] == 3

    tokenized = report["analyses"]["unicode_word_tokens"]
    overlap = tokenized["between"][0]
    assert overlap["overlap_group_count"] == 1
    assert overlap["left_row_count"] == 3
    assert overlap["right_row_count"] == 2
    assert overlap["row_pair_count"] == 6
    assert overlap["groups"][0]["normalized_text"] == "café fee"
    assert overlap["groups"][0]["right_rows"] == [
        {"id": "d1", "text": "café fee?", "label": "a"},
        {"id": "d2", "text": "café-fee", "label": "b"},
    ]
    assert tokenized["within"]["dev"]["conflicting_label_group_count"] == 1
    for normalization in report["analyses"].values():
        assert normalization["within"]["test"]["groups"] == []
        assert normalization["between"][1]["overlap_group_count"] == 0
        assert normalization["between"][2]["row_pair_count"] == 0


def test_distinct_keys_remain_separate_and_unicode_casefold_is_explicit() -> None:
    suites = synthetic_suites()
    suites[0].examples.extend(
        [
            Example(id="t5", text="Straße", label="a"),
            Example(id="t6", text="STRASSE", label="a"),
            Example(id="t7", text="!!!", label="b"),
            Example(id="t8", text="???", label="b"),
        ]
    )
    report = json.loads(json.dumps(audit_dataset.audit_suites(suites)))
    folded = report["analyses"]["casefold_whitespace"]["within"]["train"]
    assert folded["duplicate_group_count"] == 2
    assert folded["duplicate_row_count"] == 5
    assert folded["duplicate_pair_count"] == 4
    tokenized = report["analyses"]["unicode_word_tokens"]["within"]["train"]
    assert tokenized["duplicate_group_count"] == 3
    assert tokenized["duplicate_row_count"] == 7
    assert tokenized["duplicate_pair_count"] == 5
    assert [group["normalized_text"] for group in tokenized["groups"]] == [
        "",
        "café fee",
        "strasse",
    ]
    assert tokenized["conflicting_label_group_count"] == 1


def test_cli_loads_all_splits_from_requested_cache_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    suites = {suite.split: suite for suite in synthetic_suites()}
    calls: list[tuple[str, Path]] = []

    def load(*, split: Literal["train", "dev", "test"], cache_dir: Path) -> Suite:
        calls.append((split, cache_dir))
        return suites[split]

    monkeypatch.setattr(audit_dataset, "banking77", load)
    assert audit_dataset.main(["--data-cache-dir", str(tmp_path)]) == 0
    first = capsys.readouterr().out
    assert audit_dataset.main(["--data-cache-dir", str(tmp_path)]) == 0
    second = capsys.readouterr().out
    assert first == second
    assert calls == [(split, tmp_path) for split in ("train", "dev", "test")] * 2
    assert list(json.loads(first)["suites"]) == ["train", "dev", "test"]


def test_missing_cache_fails_without_creating_dataset_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        audit_dataset.main(["--data-cache-dir", str(tmp_path)])
    assert error.value.code == 2
    assert "prepare banking77" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_empty_or_duplicate_split_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="distinct split"):
        audit_dataset.audit_suites([])
    suite = synthetic_suites()[0]
    with pytest.raises(ValueError, match="distinct split"):
        audit_dataset.audit_suites([suite, suite])
