"""Challenge saved-output diagnostics using independent two-dimensional arithmetic."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import compare_encoders, encoder_probe
from benchmarks.compare_encoders import Snapshot, compare, ranking_summary, vector_difference
from benchmarks.encoder_probe import (
    Probe,
    TokenCapture,
    Tolerances,
    bind_tokens,
    json_bytes,
    prepare_probe,
    read_record,
)
from kayak.bundle import ModelSpec
from kayak.eval import load_suite

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "benchmarks/data/encoder-comparison"


def probe_fixture() -> Probe:
    return read_record(FIXTURES / "probe.json", Probe)


def reference_fixture() -> Snapshot:
    return read_record(FIXTURES / "reference.json", Snapshot)


def test_independent_vector_rank_and_margin_expectations() -> None:
    result = compare(
        probe_fixture(), reference_fixture(), read_record(FIXTURES / "candidate.json", Snapshot)
    )
    # JSON is the public artifact; inspect it independently of implementation types.
    report = json.loads(json.dumps(result, allow_nan=False))
    assert report["evidence_kind"] == "synthetic_fixture"
    assert report["comparison_passed"] is False
    assert report["within_declared_numeric_tolerances"] is False
    assert report["changed_choices"] == ["case-a"]
    assert report["maximum_embedding_error"] == pytest.approx(0.8)
    assert report["maximum_score_error"] == pytest.approx(0.8)
    changed = report["vectors"][0]["pooled"]
    assert changed["cosine_similarity"] == pytest.approx(0.6)
    assert changed["root_mean_square_error"] == pytest.approx(math.sqrt(0.4))
    assert changed["reference_norm"] == changed["candidate_norm"] == 1.0
    first, second = report["cases"]
    assert first["reference"]["ranking"] == ["left", "right"]
    assert first["candidate"]["ranking"] == ["right", "left"]
    assert first["reference"]["gold_rank"] == 1
    assert first["candidate"]["gold_rank"] == 2
    assert first["reference"]["top_two_margin"] == 1.0
    assert first["candidate"]["top_two_margin"] == pytest.approx(0.2)
    assert first["candidate"]["gold_margin"] == pytest.approx(-0.2)
    assert second["reference"] == second["candidate"]


def test_equal_artifacts_pass_without_changing_inputs() -> None:
    probe, reference = probe_fixture(), reference_fixture()
    before = (probe.model_dump_json(), reference.model_dump_json())
    report = compare(probe, reference, reference)
    assert report["comparison_passed"] is True
    assert report["maximum_embedding_error"] == report["maximum_score_error"] == 0.0
    assert report["changed_choices"] == []
    assert before == (probe.model_dump_json(), reference.model_dump_json())


def test_choice_change_below_numeric_tolerance_is_still_reported() -> None:
    probe = probe_fixture()
    reference, candidate = reference_fixture(), reference_fixture()
    reference.cases[0].scores = {"left": 0.5, "right": 0.5}
    candidate.cases[0].scores = {"left": 0.5, "right": 0.5000001}
    report = compare(probe, reference, candidate)
    assert report["within_declared_numeric_tolerances"] is True
    assert report["changed_choices"] == ["case-a"]
    assert report["comparison_passed"] is False


def test_ties_follow_candidate_order_and_optional_gold_stays_unknown() -> None:
    report = ranking_summary({"a": 1.0, "z": 1.0}, ["z", "a"], None)
    assert report["choice"] == "z"
    assert report["ranking"] == ["z", "a"]
    assert report["top_two_margin"] == 0.0
    assert report["gold_rank"] is report["gold_margin"] is None


def test_finite_extreme_vectors_do_not_overflow_the_rms_calculation() -> None:
    report = vector_difference([7e307, 7e307], [-7e307, -7e307])
    assert report["root_mean_square_error"] == 1.4e308
    assert report["cosine_similarity"] == pytest.approx(-1.0)


@pytest.mark.parametrize("excess", [0.0, 0.0001])
def test_declared_tolerance_boundary(excess: float) -> None:
    probe = read_record(FIXTURES / "texts.json", Probe)
    probe.tolerances = Tolerances(embedding_atol=0.25, score_atol=0.25)
    capture = read_record(FIXTURES / "tokens.json", TokenCapture)
    capture.text_spec_sha256 = probe.text_spec_sha256
    probe = bind_tokens(probe, capture)
    reference, candidate = reference_fixture(), reference_fixture()
    reference.probe_sha256 = candidate.probe_sha256 = probe.sha256
    candidate.embeddings[0].pooled = [1.25 + excess, 0.0]
    candidate.cases[0].scores["left"] = 0.75
    report = compare(probe, reference, candidate)
    assert report["changed_choices"] == []
    assert report["comparison_passed"] is (excess == 0)


@pytest.mark.parametrize(
    "corruption",
    [
        "probe",
        "model",
        "tokens",
        "missing",
        "extra",
        "duplicate",
        "width",
        "nan",
        "zero",
        "zero_normalized",
        "normalization",
        "head_implementation",
        "scale",
        "candidate_labels",
        "missing_case",
        "duplicate_case",
        "evidence_kind",
        "huge_scores",
    ],
)
def test_corrupt_or_incomparable_snapshot_cannot_pass(corruption: str) -> None:
    candidate = reference_fixture()
    if corruption == "probe":
        candidate.probe_sha256 = "0" * 64
    elif corruption == "model":
        candidate.model_fingerprint = "0" * 64
    elif corruption == "tokens":
        candidate.embeddings[0].token_ids = [999]
    elif corruption == "missing":
        candidate.embeddings.pop()
    elif corruption == "extra":
        row = candidate.embeddings[0].model_copy(deep=True)
        row.id = "extra-sequence"
        candidate.embeddings.append(row)
    elif corruption == "duplicate":
        candidate.embeddings.append(candidate.embeddings[0])
    elif corruption == "width":
        candidate.embeddings[0].pooled.append(0.0)
    elif corruption == "nan":
        candidate.embeddings[0].pooled[0] = float("nan")
    elif corruption == "zero":
        candidate.embeddings[0].pooled = [0.0, 0.0]
    elif corruption == "zero_normalized":
        candidate.embeddings[0].normalized = [0.0, 0.0]
    elif corruption == "normalization":
        candidate.embeddings[0].normalized = [0.0, 1.0]
    elif corruption == "head_implementation":
        candidate.heads_implementation_sha256 = "0" * 64
    elif corruption == "scale":
        candidate.score_scale = 2.0
    elif corruption == "candidate_labels":
        candidate.cases[0].scores = {"left": 1.0, "unknown": 0.0}
    elif corruption == "missing_case":
        candidate.cases.pop()
    elif corruption == "duplicate_case":
        candidate.cases.append(candidate.cases[0])
    elif corruption == "evidence_kind":
        candidate.evidence_kind = "encoder_diagnostic"
    else:
        candidate.cases[0].scores = {"left": 1e308, "right": -1e308}
    with pytest.raises(ValueError):
        compare(probe_fixture(), reference_fixture(), candidate)


def test_text_mode_still_requires_identical_actual_token_rows() -> None:
    candidate = reference_fixture()
    candidate.input_mode = "text"
    assert compare(probe_fixture(), reference_fixture(), candidate)["comparison_passed"] is True
    candidate.embeddings[0].token_ids.append(123)
    with pytest.raises(ValueError, match="token IDs"):
        compare(probe_fixture(), reference_fixture(), candidate)


def test_unbound_probe_never_accepts_saved_outputs() -> None:
    probe = read_record(FIXTURES / "texts.json", Probe)
    candidate = reference_fixture()
    candidate.probe_sha256 = probe.sha256
    with pytest.raises(ValueError, match="bind"):
        compare(probe, candidate, candidate)


@pytest.mark.parametrize("corruption", ["text", "revision", "duplicate", "missing", "overflow"])
def test_binding_rejects_wrong_text_identity_or_token_rows(corruption: str) -> None:
    probe = read_record(FIXTURES / "texts.json", Probe)
    capture = read_record(FIXTURES / "tokens.json", TokenCapture)
    if corruption == "text":
        capture.sequences[0].text_sha256 = "0" * 64
    elif corruption == "revision":
        capture.tokenizer_revision = "b" * 40
    elif corruption == "duplicate":
        capture.sequences.append(capture.sequences[0])
    elif corruption == "missing":
        capture.sequences.pop()
    else:
        capture.sequences[0].token_ids = [1] * 9  # The independent fixture's limit is eight.
    with pytest.raises(ValueError):
        bind_tokens(probe, capture)


def test_preparation_preserves_recipe_and_keeps_ids_and_labels_out_of_encoder_text() -> None:
    suite = load_suite(FIXTURES / "suite.json")
    spec = read_record(FIXTURES / "manifest.json", ModelSpec)
    probe = prepare_probe(suite, spec, Tolerances(embedding_atol=0.001, score_atol=0.001))
    assert [row.text for row in probe.sequences] == [
        "Fictional input alpha.\n\nPick one fictional direction.",
        "The left direction.",
        "The right direction.",
        "Fictional input beta.\n\nPick one fictional direction.",
    ]
    assert [case.id for case in probe.cases] == ["case-a", "case-b"]
    assert [case.gold for case in probe.cases] == ["left", "right"]
    assert probe.model == spec
    assert probe.token_capture is None
    assert all("case-a" not in row.text and "case-b" not in row.text for row in probe.sequences)
    suite.split = "test"
    with pytest.raises(ValueError, match="reserve"):
        prepare_probe(suite, spec, probe.tolerances)


def test_preparation_and_binding_are_reproducible_and_do_not_replace_files(tmp_path: Path) -> None:
    texts, bound = tmp_path / "texts.json", tmp_path / "probe.json"
    arguments = [
        "prepare",
        "--suite",
        str(FIXTURES / "suite.json"),
        "--manifest",
        str(FIXTURES / "manifest.json"),
        "--synthetic-fixture",
        "--embedding-atol",
        "0.001",
        "--score-atol",
        "0.001",
        "--output",
        str(texts),
    ]
    assert encoder_probe.main(arguments) == 0
    assert texts.read_bytes() == (FIXTURES / "texts.json").read_bytes()
    assert (
        encoder_probe.main(
            [
                "bind",
                "--probe",
                str(texts),
                "--tokens",
                str(FIXTURES / "tokens.json"),
                "--output",
                str(bound),
            ]
        )
        == 0
    )
    assert bound.read_bytes() == (FIXTURES / "probe.json").read_bytes()
    assert encoder_probe.main(arguments) == 2
    assert texts.read_bytes() == (FIXTURES / "texts.json").read_bytes()


def test_cli_retains_differences_but_rejects_bad_identity_before_writing(tmp_path: Path) -> None:
    output = tmp_path / "comparison.json"
    arguments = [
        "--probe",
        str(FIXTURES / "probe.json"),
        "--reference",
        str(FIXTURES / "reference.json"),
        "--candidate",
        str(FIXTURES / "candidate.json"),
        "--output",
        str(output),
    ]
    assert compare_encoders.main(arguments) == 1
    report = json.loads(output.read_bytes())
    assert report["changed_choices"] == ["case-a"]
    assert set(report["normalized_artifact_sha256"]) == {"reference", "candidate"}
    original = output.read_bytes()
    assert compare_encoders.main(arguments) == 2
    assert output.read_bytes() == original
    bad = reference_fixture()
    bad.model_fingerprint = "0" * 64
    candidate_path = tmp_path / "bad.json"
    candidate_path.write_bytes(json_bytes(bad.model_dump(mode="json")))
    invalid_output = tmp_path / "must-not-exist.json"
    assert (
        compare_encoders.main(
            [
                "--probe",
                str(FIXTURES / "probe.json"),
                "--reference",
                str(FIXTURES / "reference.json"),
                "--candidate",
                str(candidate_path),
                "--output",
                str(invalid_output),
            ]
        )
        == 2
    )
    assert not invalid_output.exists()


def test_probe_and_review_tools_import_without_optional_runtimes_or_network() -> None:
    # A fresh interpreter avoids another test's cached torch/transformers imports.
    code = """
import builtins
import socket
import sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'huggingface_hub', 'vllm', 'numpy'}:
        raise AssertionError('optional runtime imported: ' + name)
    return original(name, *args, **kwargs)
def forbidden(*args, **kwargs):
    raise AssertionError('network attempted')
builtins.__import__ = guarded
socket.socket.connect = forbidden
from benchmarks.encoder_probe import Probe, read_record
from benchmarks.compare_encoders import Snapshot, compare
from examples.prepare_support import main
from pathlib import Path
root = Path('benchmarks/data/encoder-comparison')
probe = read_record(root / 'probe.json', Probe)
snapshot = read_record(root / 'reference.json', Snapshot)
assert compare(probe, snapshot, snapshot)['comparison_passed']
assert main(['audit', '--suites', 'examples/suites/support_pilot.json']) == 0
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
