"""Check the probe tooling; controlled answers do not establish model quality."""

import json
from pathlib import Path

import httpx
import pytest
from test_contract import sample_result

from benchmarks import typed_judgments as judgments
from kayak import Client, DecisionRequest
from kayak.decisions import answer_from_scores
from kayak.runtime._preparation import prepare_texts


def test_corpus_labels_include_negative_and_ambiguous_cases() -> None:
    probes = [
        judgments.Probe.model_validate_json(line)
        for line in judgments.SUITE.read_bytes().splitlines()
    ]
    assert len(probes) == len({probe.id for probe in probes}) == 12
    for kind in ("noul", "score"):
        group = [probe for probe in probes if probe.kind == kind]
        assert sum(probe.expected is None for probe in group) == 2
        assert sum(probe.expected is not None for probe in group) == 4
    assert probes[1].expected == "false"
    assert next(probe for probe in probes if probe.id == "score-boundary").expected == "2"


def test_pinned_noul_strings_and_score_meaning() -> None:
    probe = judgments.Probe(
        id="noul",
        kind="noul",
        state="  Context.\n",
        instructions="  Is it true? \n",
        criteria=None,
        expected=None,
        note="Ambiguous example",
    )
    assert prepare_texts(probe.request()) == [
        "Context.\n\nIs it true?",
        "false: No. This is false: Is it true?",
        "true: Yes. This is true: Is it true?",
    ]
    assert judgments.decode(probe, answer_from_scores(["false", "true"], [0.0, 0.0])) == {
        "type": "noul",
        "noul": 0.5,
    }
    custom = judgments.Probe.model_validate(
        {**probe.model_dump(), "criteria": {"true": "  Oui.\n"}}
    )
    assert prepare_texts(custom.request())[-1] == "true:   Oui.\n"
    score = judgments.Probe.model_validate(
        {**probe.model_dump(), "kind": "score", "criteria": ["  Low\n", "Medium", "High"]}
    )
    assert prepare_texts(score.request())[1:] == ["  Low\n", "Medium", "High"]
    answer = answer_from_scores(["0", "1", "2"], [0.0, 0.0, 0.0])
    decoded = judgments.decode(score, answer)
    assert decoded["score"] == 1.0  # Expected index, although the first mode is index zero.
    assert decoded["confidence"] == 0.0
    assert decoded["legend"] == {"0": "  Low\n", "1": "Medium", "2": "High"}
    nonuniform = answer_from_scores(["0", "1", "2"], [0.0, 1.0, 2.0])
    reordered = nonuniform.model_copy(
        update={"probabilities": dict(reversed(list(nonuniform.probabilities.items())))}
    )
    assert judgments.decode(score, reordered) == judgments.decode(score, nonuniform)


def test_reference_requires_exact_pinned_source_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "schema.py"
    path.write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="SHA-256"):
        judgments.check_reference([], path)


def test_probe_evaluation_keeps_failures_and_ambiguity() -> None:
    probes = [
        judgments.Probe.model_validate_json(line)
        for line in judgments.SUITE.read_bytes().splitlines()
    ]
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = DecisionRequest.model_validate_json(request.content)
        assert set(json.loads(request.content)) == {"state", "questions"}
        if calls == 1:
            raise httpx.ConnectError("unavailable", request=request)
        keys = list(body.questions["probe"].criteria)
        result = sample_result().model_copy(
            update={"answers": {"probe": answer_from_scores(keys, [0.0] * len(keys))}}
        )
        return httpx.Response(200, content=result.model_dump_json())

    with Client(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        observations = judgments.evaluate(probes, client)
    assert calls == len(observations) == len(probes)
    assert observations[0]["error"] == "TransportError"
    assert observations[0]["correct"] is False
    assert observations[1]["label"] == "true"  # Pinned diagnostic Noul tie rule.
    assert observations[1]["correct"] is False
    for observation, probe in zip(observations, probes, strict=True):
        if probe.expected is None:
            assert observation["correct"] is None
