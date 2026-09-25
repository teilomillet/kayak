"""Check actual MaxSim arithmetic and its adapter without learned model weights."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from examples.evaluate_late_interaction import MaxSimRanker, TokenVectors, maxsim
from kayak import RankingRequest
from kayak.eval import RankedOutput, RankingDataset, evaluate_cases, summarize_cases


@pytest.mark.parametrize(
    ("query", "document", "expected"),
    [
        (((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0), (0.0, 1.0)), 2.0),
        (((1.0, 0.0), (0.0, 1.0)), ((1.0, 0.0),), 1.0),
        (((1.0, 0.0),), ((-1.0, 0.0),), -1.0),
        (((1.0, 0.0),), ((1.0, 0.0), (1.0, 0.0)), 1.0),
        (((1.0, 0.0), (1.0, 0.0)), ((1.0, 0.0),), 2.0),
        (((1.0, 0.0), (-1.0, 0.0)), ((1.0, 0.0), (-1.0, 0.0)), 2.0),
    ],
)
def test_maxsim_known_arithmetic(
    query: TokenVectors, document: TokenVectors, expected: float
) -> None:
    assert maxsim(query, document) == expected


@pytest.mark.parametrize(
    ("query", "document"),
    [
        ((), ((1.0, 0.0),)),
        (((1.0, 0.0),), ()),
        (((),), ((),)),
        (((1.0, 0.0),), ((1.0,),)),
        (((1.0, 0.0), (1.0,)), ((1.0, 0.0),)),
        (((math.nan, 0.0),), ((1.0, 0.0),)),
        (((1.0, 0.0),), ((math.inf, 0.0),)),
        (((1.0, 0.0),), ((-math.inf, 0.0),)),
        (((1e308, 0.0),), ((1e308, 0.0),)),
    ],
)
def test_maxsim_rejects_unusable_matrices(query: TokenVectors, document: TokenVectors) -> None:
    with pytest.raises(ValueError):
        maxsim(query, document)


@pytest.fixture
def sample() -> tuple[RankingRequest, MaxSimRanker]:
    request = RankingRequest(
        state="two-token query",
        instructions="Rank by the supplied synthetic vectors.",
        candidates={"partial": "A", "second": "B", "first": "B", "negative": "C"},
    )
    ranker = MaxSimRanker(
        {request.state: ((1.0, 0.0), (0.0, 1.0))},
        {
            "first": ((1.0, 0.0), (0.0, 1.0)),
            "second": ((1.0, 0.0), (0.0, 1.0)),
            "partial": ((1.0, 0.0),),
            "negative": ((-1.0, 0.0),),
        },
    )
    return request, ranker


def test_ranker_preserves_ids_raw_scores_and_submitted_ties(
    sample: tuple[RankingRequest, MaxSimRanker],
) -> None:
    request, ranker = sample
    original = request.model_dump()
    result = ranker.rank(request)
    assert result.ids == ["second", "first", "partial", "negative"]
    assert result.scores == {"partial": 1.0, "second": 2.0, "first": 2.0, "negative": -1.0}
    assert request.model_dump() == original
    assert set(result.model_dump()) == {"ids", "scores", "provenance"}
    result.ids.reverse()
    assert ranker.rank(request).ids == ["second", "first", "partial", "negative"]


def test_ranker_snapshots_caller_mappings() -> None:
    queries = {"query": ((1.0, 0.0),)}
    documents = {"document": ((1.0, 0.0),)}
    ranker = MaxSimRanker(queries, documents)
    queries.clear()
    documents.clear()
    request = RankingRequest(
        state="query", instructions="Score the prepared vectors.", candidates={"document": "text"}
    )
    assert ranker.rank(request).scores == {"document": 1.0}


def test_judgments_change_metrics_without_changing_scoring_or_provenance(
    sample: tuple[RankingRequest, MaxSimRanker],
) -> None:
    request, ranker = sample
    received: list[RankingRequest] = []

    def rank(submitted: RankingRequest) -> RankedOutput:
        received.append(submitted)
        return ranker.rank(submitted)

    def dataset(relevant: list[str], case_id: str) -> RankingDataset:
        return RankingDataset.model_validate(
            {
                "name": "synthetic",
                "kind": "ranking",
                "examples": ["examples/evaluate_late_interaction.py"],
                "k": 2,
                "cases": [{"id": case_id, "request": request.model_dump(), "relevant": relevant}],
            }
        )

    positive = dataset(["first", "second"], "positive-judgment")
    negative = dataset(["negative"], "different-case-and-judgment")
    correct = evaluate_cases(positive, rank=rank)
    incorrect = evaluate_cases(negative, rank=rank)
    assert received == [request, request]
    assert correct[0].result == incorrect[0].result
    assert correct[0].metrics["top1"] == 1.0
    assert incorrect[0].metrics["top1"] == 0.0
    assert summarize_cases(positive, correct)["input_order_top1"] == 0.0
    assert summarize_cases(positive, correct)["recall_at_k"] == 1.0


def test_module_entrypoint_runs_without_inference_imports() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            "runpy.run_module('examples.evaluate_late_interaction', run_name='__main__'); "
            "forbidden = {'torch', 'transformers', 'huggingface_hub', 'colbert'}; "
            "assert not forbidden & sys.modules.keys()",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    output = json.loads(process.stdout)
    assert "Synthetic MaxSim" in output["evidence"]
    assert "no learned-model quality" in output["evidence"]
    assert output["summary"]["failed"] == 0
    assert output["summary"]["top1"] == 1.0
    assert output["reports"][0]["result"]["ids"] == ["invoice", "copy", "intro", "opposite"]
    assert output["reports"][0]["result"]["scores"]["opposite"] == -1.0
