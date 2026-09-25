"""Evaluate real MaxSim arithmetic with manually supplied synthetic token vectors.

Install: uv sync
Run: uv run -m examples.evaluate_late_interaction

No model, download, or service is used. This checks scoring and the evaluation
adapter; it measures neither learned ColBERT quality nor model inference speed.
The caller prepares query/document vectors before evaluation, so timed calls
cover lookup and scoring only. State selects a query matrix; instructions and
candidate text are recorded but are not encoded by this synthetic example.

Replace MaxSimRanker.rank with your library's query encoding and scoring, while
loading its document index once outside evaluate_cases. Return original IDs and
native scores in RankedOutput; probabilities and CLM metadata are unnecessary.
The formula follows ColBERT's sum of each query token's best document-token dot
product: https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/modeling/colbert.py
"""

import json
import math
from collections.abc import Mapping

from kayak import RankingRequest
from kayak.eval import RankedOutput, RankingDataset, evaluate_cases, summarize_cases

TokenVectors = tuple[tuple[float, ...], ...]


def maxsim(query: TokenVectors, document: TokenVectors) -> float:
    """Sum best token dot products; unit vectors make each dot product a cosine."""
    if not query or not document:
        raise ValueError("query and document matrices must be nonempty")
    dimension = len(query[0])
    if dimension == 0 or any(len(token) != dimension for token in (*query, *document)):
        raise ValueError("all tokens need the same positive embedding dimension")
    if not all(math.isfinite(value) for token in (*query, *document) for value in token):
        raise ValueError("token coordinates must be finite")
    matches = []
    for query_token in query:
        similarities = [
            math.fsum(q * d for q, d in zip(query_token, document_token, strict=True))
            for document_token in document
        ]
        matches.append(max(similarities))
    score = math.fsum(matches)
    if not math.isfinite(score):
        raise ValueError("MaxSim score must be finite")
    return score


class MaxSimRanker:
    """Own snapshots of prepared matrices; ranking receives requests, never labels."""

    def __init__(
        self, queries: Mapping[str, TokenVectors], documents: Mapping[str, TokenVectors]
    ) -> None:
        self.queries = dict(queries)
        self.documents = dict(documents)

    def rank(self, request: RankingRequest) -> RankedOutput:
        query = self.queries[request.state]
        scores = {
            identifier: maxsim(query, self.documents[identifier])
            for identifier in request.candidates
        }
        # Stable sorting preserves submitted candidate order when scores tie.
        return RankedOutput(
            ids=sorted(request.candidates, key=scores.__getitem__, reverse=True),
            scores=scores,
            provenance={
                "implementation": "synthetic-maxsim-v1",
                "embeddings": "manually-supplied-vectors",
                "query_recipe": "state-only-matrix-lookup",
            },
        )


def main() -> None:
    # These unit vectors are illustrative inputs, not a trained text encoder.
    ranker = MaxSimRanker(
        queries={"invoice PDF": ((1.0, 0.0), (0.0, 1.0))},
        documents={
            "intro": ((1.0, 0.0),),
            "invoice": ((1.0, 0.0), (0.0, 1.0)),
            "copy": ((1.0, 0.0), (0.0, 1.0)),
            "opposite": ((-1.0, 0.0),),
        },
    )
    dataset = RankingDataset.model_validate(
        {
            "name": "synthetic-maxsim",
            "kind": "ranking",
            "examples": ["examples/evaluate_late_interaction.py"],
            "k": 2,
            "cases": [
                {
                    "id": "two-matching-tokens",
                    "request": {
                        "state": "invoice PDF",
                        "instructions": "Rank by sum of maximum token dot products.",
                        "candidates": {
                            "intro": "Invoice introduction without a PDF download.",
                            "invoice": "Download the invoice PDF.",
                            "copy": "Download the invoice PDF.",
                            "opposite": "An unrelated passage in this synthetic fixture.",
                        },
                    },
                    "relevant": ["invoice", "copy"],
                }
            ],
        }
    )
    reports = evaluate_cases(dataset, rank=ranker.rank)
    print(
        json.dumps(
            {
                "evidence": (
                    "Synthetic MaxSim arithmetic; no learned-model quality or inference speed."
                ),
                "dataset": dataset.model_dump(mode="json"),
                "reports": [report.model_dump(mode="json") for report in reports],
                "summary": summarize_cases(dataset, reports),
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
