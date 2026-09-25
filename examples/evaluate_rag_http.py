"""Adapt an HTTP RAG service without giving it evaluation references.

Install: uv sync
Run: uv run -m examples.evaluate_rag_http
     uv run -m examples.evaluate_rag_http --base-url http://localhost:9000

The default MockTransport uses no network or model. An explicit --base-url calls
POST /rag on your service: accept RAGInput JSON and return RAGOutput JSON.
For another response format, map its fields in HTTPPipeline.run. The caller owns
the HTTP client, timeout, and cleanup. The fixture review is exact-match only.
"""

import argparse

import httpx

from kayak.eval import (
    RAGCase,
    RAGContext,
    RAGDataset,
    RAGEvalConfig,
    RAGInput,
    RAGJudgments,
    RAGOutput,
    RAGReview,
    RAGReviewInput,
    RAGTrace,
    RankedOutput,
    evaluate_rag,
)


class HTTPPipeline:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def run(self, request: RAGInput) -> RAGOutput:
        response = self.client.post("/rag", json=request.model_dump(mode="json"))
        response.raise_for_status()
        # Map a framework-specific response here instead of inventing missing stages.
        return RAGOutput.model_validate_json(response.content)


def mock_service(request: httpx.Request) -> httpx.Response:
    if request.method != "POST" or request.url.path != "/rag":
        return httpx.Response(404)
    submitted = RAGInput.model_validate_json(request.content)
    text = "The retention period is 30 days."
    output = RAGOutput(
        final=RAGTrace(
            id=submitted.id,
            query=submitted.query,
            documents={"retention-policy": text},
            retrieval=RankedOutput(ids=["retention-policy"]),
            context=RAGContext(ids=["retention-policy"], text=text),
            answer=text,
            provenance={"service": "one-case-http-fixture-v1"},
        )
    )
    return httpx.Response(200, json=output.model_dump(mode="json"))


def dataset() -> RAGDataset:
    return RAGDataset(
        name="http-rag-contract-fixture",
        cases=[
            RAGCase(
                input=RAGInput(
                    id="retention",
                    query="What is the audit-log retention period?",
                    parameters={"collection": "help"},
                ),
                judgments=RAGJudgments(
                    relevance={"retention-policy": 1}, evidence_sets=[["retention-policy"]]
                ),
                reference_answer="The retention period is 30 days.",
            )
        ],
        provenance={"purpose": "wire-contract fixture; no model quality measured"},
    )


def review(request: RAGReviewInput) -> RAGReview:
    return RAGReview(
        review_input_sha256=request.sha256,
        correct=request.output.final.answer == request.case.reference_answer,
        provenance={"rubric": "fixture-exact-answer-v1"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="explicitly call a compatible live service")
    args = parser.parse_args()
    base_url = args.base_url or "http://kayak.fixture"
    transport = None if args.base_url else httpx.MockTransport(mock_service)
    config = RAGEvalConfig(system={"base_url": base_url, "timeout_seconds": 30.0})
    with httpx.Client(base_url=base_url, transport=transport, timeout=30.0) as client:
        pipeline = HTTPPipeline(client)
        report = evaluate_rag(dataset(), pipeline.run, config=config, review=review)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
