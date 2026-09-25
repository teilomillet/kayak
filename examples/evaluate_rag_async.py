"""Evaluate an async application and retain intermediate rewritten-query searches.

Install: uv sync
Run: uv run -m examples.evaluate_rag_async --max-concurrency 2 --repeats 2

This model-free fixture rewrites a literal alias and searches two passages.
Intermediate observations keep their own queries; final-task labels score only
the final trace. No causal graph or model quality is inferred from these records.
Replace AsyncPipeline.run with your async framework call and explicit mapping.
"""

import argparse
import asyncio

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
    aevaluate_rag,
)


class AsyncPipeline:
    def __init__(self, alias: str, canonical: str) -> None:
        self.alias = alias
        self.canonical = canonical
        self.documents = {
            "invoice": "Download Kayak invoices from Billing > Invoices.",
            "password": "Reset your Kayak password from Security > Password.",
        }

    async def run(self, request: RAGInput) -> RAGOutput:
        rewritten = request.query.replace(self.alias, self.canonical)
        # A stand-in for awaited application I/O, not a latency benchmark.
        await asyncio.sleep(0)
        identifiers = [
            identifier
            for identifier, text in self.documents.items()
            if identifier in rewritten.lower() and self.canonical in text
        ]
        documents = {identifier: self.documents[identifier] for identifier in identifiers}
        search = RAGTrace(
            id=f"{request.id}/search",
            query=rewritten,
            documents=documents,
            retrieval=RankedOutput(ids=identifiers),
            provenance={"search": "literal-term-fixture-v1"},
        )
        context = RAGContext(ids=identifiers, text="\n".join(documents.values()))
        final = RAGTrace(
            id=request.id,
            query=request.query,
            documents=documents,
            context_input_ids=identifiers,
            context=context,
            answer=context.text or "No matching passage.",
            provenance={"answerer": "extractive-fixture-v1"},
        )
        return RAGOutput(final=final, steps=[search])


def dataset() -> RAGDataset:
    return RAGDataset(
        name="rewritten-query-async-fixture",
        cases=[
            RAGCase(
                input=RAGInput(id="invoice", query="Where is the Aurora invoice page?"),
                judgments=RAGJudgments(evidence_sets=[["invoice"]]),
                reference_answer="Download Kayak invoices from Billing > Invoices.",
            ),
            RAGCase(
                input=RAGInput(id="password", query="Where is the Aurora password page?"),
                judgments=RAGJudgments(evidence_sets=[["password"]]),
                reference_answer="Reset your Kayak password from Security > Password.",
            ),
        ],
        provenance={"purpose": "controlled async integration; no learned-model quality"},
    )


async def review(request: RAGReviewInput) -> RAGReview:
    return RAGReview(
        review_input_sha256=request.sha256,
        correct=request.output.final.answer == request.case.reference_answer,
        provenance={"rubric": "fixture-exact-answer-v1"},
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    alias, canonical = "Aurora", "Kayak"
    config = RAGEvalConfig(
        max_concurrency=args.max_concurrency,
        repeats=args.repeats,
        system={"alias": alias, "canonical": canonical},
    )
    pipeline = AsyncPipeline(alias, canonical)
    report = await aevaluate_rag(dataset(), pipeline.run, config=config, review=review)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
