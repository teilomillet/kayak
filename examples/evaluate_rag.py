"""Locate a RAG evidence loss and run a separate diagnostic replay.

Install: uv sync
Run: uv run -m examples.evaluate_rag

This deterministic fixture uses no model, download, or service. The application
records its own stage outputs; kayak.eval only assesses those observations.
The replay changes context packing and runs the same answer function again.
It demonstrates a controlled fixture, not model quality or production causality.
"""

import json

from kayak.eval import (
    RAGAnswerReview,
    RAGContext,
    RAGJudgments,
    RAGReplay,
    RAGTrace,
    RankedOutput,
    assess_rag,
)


def pack_context(documents: dict[str, str], identifiers: list[str], limit: int) -> RAGContext:
    """Pack the first passages intact; a real application owns its token budget."""
    selected = identifiers[:limit]
    return RAGContext(
        ids=selected,
        text="\n\n".join(f"[{identifier}] {documents[identifier]}" for identifier in selected),
    )


def answer(context: RAGContext) -> str:
    """A deterministic fixture with an observable dependency on supplied context."""
    if "Settings > Billing > Invoices" in context.text:
        return "Open Settings > Billing > Invoices and download the PDF."
    return "The supplied context does not explain where to download the invoice."


def record_runs() -> tuple[RAGTrace, RAGTrace]:
    documents = {
        "profile": "Settings > Profile changes your display name.",
        "invoice": "Settings > Billing > Invoices lists downloadable invoice PDFs.",
    }
    # Fixed retrieval and reranking outputs intentionally put the wrong passage first.
    retrieval = RankedOutput(ids=["profile", "invoice"], provenance={"retriever": "fixture-v1"})
    reranking = RankedOutput(ids=["profile", "invoice"], provenance={"reranker": "fixture-v1"})
    context = pack_context(documents, reranking.ids, limit=1)
    original = RAGTrace(
        id="invoice-original",
        query="Where do I download my invoice PDF?",
        documents=documents,
        retrieval=retrieval,
        reranking=reranking,
        context=context,
        answer=answer(context),
        provenance={"answerer": "context-lookup-fixture-v1"},
    )
    # Retain the original run. Change one boundary, then execute downstream again.
    replay_context = pack_context(documents, reranking.ids, limit=2)
    replay = RAGTrace(
        id="invoice-more-context",
        query=original.query,
        documents=documents,
        retrieval=retrieval,
        reranking=reranking,
        context=replay_context,
        answer=answer(replay_context),
        provenance=original.provenance,
        replay=RAGReplay(
            original_trace_sha256=original.sha256,
            changed_boundary="context",
            purpose="Test whether retaining both retrieved passages changes the answer.",
        ),
    )
    return original, replay


def main() -> None:
    for trace in record_runs():
        # The caller owns these judgments. This exact-match rule is only for this fixture;
        # real answers need an independent rubric and recorded human or judge provenance.
        judgments = RAGJudgments(
            relevance={"profile": 0, "invoice": 1},
            evidence_sets=[["invoice"]],
            evidence_texts={"invoice": "Settings > Billing > Invoices"},
            answer=RAGAnswerReview(
                trace_sha256=trace.sha256,
                correct=trace.answer == "Open Settings > Billing > Invoices and download the PDF.",
                provenance={"rubric": "fixture-exact-answer-v1"},
            ),
        )
        assessment = assess_rag(trace, judgments, k=1)
        print(
            json.dumps(
                {
                    "evidence": "Deterministic RAG fixture; no model quality measured.",
                    "trace": trace.model_dump(mode="json"),
                    "judgments": judgments.model_dump(mode="json"),
                    "assessment": assessment.model_dump(mode="json"),
                },
                allow_nan=False,
            )
        )


if __name__ == "__main__":
    main()
