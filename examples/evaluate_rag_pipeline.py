"""Configure and evaluate an ordinary callable RAG pipeline.

Install: uv sync
Run: uv run -m examples.evaluate_rag_pipeline
     uv run -m examples.evaluate_rag_pipeline --context-limit 0
     uv run -m examples.evaluate_rag_pipeline --output /tmp/rag-report.json

This lexical/extractive fixture uses no model or network. The pipeline sees
inputs only; its independent reviewer sees references. Replace Pipeline.run
with your framework call and map its observed outputs to RAGTrace/RAGOutput.
The exact-answer rubric demonstrates wiring, not semantic model quality.
"""

import argparse
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from kayak.eval import (
    RAGContext,
    RAGDataset,
    RAGEvalConfig,
    RAGInput,
    RAGReview,
    RAGReviewInput,
    RAGScore,
    RAGTrace,
    RankedOutput,
    evaluate_rag,
    save_rag_report,
)

DATA = Path(__file__).with_name("rag")


class PipelineSettings(BaseModel):
    """Application settings are validated here, independently of evaluator settings."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    retrieval_limit: int = Field(default=2, ge=0)
    context_limit: int = Field(default=1, ge=0)


class Pipeline:
    def __init__(self, settings: PipelineSettings) -> None:
        # Construct/load your index and model once at this boundary.
        self.settings = PipelineSettings.model_validate(settings.model_dump())
        self.documents = {
            "invoice": "Download your invoice from Settings > Billing > Invoices.",
            "password": "Reset your password from Settings > Security > Password.",
            "logs": "Export audit logs from Settings > Security > Audit logs.",
        }

    def run(self, request: RAGInput) -> RAGTrace:
        query_words = set(re.findall(r"[a-z0-9]+", request.query.lower()))
        scores = {
            identifier: float(len(query_words & set(re.findall(r"[a-z0-9]+", text.lower()))))
            for identifier, text in self.documents.items()
        }
        ordered = sorted(scores, key=scores.__getitem__, reverse=True)
        retrieved = [identifier for identifier in ordered if scores[identifier] > 0][
            : self.settings.retrieval_limit
        ]
        selected = retrieved[: self.settings.context_limit]
        context = RAGContext(
            ids=selected, text="\n\n".join(self.documents[identifier] for identifier in selected)
        )
        # This fixture copies a passage. A real application calls its generator here.
        answer = self.documents[selected[0]] if selected else "No supporting passage is in context."
        return RAGTrace(
            id=request.id,
            query=request.query,
            documents={identifier: self.documents[identifier] for identifier in retrieved},
            retrieval=RankedOutput(
                ids=retrieved, scores={identifier: scores[identifier] for identifier in retrieved}
            ),
            context=context,
            answer=answer,
            provenance={"retriever": "word-overlap-fixture-v1", "answerer": "extractive-v1"},
        )


def review(request: RAGReviewInput) -> RAGReview:
    """Only this review boundary receives references; equality is a fixture-only rubric."""
    trace = request.output.final
    if trace.answer is None or trace.context is None:
        raise ValueError("the fixture review requires an observed answer and context")
    return RAGReview(
        review_input_sha256=request.sha256,
        correct=trace.answer == request.case.reference_answer,
        grounded=bool(trace.answer) and trace.answer in trace.context.text,
        scores={"answer_words": RAGScore(value=float(len(trace.answer.split())))},
        provenance={
            "rubric": "fixture-exact-answer-and-literal-grounding-v1",
            "answer_words_scale": "whitespace-delimited word count; not a quality score",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATA / "dataset.json")
    parser.add_argument("--config", type=Path, default=DATA / "config.json")
    parser.add_argument("--context-limit", type=int, help="override the application context limit")
    parser.add_argument("--output", type=Path, help="save a checked report to a new file")
    args = parser.parse_args()
    dataset = RAGDataset.model_validate_json(args.dataset.read_bytes())
    config = RAGEvalConfig.model_validate_json(args.config.read_bytes())
    requested = dict(config.system)
    if args.context_limit is not None:
        requested["context_limit"] = args.context_limit
    settings = PipelineSettings.model_validate(requested)
    # Record effective defaults and overrides, then construct the pipeline explicitly.
    config = RAGEvalConfig.model_validate(
        {**config.model_dump(), "system": settings.model_dump(mode="json")}
    )
    pipeline = Pipeline(settings)
    report = evaluate_rag(dataset, pipeline.run, config=config, review=review)
    if args.output is not None:
        save_rag_report(report, args.output)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
