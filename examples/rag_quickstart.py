"""An application-owned retrieval/generation recipe, assessed with kayak.eval.
Setup: uv sync; ollama pull gemma3:1b (Ollama must be running).
Run: uv run -m examples.rag_quickstart
"""

import re

import httpx
from pydantic import BaseModel, ConfigDict

from kayak.eval import RAGContext, RAGTrace, RankedOutput

DOCUMENTS = {
    "retention": "Audit logs are kept for 30 days.",
    "password": "Reset your password from Settings.",
}


class Generation(BaseModel):
    model_config = ConfigDict(strict=True)
    response: str
    done: bool


def main() -> None:
    query = "How long are audit logs kept?"
    words = set(re.findall(r"\w+", query.casefold()))
    scores = {
        source: len(words & set(re.findall(r"\w+", text.casefold())))
        for source, text in DOCUMENTS.items()
    }
    source = max(scores, key=scores.__getitem__, default=None)
    if source is None or scores[source] == 0:
        raise ValueError("No evidence retrieved; generation skipped")
    context = RAGContext(ids=[source], text=f"[{source}]\n{DOCUMENTS[source]}")
    prompt = f"Context:\n{context.text}\nQuestion: {query}\nAnswer only as '<number> days'."
    with httpx.Client(base_url="http://localhost:11434", timeout=120) as client:
        response = client.post(
            "/api/generate", json={"model": "gemma3:1b", "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        generation = Generation.model_validate_json(response.content)
    if not generation.done or not generation.response.strip():
        raise ValueError("Ollama returned an incomplete or blank answer")
    trace = RAGTrace(
        id="retention",
        query=query,
        documents={source: DOCUMENTS[source]},
        retrieval=RankedOutput(ids=[source], scores={source: float(scores[source])}),
        context=context,
        answer=generation.response,
        provenance={"generator": "ollama", "model": "gemma3:1b", "prompt": prompt},
    )
    checks = trace.evaluate(expected_answer="30 days", expected_sources=["retention"])
    print("Answer:", trace.answer)
    print("Expected source in context:", checks.source_coverage["context"])
    print("Answer matches reference:", checks.answer_correct)


if __name__ == "__main__":
    main()
