"""Feed retrieved evidence to Kayak and receive Choice, Noul, and Score answers.
Setup: uv sync; start `uv run --extra serve kayak serve --device auto` separately.
Run: uv run -m examples.rag_decisions
KAYAK_BASE_URL and KAYAK_API_KEY configure the client. Scores are uncalibrated.
"""

import json
import os

from kayak import Choice, Client, Noul, Score


def retrieve(query: str) -> dict[str, str]:
    # Replace this tiny lookup with your index or RAG service's source IDs and text.
    if "audit logs" not in query.casefold():
        return {}
    return {"retention": "Audit logs are kept for 30 days."}


def main() -> None:
    query = "How long are audit logs kept?"
    documents = retrieve(query)
    if not documents:
        raise ValueError("No evidence retrieved; no decision requested")
    state = json.dumps({"query": query, "context": documents}, ensure_ascii=False)
    with Client(
        base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
        api_key=os.environ.get("KAYAK_API_KEY"),
    ) as client:
        result = client.judge(
            state=state,
            questions={
                "retention": Choice(
                    instructions="Which retention period is stated in the context?",
                    criteria={"30_days": "30 days", "90_days": "90 days", "unknown": "Not stated"},
                ),
                "stated": Noul(instructions="The context states the audit-log retention period."),
                "completeness": Score(
                    instructions="How completely does the context answer the query?",
                    criteria=["No answer", "Partial answer", "Complete answer"],
                ),
            },
        )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
