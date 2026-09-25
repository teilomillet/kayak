"""Rerank retrieved documents while retaining original IDs and sources.

Install client: uv sync
Start server in another terminal: uv run --extra serve kayak serve --device auto
Run: uv run -m examples.rerank_documents
Override: KAYAK_BASE_URL=http://127.0.0.1:8000 KAYAK_API_KEY=...

The application supplies a small search shortlist. Kayak orders its text;
original records supply the IDs, sources, and text printed afterward. This
does not perform retrieval or establish answer correctness. Rankings are
uncalibrated comparisons; evaluate relevance on your own query/document pairs.
"""

import os
from dataclasses import dataclass

import kayak


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    source: str


def main() -> None:
    state = "Where can I download an invoice for last month's payment?"
    documents = [
        Document("doc-plan", "Change your plan in Billing, under Subscription.", "help/plans"),
        Document(
            "doc-invoice",
            "Open Billing, then Invoices. Select the month and download the invoice PDF.",
            "help/invoices",
        ),
        Document("doc-access", "Workspace owners can invite members in Settings.", "help/members"),
    ]
    documents_by_id = {document.id: document for document in documents}
    candidates = {document.id: document.text for document in documents}
    base_url = os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000")
    api_key = os.environ.get("KAYAK_API_KEY")

    with kayak.Client(base_url=base_url, api_key=api_key) as client:
        result = client.rank(
            state=state,
            instructions="Which document most directly helps answer this question?",
            candidates=candidates,
        )

    for position, ranked in enumerate(result.ranked, start=1):
        document = documents_by_id[ranked.id]
        print(f"{position}. {document.id} ({document.source})")
        print(document.text)


if __name__ == "__main__":
    main()
