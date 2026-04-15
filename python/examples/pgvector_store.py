from __future__ import annotations

import hashlib
import os

import numpy as np

import kayak


def _token_vectors(text: str) -> np.ndarray:
    rows = []
    for token in text.lower().split():
        vector = np.zeros(128, dtype=np.float32)
        hot_index = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % 128
        vector[hot_index] = np.float32(1.0)
        rows.append(vector)
    return np.stack(rows)


def main() -> None:
    dsn = os.environ.get("KAYAK_PGVECTOR_DSN")
    if not dsn:
        raise SystemExit(
            "Set KAYAK_PGVECTOR_DSN to a Postgres database with the pgvector extension enabled."
        )

    encoder = kayak.open_encoder(
        "callable",
        query_encoder=_token_vectors,
        document_encoder=_token_vectors,
    )
    backend = kayak.MOJO_EXACT_CPU_BACKEND

    documents = encoder.encode_documents(
        ["doc-a", "doc-b", "doc-c"],
        [
            "one environment keeps python mojo and kayak together",
            "pgvector keeps late interaction rows in postgres",
            "kayak loads the exact slice and searches it locally",
        ],
    )

    with kayak.open_store(
        "pgvector",
        dsn=dsn,
        table_name="kayak_docs",
    ) as store:
        store.upsert(
            documents,
            metadata=[
                {"topic": "installation"},
                {"topic": "storage"},
                {"topic": "search"},
            ],
        )

        index = store.load_index(where={"topic": "search"}, include_text=True)
        query = encoder.encode_query("exact late interaction search")
        hits = kayak.search(query, index, k=1, backend=backend)

        print("store stats:", store.stats())
        print("filtered docs:", index.doc_ids)
        print("backend:", backend)
        print("top hit:", (hits[0].doc_id, round(hits[0].score, 3)))


if __name__ == "__main__":
    main()
