from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

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
    encoder = kayak.open_encoder(
        "callable",
        query_encoder=_token_vectors,
        document_encoder=_token_vectors,
    )
    backend = kayak.MOJO_EXACT_CPU_BACKEND

    documents = encoder.encode_documents(
        ["doc-a", "doc-b", "doc-c"],
        [
            "pixi installs python mojo and kayak together",
            "lancedb keeps multivector rows durable on disk",
            "kayak loads rows from lancedb and runs exact search locally",
        ],
    )

    with tempfile.TemporaryDirectory(prefix="kayak-lancedb-example-") as tmpdir:
        store = kayak.open_store(
            "lancedb",
            path=Path(tmpdir) / "lancedb",
            table_name="docs",
        )
        store.upsert(
            documents,
            metadata=[
                {"topic": "installation"},
                {"topic": "storage"},
                {"topic": "search"},
            ],
        )

        index = store.load_index(where={"topic": "search"}, include_text=True)
        query = encoder.encode_query("exact multivector search")
        hits = kayak.search(query, index, k=1, backend=backend)

        print("store stats:", store.stats())
        print("filtered docs:", index.doc_ids)
        print("backend:", backend)
        print("top hit:", (hits[0].doc_id, round(hits[0].score, 3)))


if __name__ == "__main__":
    main()
