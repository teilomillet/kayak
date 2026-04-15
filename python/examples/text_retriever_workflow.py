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
    with tempfile.TemporaryDirectory(prefix="kayak-text-retriever-") as tmpdir:
        retriever = kayak.open_text_retriever(
            encoder="callable",
            store="kayak",
            encoder_kwargs={
                "query_encoder": _token_vectors,
                "document_encoder": _token_vectors,
            },
            store_kwargs={"path": Path(tmpdir) / "late-store"},
        )

        retriever.upsert_texts(
            ["doc-a", "doc-b", "doc-c"],
            [
                "pixi installs python mojo and kayak together",
                "lancedb can keep multivector rows on disk",
                "kayak loads the slice and runs exact late interaction search",
            ],
            metadata=[
                {"topic": "installation"},
                {"topic": "storage"},
                {"topic": "search"},
            ],
        )

        hits = retriever.search_text(
            "python mojo kayak installation",
            k=2,
        )

        print("store stats:", retriever.stats())
        print("default backend:", retriever.default_backend)
        print("top hits:", [(hit.doc_id, round(hit.score, 3)) for hit in hits])


if __name__ == "__main__":
    main()
