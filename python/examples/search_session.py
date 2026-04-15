from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
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
    with tempfile.TemporaryDirectory(prefix="kayak-search-session-") as tmpdir:
        retriever = kayak.open_text_retriever(
            encoder="callable",
            store="kayak",
            encoder_kwargs={
                "query_encoder": _token_vectors,
                "document_encoder": _token_vectors,
            },
            store_kwargs={"path": Path(tmpdir) / "late-store"},
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        retriever.upsert_texts(
            ["doc-install", "doc-storage", "doc-search"],
            [
                "uv or pixi can install kayak and keep mojo available",
                "qdrant chroma and lancedb can keep the encoded rows",
                "kayak loads one exact slice and serves repeated late interaction search",
            ],
            metadata=[
                {"topic": "installation"},
                {"topic": "storage"},
                {"topic": "search"},
            ],
        )

        session = retriever.session(include_text=True)
        single_hits = session.search_text("exact late interaction search", k=2)

        query_texts = [
            "mojo install kayak",
            "encoded rows storage",
            "repeated search",
        ]

        def run_query(text: str) -> tuple[kayak.SearchHit, ...]:
            return session.search_text(text, k=1)

        with ThreadPoolExecutor(max_workers=3) as executor:
            parallel_hits = list(executor.map(run_query, query_texts))

        print("default backend:", session.default_backend)
        print("session docs:", session.index.doc_ids)
        print(
            "single hits:",
            [(hit.doc_id, round(hit.score, 3)) for hit in single_hits],
        )
        print(
            "parallel top hits:",
            [
                (text, hits[0].doc_id, round(hits[0].score, 3))
                for text, hits in zip(query_texts, parallel_hits, strict=True)
            ],
        )


if __name__ == "__main__":
    main()
