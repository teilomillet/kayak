from __future__ import annotations

import hashlib

import numpy as np

import kayak
from kayak.typing import TokenMatrixInput


class ToyLateInteractionModel:
    """Small runnable example for the callable encoder path.

    This stands in for a Hugging Face or custom model wrapper that already knows
    how to emit one token-level vector matrix per input string.
    """

    def __init__(self, *, vector_dim: int = 128) -> None:
        self.vector_dim = int(vector_dim)

    def encode_query_tokens(self, text: str) -> TokenMatrixInput:
        return self._encode_tokens(text)

    def encode_document_tokens(self, text: str) -> TokenMatrixInput:
        return self._encode_tokens(text)

    def _encode_tokens(self, text: str) -> TokenMatrixInput:
        rows: list[np.ndarray] = []
        for token in text.lower().split():
            vector = np.zeros(self.vector_dim, dtype=np.float32)
            hot_index = (
                int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
                % self.vector_dim
            )
            vector[hot_index] = np.float32(1.0)
            rows.append(vector)
        if not rows:
            return np.zeros((1, self.vector_dim), dtype=np.float32)
        return np.stack(rows)


def main() -> None:
    model = ToyLateInteractionModel()
    encoder = kayak.open_encoder(
        "callable",
        query_encoder=model.encode_query_tokens,
        document_encoder=model.encode_document_tokens,
    )

    retriever = kayak.open_text_retriever(
        encoder=encoder,
        store="memory",
        backend=kayak.NUMPY_REFERENCE_BACKEND,
    )

    retriever.upsert_texts(
        ["doc-a", "doc-b", "doc-c"],
        [
            "one environment keeps python mojo and kayak together",
            "kayak uses late interaction token vectors",
            "vector databases can stay optional",
        ],
        metadata=[
            {"topic": "installation"},
            {"topic": "retrieval"},
            {"topic": "storage"},
        ],
    )

    hits = retriever.search_text("python mojo kayak", k=2)

    print("default backend:", retriever.default_backend)
    print("top hits:", [(hit.doc_id, round(hit.score, 3)) for hit in hits])


if __name__ == "__main__":
    main()
