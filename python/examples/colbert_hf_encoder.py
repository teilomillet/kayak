from __future__ import annotations

import kayak


def main() -> None:
    # `model_name` is the Hugging Face repo id for the ColBERT checkpoint.
    encoder = kayak.open_encoder(
        "colbert",
        model_name="colbert-ir/colbertv2.0",
    )

    retriever = kayak.open_text_retriever(
        encoder=encoder,
        store="memory",
        backend=kayak.NUMPY_REFERENCE_BACKEND,
    )

    retriever.upsert_texts(
        ["doc-a", "doc-b"],
        [
            "One environment can keep Python, Mojo, and kayak together.",
            "Kayak searches ColBERT-style token vectors with late interaction.",
        ],
    )

    hits = retriever.search_text(
        "python mojo environment",
        k=2,
    )

    print("top hits:", [(hit.doc_id, round(hit.score, 3)) for hit in hits])


if __name__ == "__main__":
    main()
