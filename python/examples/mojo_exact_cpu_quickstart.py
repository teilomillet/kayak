from __future__ import annotations

import numpy as np

import kayak


def main() -> None:
    query_vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    document_vectors = [
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    ]

    query = kayak.query(query_vectors)
    index = kayak.documents(["doc-a", "doc-b"], document_vectors).pack()

    scores = kayak.maxsim(
        query, index, backend=kayak.MOJO_EXACT_CPU_BACKEND
    )
    hits = kayak.search(
        query, index, k=2, backend=kayak.MOJO_EXACT_CPU_BACKEND
    )

    print("backend:", kayak.MOJO_EXACT_CPU_BACKEND)
    print("scores:", scores.numpy().tolist())
    print("hits:", [(hit.doc_id, hit.score) for hit in hits])


if __name__ == "__main__":
    main()
