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
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    ]

    query = kayak.query(query_vectors)
    documents = kayak.documents(["doc-a", "doc-b", "doc-c"], document_vectors)
    index = documents.pack()

    scores = kayak.maxsim(
        query, index, backend=kayak.NUMPY_REFERENCE_BACKEND
    )
    hits = kayak.search(
        query, index, k=2, backend=kayak.NUMPY_REFERENCE_BACKEND
    )

    print("backend:", kayak.NUMPY_REFERENCE_BACKEND)
    print("scores:", scores.numpy().tolist())
    print("hits:", [(hit.doc_id, hit.score) for hit in hits])


if __name__ == "__main__":
    main()
