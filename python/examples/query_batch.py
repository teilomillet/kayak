from __future__ import annotations

import numpy as np

import kayak


def dim128(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


def main() -> None:
    queries = kayak.query_batch(
        [
            np.stack([dim128(0), dim128(1)]),
            np.stack([dim128(0), dim128(1), dim128(2)]),
        ]
    )
    index = kayak.documents(
        ["doc-a", "doc-b", "doc-c"],
        [
            np.stack([dim128(0), dim128(1), dim128(2)]),
            np.stack([dim128(0), dim128(1)]),
            np.stack([dim128(2)]),
        ],
    ).pack()

    scores_batch = kayak.maxsim_batch(
        queries,
        index,
        backend=kayak.NUMPY_REFERENCE_BACKEND,
    )

    print("batch_size:", queries.batch_size)
    print("vector_counts:", queries.vector_counts)
    print("batch_scores:", [scores.numpy().tolist() for scores in scores_batch])


if __name__ == "__main__":
    main()
