from __future__ import annotations

import unittest

import numpy as np

import kayak
from kayak_bridge.reference_softmax_pooling import softmax_similarity_scores


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class ReferenceSoftmaxPoolingTests(unittest.TestCase):
    def test_positive_temperature_is_required(self) -> None:
        query = kayak.query(_matrix((1.0, 0.0)))
        index = kayak.documents(["doc-a"], [_matrix((1.0, 0.0))]).pack()

        with self.assertRaises(ValueError):
            softmax_similarity_scores(query, index, temperature=0.0)

    def test_softmax_pooling_rewards_additional_support(self) -> None:
        query = kayak.query(_matrix((1.0, 0.0)))
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (1.0, 0.0)),
                _matrix((1.0, 0.0), (0.5, 0.0)),
            ],
        ).pack()

        pooled = softmax_similarity_scores(query, index, temperature=0.5)

        self.assertGreater(float(pooled.values[0]), float(pooled.values[1]))


if __name__ == "__main__":
    unittest.main()
