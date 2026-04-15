from __future__ import annotations

import unittest

import numpy as np

import kayak
from kayak_bridge.reference_topk_pooling import topk_mean_similarity_scores


def _matrix(*rows: tuple[float, float]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


class ReferenceTopKPoolingTests(unittest.TestCase):
    def test_match_k_one_matches_maxsim(self) -> None:
        query = kayak.query(_matrix((1.0, 0.0), (0.0, 1.0)))
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (0.0, 1.0)),
                _matrix((1.0, 0.0), (1.0, 0.0)),
            ],
        ).pack()

        pooled = topk_mean_similarity_scores(query, index, match_k=1)
        maxsim = kayak.maxsim(query, index)

        np.testing.assert_allclose(pooled.values, maxsim.values)

    def test_match_k_two_rewards_multiple_matches(self) -> None:
        query = kayak.query(_matrix((1.0, 0.0)))
        index = kayak.documents(
            ["doc-a", "doc-b"],
            [
                _matrix((1.0, 0.0), (1.0, 0.0)),
                _matrix((1.0, 0.0), (0.0, 1.0)),
            ],
        ).pack()

        pooled = topk_mean_similarity_scores(query, index, match_k=2)

        self.assertGreater(float(pooled.values[0]), float(pooled.values[1]))


if __name__ == "__main__":
    unittest.main()
