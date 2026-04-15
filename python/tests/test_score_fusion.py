from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge.late_scores import LateScores
from kayak_bridge.score_fusion import weighted_sum_score_batches


def _scores(doc_ids: tuple[str, ...], values: tuple[float, ...]) -> LateScores:
    return LateScores.from_values(
        "test",
        doc_ids,
        np.asarray(values, dtype=np.float32),
    )


class ScoreFusionTests(unittest.TestCase):
    def test_weighted_sum_score_batches_adds_aligned_values(self) -> None:
        fused = weighted_sum_score_batches(
            (
                (
                    1.0,
                    (
                        _scores(("a", "b"), (1.0, 2.0)),
                        _scores(("a", "b"), (3.0, 4.0)),
                    ),
                ),
                (
                    2.0,
                    (
                        _scores(("a", "b"), (0.5, 1.5)),
                        _scores(("a", "b"), (1.0, 2.0)),
                    ),
                ),
            )
        )
        self.assertEqual(tuple(fused[0].doc_ids), ("a", "b"))
        np.testing.assert_allclose(fused[0].values, np.asarray((2.0, 5.0)))
        np.testing.assert_allclose(fused[1].values, np.asarray((5.0, 8.0)))

    def test_weighted_sum_score_batches_rejects_misaligned_doc_ids(self) -> None:
        with self.assertRaises(ValueError):
            weighted_sum_score_batches(
                (
                    (1.0, (_scores(("a", "b"), (1.0, 2.0)),)),
                    (1.0, (_scores(("b", "a"), (1.0, 2.0)),)),
                )
            )


if __name__ == "__main__":
    unittest.main()
