from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge.tachiom_arrays import _top_positions


def _reference_top_positions(scores: np.ndarray, k: int) -> tuple[int, ...]:
    positions = np.arange(int(scores.shape[0]))
    order = np.lexsort((positions, -scores))
    return tuple(int(position) for position in positions[order[:k]])


class TachiomArrayTests(unittest.TestCase):
    def test_top_positions_keeps_low_position_tie_breaks(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.8, 0.8, 0.7], dtype=np.float32)

        self.assertEqual(tuple(_top_positions(scores, 3)), (0, 1, 2))

    def test_top_positions_matches_full_sort_reference(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.normal(size=1024).astype(np.float32)
        scores[10:20] = 0.5

        for k in (1, 8, 32, 1024):
            self.assertEqual(
                tuple(_top_positions(scores, k)),
                _reference_top_positions(scores, k),
            )


if __name__ == "__main__":
    unittest.main()
