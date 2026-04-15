from __future__ import annotations

import unittest

import torch

from kayak_bridge.colbert_encoder import _trim_zero_padded_rows


class ColbertEncoderTests(unittest.TestCase):
    def test_trim_zero_padded_rows_drops_trailing_zeros(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        trimmed = _trim_zero_padded_rows(tensor)
        self.assertEqual(tuple(trimmed.shape), (2, 2))
        self.assertTrue(torch.equal(trimmed, tensor[:2]))


if __name__ == "__main__":
    unittest.main()
