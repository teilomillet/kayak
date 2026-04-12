from __future__ import annotations

import unittest

import kayak
import kayak_bridge


class PublicApiContractTests(unittest.TestCase):
    def test_public_api_exports_match_supported_contract(self) -> None:
        expected = {
            "LateDocuments",
            "LateIndex",
            "LateQuery",
            "LateScores",
            "SearchHit",
            "MOJO_EXACT_CPU_BACKEND",
            "NUMPY_REFERENCE_BACKEND",
            "documents",
            "flat_query_dim128",
            "hybrid_flat_dim128_index",
            "maxsim",
            "packed_index",
            "query",
            "search",
        }

        self.assertEqual(set(kayak.PUBLIC_API), expected)
        self.assertEqual(set(kayak.__all__), expected)

        for name in expected:
            self.assertTrue(hasattr(kayak, name), name)

    def test_internal_bridge_is_marked_unstable(self) -> None:
        self.assertIsNotNone(kayak_bridge.__doc__)
        self.assertIn("not a stable public import surface", kayak_bridge.__doc__)


if __name__ == "__main__":
    unittest.main()
