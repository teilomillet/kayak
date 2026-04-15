from __future__ import annotations

import unittest

import numpy as np

from kayak_bridge.r2med_biology_late_interaction_policy_expansion import (
    _topk_doc_ids,
    default_expansion_variant_names,
    default_expansion_weights,
    default_seed_policy,
)


class R2MEDLateInteractionPolicyExpansionTests(unittest.TestCase):
    def test_default_seed_policy_matches_two_variant_anchor(self) -> None:
        policy = default_seed_policy()

        self.assertEqual(policy.name, "query2doc_plus_1.9_lamer")
        self.assertEqual(
            policy.weights_by_variant_name,
            {"query2doc_gpt4": 1.0, "lamer_gpt4": 1.9},
        )

    def test_default_expansion_variants_exclude_seed_variants(self) -> None:
        variant_names = default_expansion_variant_names()

        self.assertNotIn("query2doc_gpt4", variant_names)
        self.assertNotIn("lamer_gpt4", variant_names)
        self.assertIn("search_r1_qwen7b_ins", variant_names)

    def test_default_expansion_weights_cover_fine_low_weight_region(self) -> None:
        weights = default_expansion_weights()

        self.assertIn(0.25, weights)
        self.assertIn(0.35, weights)
        self.assertIn(0.75, weights)

    def test_topk_doc_ids_orders_scores_descending_with_stable_ties(self) -> None:
        doc_ids = ("doc-a", "doc-b", "doc-c", "doc-d")
        values = np.asarray([0.5, 0.8, 0.8, 0.1], dtype=np.float32)

        self.assertEqual(
            _topk_doc_ids(doc_ids, values, 3),
            ("doc-b", "doc-c", "doc-a"),
        )


if __name__ == "__main__":
    unittest.main()
