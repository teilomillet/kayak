from __future__ import annotations

import unittest

from kayak_bridge.r2med_biology_late_interaction_rescoring import (
    default_candidate_policy,
    default_rescore_operator_specs,
)


class R2MEDLateInteractionRescoringTests(unittest.TestCase):
    def test_default_candidate_policy_matches_best_rescored_fusion(self) -> None:
        policy = default_candidate_policy()

        self.assertEqual(
            policy.name,
            "query2doc_plus_1.9_lamer_plus_0.35_search_r1_qwen7b_ins",
        )
        self.assertEqual(
            policy.weights_by_variant_name,
            {
                "query2doc_gpt4": 1.0,
                "lamer_gpt4": 1.9,
                "search_r1_qwen7b_ins": 0.35,
            },
        )

    def test_default_rescore_operator_specs_include_control_and_alternatives(
        self,
    ) -> None:
        specs = default_rescore_operator_specs()

        self.assertEqual(specs[0].name, "maxsim_shortlist")
        self.assertEqual(specs[0].match_k, 1)
        self.assertTrue(any(spec.name == "topk_mean_k4" for spec in specs))
        self.assertTrue(any(spec.name == "softmax_tau0.5" for spec in specs))


if __name__ == "__main__":
    unittest.main()
