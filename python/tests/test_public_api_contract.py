from __future__ import annotations

import unittest

import kayak
import kayak_bridge


class PublicApiContractTests(unittest.TestCase):
    def test_public_api_exports_match_supported_contract(self) -> None:
        expected = {
            "BackendInfo",
            "CandidateGenerator",
            "CandidateStageResult",
            "LateDocuments",
            "LateIndex",
            "LateQuery",
            "LateQueryBatch",
            "LateScores",
            "SearchHit",
            "SearchPlan",
            "SearchPlanResult",
            "SearchStageProfile",
            "StageArtifactMaterialization",
            "MOJO_EXACT_CPU_BACKEND",
            "NUMPY_REFERENCE_BACKEND",
            "available_backends",
            "backend_info",
            "clause_text_stage2_operator",
            "document_proxy_candidate_generator",
            "document_proxy_search_plan",
            "documents",
            "exact_full_scan_clause_text_search_plan",
            "exact_full_scan_candidate_generator",
            "exact_full_scan_search_plan",
            "exact_late_interaction_stage2_operator",
            "flat_query_dim128",
            "generate_candidates",
            "hybrid_flat_dim128_index",
            "maxsim",
            "maxsim_batch",
            "noop_topk_stage2_operator",
            "packed_index",
            "query",
            "query_batch",
            "search",
            "search_batch",
            "search_with_plan",
            "Stage2Operator",
        }

        self.assertEqual(set(kayak.PUBLIC_API), expected)
        self.assertEqual(set(kayak.__all__), expected)

        for name in expected:
            self.assertTrue(hasattr(kayak, name), name)

    def test_internal_bridge_is_marked_unstable(self) -> None:
        self.assertIsNotNone(kayak_bridge.__doc__)
        self.assertIn("not a stable public import surface", kayak_bridge.__doc__)

    def test_package_docstrings_match_sdk_boundary(self) -> None:
        self.assertIsNotNone(kayak.__doc__)
        self.assertIn("Public Python SDK", kayak.__doc__)
        self.assertIn("not the hosted engine surface", kayak.__doc__)
        self.assertIsNotNone(kayak_bridge.__doc__)
        self.assertIn("internal implementation layer", kayak_bridge.__doc__.lower())


if __name__ == "__main__":
    unittest.main()
