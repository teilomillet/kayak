from __future__ import annotations

import unittest

from kayak_bridge.plaid_task_candidate_profile import (
    aggregate_profile_rows,
    candidate_document_vector_count,
    candidate_recall_at_k,
    vector_count_stats,
)


class PlaidTaskCandidateProfileTests(unittest.TestCase):
    def test_vector_count_stats_reports_shape_axis(self) -> None:
        stats = vector_count_stats((12, 20, 28))

        self.assertEqual(stats.min_count, 12)
        self.assertEqual(stats.mean_count, 20.0)
        self.assertEqual(stats.max_count, 28)
        self.assertEqual(stats.total_count, 60)

    def test_candidate_recall_at_k_uses_reference_top_k(self) -> None:
        recall = candidate_recall_at_k(
            ("d0", "d2", "d4"),
            ("d0", "d1", "d2", "d3"),
            k=4,
        )

        self.assertEqual(recall, 0.5)

    def test_candidate_document_vector_count_sums_selected_docs(self) -> None:
        count = candidate_document_vector_count((4, 8, 16, 32), (3, 1))

        self.assertEqual(count, 40)

    def test_aggregate_profile_rows_keeps_optional_recall_explicit(self) -> None:
        summary = aggregate_profile_rows(
            (
                {
                    "candidate_document_vector_count": 64,
                    "candidate_i8_token_payload_bytes_estimate": 8448,
                    "nextplaid_4bit_residual_bytes_estimate": 4096,
                    "candidate_recall_at_k_vs_exact": 1.0,
                    "profile": {
                        "full_candidate_mean_seconds": 0.4,
                        "workspace_full_candidate_mean_seconds": 0.3,
                        "unordered_candidate_mean_seconds": 0.2,
                        "centroid_scoring_mean_seconds": 0.1,
                        "centroid_selection_mean_seconds": 0.05,
                        "posting_accumulation_mean_seconds": 0.08,
                        "final_topk_mean_seconds": 0.02,
                        "unordered_final_topk_mean_seconds": 0.01,
                        "posting_visit_count": 100,
                        "touched_document_count": 25,
                    },
                },
                {
                    "candidate_document_vector_count": 128,
                    "candidate_i8_token_payload_bytes_estimate": 16896,
                    "nextplaid_4bit_residual_bytes_estimate": 8192,
                    "candidate_recall_at_k_vs_exact": 0.5,
                    "profile": {
                        "full_candidate_mean_seconds": 0.8,
                        "workspace_full_candidate_mean_seconds": 0.6,
                        "unordered_candidate_mean_seconds": 0.4,
                        "centroid_scoring_mean_seconds": 0.2,
                        "centroid_selection_mean_seconds": 0.1,
                        "posting_accumulation_mean_seconds": 0.16,
                        "final_topk_mean_seconds": 0.04,
                        "unordered_final_topk_mean_seconds": 0.02,
                        "posting_visit_count": 200,
                        "touched_document_count": 50,
                    },
                },
            )
        )

        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["candidate_document_vector_count_mean"], 96.0)
        self.assertEqual(summary["posting_visit_count_mean"], 150.0)
        self.assertEqual(summary["candidate_recall_at_k_vs_exact_mean"], 0.75)


if __name__ == "__main__":
    unittest.main()
