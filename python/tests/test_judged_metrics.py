from __future__ import annotations

import unittest

from kayak_bridge.judged_metrics import (
    choose_primary_value,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    success_at_k,
    summarize_ranked_task,
)


class JudgedMetricsTests(unittest.TestCase):
    def test_rank_metrics_match_repo_semantics(self) -> None:
        ranked_doc_ids = ("doc-b", "doc-a", "doc-c")
        relevant_doc_ids = ("doc-a", "doc-c")

        self.assertAlmostEqual(reciprocal_rank_at_k(ranked_doc_ids, relevant_doc_ids, 3), 0.5)
        self.assertAlmostEqual(success_at_k(ranked_doc_ids, relevant_doc_ids, 3), 1.0)
        self.assertAlmostEqual(recall_at_k(ranked_doc_ids, relevant_doc_ids, 3), 1.0)
        self.assertAlmostEqual(
            ndcg_at_k(ranked_doc_ids, relevant_doc_ids, 3),
            (1.0 / 1.584962500721156 + 1.0 / 2.0)
            / (1.0 + 1.0 / 1.584962500721156),
        )

    def test_choose_primary_value_rejects_unknown_metric(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown primary metric"):
            choose_primary_value(
                "map",
                mean_ndcg_at_k=0.0,
                mean_reciprocal_rank=0.0,
                mean_recall_at_k=0.0,
                success_rate_at_k=0.0,
            )

    def test_summarize_ranked_task_uses_task_primary_metric(self) -> None:
        task = {
            "primary_metric": "mrr",
            "k": 2,
            "documents": [{"doc_id": "doc-a"}, {"doc_id": "doc-b"}],
            "queries": [
                {"relevant_doc_ids": ["doc-a"]},
                {"relevant_doc_ids": ["doc-b"]},
            ],
        }

        summary = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=[
                ("doc-a", "doc-b"),
                ("doc-a", "doc-b"),
            ],
        )

        self.assertEqual(summary.primary_metric, "mrr")
        self.assertAlmostEqual(summary.mean_reciprocal_rank, 0.75)
        self.assertAlmostEqual(summary.primary_value, 0.75)
        self.assertEqual(summary.document_count, 2)
        self.assertEqual(summary.query_count, 2)


if __name__ == "__main__":
    unittest.main()
