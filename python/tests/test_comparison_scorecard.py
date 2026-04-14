from __future__ import annotations

import unittest

from kayak_bridge.comparison_scorecard import build_comparison_scorecard


class ComparisonScorecardTests(unittest.TestCase):
    def test_builds_pairwise_comparisons_against_named_baseline(self) -> None:
        systems = [
            (
                "kayak_exact",
                ".cache/kayak/kayak.json",
                {
                    "dataset_id": "dataset://a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "k": 10,
                    "primary_value": 0.25,
                    "mean_search_seconds": 0.01,
                },
            ),
            (
                "lancedb_scan",
                ".cache/kayak/lancedb_scan.json",
                {
                    "dataset_id": "dataset://a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "k": 10,
                    "primary_value": 0.20,
                    "mean_search_seconds": 0.02,
                    "engine": "lancedb",
                    "index_kind": "none",
                    "bytes_per_document": 10.0,
                },
            ),
        ]

        scorecard = build_comparison_scorecard(
            systems=systems,
            baseline_name="kayak_exact",
        )

        self.assertEqual(scorecard.dataset_id, "dataset://a")
        self.assertEqual(len(scorecard.systems), 2)
        self.assertEqual(len(scorecard.pairwise), 1)
        self.assertEqual(scorecard.pairwise[0].baseline, "kayak_exact")
        self.assertEqual(scorecard.pairwise[0].candidate, "lancedb_scan")
        self.assertAlmostEqual(scorecard.pairwise[0].primary_value_delta, -0.05)
        self.assertAlmostEqual(
            scorecard.pairwise[0].mean_search_seconds_ratio_vs_baseline, 2.0
        )

    def test_rejects_mismatched_slice(self) -> None:
        with self.assertRaisesRegex(ValueError, "slice_name"):
            build_comparison_scorecard(
                systems=[
                    (
                        "a",
                        "a.json",
                        {
                            "dataset_id": "dataset://a",
                            "family": "browsecomp_plus",
                            "slice_name": "gold",
                            "primary_metric": "ndcg",
                            "k": 10,
                            "primary_value": 0.25,
                            "mean_search_seconds": 0.01,
                        },
                    ),
                    (
                        "b",
                        "b.json",
                        {
                            "dataset_id": "dataset://a",
                            "family": "browsecomp_plus",
                            "slice_name": "evidence",
                            "primary_metric": "ndcg",
                            "k": 10,
                            "primary_value": 0.20,
                            "mean_search_seconds": 0.02,
                        },
                    ),
                ],
                baseline_name="a",
            )


if __name__ == "__main__":
    unittest.main()
