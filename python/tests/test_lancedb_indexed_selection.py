from __future__ import annotations

import unittest

from kayak_bridge.lancedb_indexed_selection import (
    select_lancedb_indexed_candidates_from_sweep_summary,
)


class LanceDbIndexedSelectionTests(unittest.TestCase):
    def test_selects_default_best_quality_and_fastest_improving(self) -> None:
        summary, specs = select_lancedb_indexed_candidates_from_sweep_summary(
            {
                "dataset_id": "dataset://a",
                "family": "mock",
                "slice_name": "slice",
                "primary_metric": "ndcg",
                "rows": [
                    {
                        "config_name": "default",
                        "primary_value": 0.81,
                        "mean_search_seconds": 0.024,
                        "primary_value_delta_vs_scan": 0.01,
                    },
                    {
                        "config_name": "quality",
                        "primary_value": 0.84,
                        "mean_search_seconds": 0.030,
                        "primary_value_delta_vs_scan": 0.04,
                        "indexed_refine_factor": 1,
                    },
                    {
                        "config_name": "tradeoff",
                        "primary_value": 0.82,
                        "mean_search_seconds": 0.021,
                        "primary_value_delta_vs_scan": 0.02,
                        "index_num_partitions": 8,
                        "indexed_refine_factor": 2,
                    },
                ],
            }
        )

        self.assertEqual(summary.selected_count, 3)
        self.assertEqual(
            [row.role for row in summary.rows],
            ["default", "best_quality", "fastest_quality_improving"],
        )
        self.assertEqual(
            [spec.name for spec in specs],
            ["default", "quality", "tradeoff"],
        )
        self.assertEqual(specs[2].index_build_controls.num_partitions, 8)
        self.assertEqual(specs[2].indexed_query_controls.refine_factor, 2)

    def test_deduplicates_when_default_is_also_fastest_improving(self) -> None:
        summary, specs = select_lancedb_indexed_candidates_from_sweep_summary(
            {
                "dataset_id": "dataset://a",
                "family": "mock",
                "slice_name": "slice",
                "primary_metric": "ndcg",
                "rows": [
                    {
                        "config_name": "default",
                        "primary_value": 0.83,
                        "mean_search_seconds": 0.020,
                        "primary_value_delta_vs_scan": 0.03,
                    },
                    {
                        "config_name": "quality",
                        "primary_value": 0.85,
                        "mean_search_seconds": 0.025,
                        "primary_value_delta_vs_scan": 0.05,
                        "indexed_refine_factor": 1,
                    },
                ],
            }
        )

        self.assertEqual(summary.selected_count, 2)
        self.assertEqual([spec.name for spec in specs], ["default", "quality"])


if __name__ == "__main__":
    unittest.main()
