from __future__ import annotations

import unittest

from kayak_bridge.benchmark_variance import (
    freeze_benchmark_summary_mean,
    summarize_benchmark_variance,
)


class BenchmarkVarianceTests(unittest.TestCase):
    def test_summarizes_primary_value_and_latency_ranges(self) -> None:
        summary = summarize_benchmark_variance(
            [
                {
                    "dataset_id": "dataset://a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "index_kind": "ivf_pq",
                    "primary_value": 0.2,
                    "mean_search_seconds": 0.03,
                },
                {
                    "dataset_id": "dataset://a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "index_kind": "ivf_pq",
                    "primary_value": 0.4,
                    "mean_search_seconds": 0.02,
                },
            ]
        )

        self.assertEqual(summary.rebuild_count, 2)
        self.assertAlmostEqual(summary.primary_value_min, 0.2)
        self.assertAlmostEqual(summary.primary_value_max, 0.4)
        self.assertAlmostEqual(summary.primary_value_mean, 0.3)
        self.assertAlmostEqual(summary.mean_search_seconds_min, 0.02)
        self.assertAlmostEqual(summary.mean_search_seconds_max, 0.03)
        self.assertAlmostEqual(summary.mean_search_seconds_mean, 0.025)

    def test_rejects_mismatched_index_kind(self) -> None:
        with self.assertRaisesRegex(ValueError, "index_kind"):
            summarize_benchmark_variance(
                [
                    {
                        "dataset_id": "dataset://a",
                        "family": "browsecomp_plus",
                        "slice_name": "gold",
                        "primary_metric": "ndcg",
                        "index_kind": "ivf_pq",
                        "primary_value": 0.2,
                        "mean_search_seconds": 0.03,
                    },
                    {
                        "dataset_id": "dataset://a",
                        "family": "browsecomp_plus",
                        "slice_name": "gold",
                        "primary_metric": "ndcg",
                        "index_kind": "none",
                        "primary_value": 0.4,
                        "mean_search_seconds": 0.02,
                    },
                ]
            )

    def test_rejects_mismatched_indexed_refine_factor(self) -> None:
        with self.assertRaisesRegex(ValueError, "indexed_refine_factor"):
            summarize_benchmark_variance(
                [
                    {
                        "dataset_id": "dataset://a",
                        "family": "browsecomp_plus",
                        "slice_name": "gold",
                        "primary_metric": "ndcg",
                        "index_kind": "ivf_pq",
                        "indexed_refine_factor": 2,
                        "primary_value": 0.2,
                        "mean_search_seconds": 0.03,
                    },
                    {
                        "dataset_id": "dataset://a",
                        "family": "browsecomp_plus",
                        "slice_name": "gold",
                        "primary_metric": "ndcg",
                        "index_kind": "ivf_pq",
                        "indexed_refine_factor": 4,
                        "primary_value": 0.4,
                        "mean_search_seconds": 0.02,
                    },
                ]
            )

    def test_freezes_mean_summary_from_runs(self) -> None:
        frozen = freeze_benchmark_summary_mean(
            [
                {
                    "dataset_id": "dataset://a",
                    "model_name": "model-a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "k": 10,
                    "query_count": 4,
                    "document_count": 90,
                    "nominal_query_vector_count": 32,
                    "nominal_document_vector_count": 175,
                    "vector_dim": 128,
                    "engine": "lancedb",
                    "engine_version": "0.30.2",
                    "index_kind": "ivf_pq",
                    "index_num_partitions": 8,
                    "index_num_sub_vectors": 16,
                    "index_target_partition_size": 256,
                    "indexed_nprobes": 32,
                    "indexed_refine_factor": 2,
                    "vector_metric": "cosine",
                    "primary_value": 0.2,
                    "mean_ndcg_at_k": 0.2,
                    "mean_recall_at_k": 0.3,
                    "mean_reciprocal_rank": 0.4,
                    "success_rate_at_k": 0.5,
                    "mean_search_seconds": 0.03,
                    "storage_byte_size": 10,
                    "bytes_per_document": 5.0,
                    "bytes_per_vector": 2.0,
                },
                {
                    "dataset_id": "dataset://a",
                    "model_name": "model-a",
                    "family": "browsecomp_plus",
                    "slice_name": "gold",
                    "primary_metric": "ndcg",
                    "k": 10,
                    "query_count": 4,
                    "document_count": 90,
                    "nominal_query_vector_count": 32,
                    "nominal_document_vector_count": 175,
                    "vector_dim": 128,
                    "engine": "lancedb",
                    "engine_version": "0.30.2",
                    "index_kind": "ivf_pq",
                    "index_num_partitions": 8,
                    "index_num_sub_vectors": 16,
                    "index_target_partition_size": 256,
                    "indexed_nprobes": 32,
                    "indexed_refine_factor": 2,
                    "vector_metric": "cosine",
                    "primary_value": 0.4,
                    "mean_ndcg_at_k": 0.4,
                    "mean_recall_at_k": 0.5,
                    "mean_reciprocal_rank": 0.6,
                    "success_rate_at_k": 0.7,
                    "mean_search_seconds": 0.01,
                    "storage_byte_size": 14,
                    "bytes_per_document": 7.0,
                    "bytes_per_vector": 3.0,
                },
            ],
            freeze_policy="mean_across_2_rebuilds",
        )

        self.assertEqual(frozen.freeze_policy, "mean_across_2_rebuilds")
        self.assertEqual(frozen.rebuild_count, 2)
        self.assertEqual(frozen.engine, "lancedb")
        self.assertEqual(frozen.index_kind, "ivf_pq")
        self.assertEqual(frozen.index_num_partitions, 8)
        self.assertEqual(frozen.index_num_sub_vectors, 16)
        self.assertEqual(frozen.index_target_partition_size, 256)
        self.assertEqual(frozen.indexed_nprobes, 32)
        self.assertEqual(frozen.indexed_refine_factor, 2)
        self.assertAlmostEqual(frozen.primary_value, 0.3)
        self.assertAlmostEqual(frozen.mean_ndcg_at_k, 0.3)
        self.assertAlmostEqual(frozen.mean_search_seconds, 0.02)
        self.assertAlmostEqual(frozen.storage_byte_size, 12.0)
        self.assertAlmostEqual(frozen.source_primary_value_min, 0.2)
        self.assertAlmostEqual(frozen.source_primary_value_max, 0.4)


if __name__ == "__main__":
    unittest.main()
