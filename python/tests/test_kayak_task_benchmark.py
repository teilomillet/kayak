from __future__ import annotations

import unittest

from kayak_bridge.kayak_task_benchmark import benchmark_task_with_kayak_exact


class KayakTaskBenchmarkTests(unittest.TestCase):
    def test_benchmarks_tiny_task_with_numpy_backend(self) -> None:
        task = {
            "dataset_id": "dataset://tiny",
            "model_name": "unit-test-model",
            "family": "tiny_family",
            "slice_name": "tiny_slice",
            "primary_metric": "ndcg",
            "k": 2,
            "nominal_query_vector_count": 1,
            "nominal_document_vector_count": 1,
            "vector_dim": 2,
            "documents": [
                {
                    "doc_id": "doc-a",
                    "text": "alpha",
                    "vector_count": 1,
                    "vectors": [[1.0, 0.0]],
                },
                {
                    "doc_id": "doc-b",
                    "text": "beta",
                    "vector_count": 1,
                    "vectors": [[0.0, 1.0]],
                },
            ],
            "queries": [
                {
                    "query_id": "q-1",
                    "text": "alpha question",
                    "relevant_doc_ids": ["doc-a"],
                    "vector_count": 1,
                    "vectors": [[1.0, 0.0]],
                }
            ],
        }

        summary = benchmark_task_with_kayak_exact(
            task,
            warmup_iterations=0,
            measurement_iterations=1,
            backend="numpy_reference",
        )

        self.assertEqual(summary.engine, "kayak")
        self.assertEqual(summary.index_kind, "exact_packed")
        self.assertEqual(summary.vector_metric, "dot_product")
        self.assertEqual(summary.document_count, 2)
        self.assertEqual(summary.document_vector_count_total, 2)
        self.assertAlmostEqual(summary.primary_value, 1.0)
        self.assertAlmostEqual(summary.mean_ndcg_at_k, 1.0)
        self.assertAlmostEqual(summary.mean_recall_at_k, 1.0)
        self.assertAlmostEqual(summary.success_rate_at_k, 1.0)


if __name__ == "__main__":
    unittest.main()
