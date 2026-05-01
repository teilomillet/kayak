from __future__ import annotations

import unittest

from kayak_bridge.tachiom_hnsw import TachiomHnswConfig
from kayak_bridge.tachiom_pq import TachiomResidualPqConfig
from kayak_bridge.tachiom_task_benchmark import benchmark_task_with_tachiom
from kayak_bridge.tachiom_types import TachiomTacConfig


def _tiny_task(*, include_token_ids: bool = True) -> dict:
    documents = [
        {
            "doc_id": "doc-a",
            "text": "alpha beta",
            "vector_count": 2,
            "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        },
        {
            "doc_id": "doc-b",
            "text": "gamma",
            "vector_count": 1,
            "vectors": [[0.0, 0.0, 1.0, 0.0]],
        },
    ]
    if include_token_ids:
        documents[0]["token_ids"] = [10, 20]
        documents[1]["token_ids"] = [30]
    return {
        "dataset_id": "dataset://tiny",
        "model_name": "unit-test-model",
        "family": "tiny_family",
        "slice_name": "tiny_slice",
        "primary_metric": "ndcg",
        "k": 1,
        "nominal_query_vector_count": 2,
        "nominal_document_vector_count": 2,
        "vector_dim": 4,
        "documents": documents,
        "queries": [
            {
                "query_id": "q-1",
                "text": "alpha beta question",
                "relevant_doc_ids": ["doc-a"],
                "vector_count": 2,
                "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            },
            {
                "query_id": "q-2",
                "text": "gamma question",
                "relevant_doc_ids": ["doc-b"],
                "vector_count": 1,
                "vectors": [[0.0, 0.0, 1.0, 0.0]],
            },
        ],
    }


class TachiomTaskBenchmarkTests(unittest.TestCase):
    def test_full_task_benchmark_scores_tiny_judged_task(self) -> None:
        summary = benchmark_task_with_tachiom(
            _tiny_task(),
            engine="tachiom_tac_hnsw_pq",
            tac_config=TachiomTacConfig(centroid_count=4, candidate_k=2),
            hnsw_config=TachiomHnswConfig(
                max_neighbors=2,
                ef_construction=2,
                ef_search=2,
            ),
            pq_config=TachiomResidualPqConfig(
                subspace_count=2,
                codebook_size=2,
                kmeans_iterations=2,
            ),
            warmup_iterations=0,
            measurement_iterations=1,
            exact_backend="numpy_reference",
        )

        self.assertEqual(summary.engine, "tachiom_tac_hnsw_pq")
        self.assertEqual(summary.document_vector_count_total, 3)
        self.assertAlmostEqual(summary.primary_value, 1.0)
        self.assertAlmostEqual(summary.exact_primary_value, 1.0)
        self.assertAlmostEqual(summary.final_recall_at_k_vs_exact, 1.0)
        self.assertAlmostEqual(summary.candidate_recall_at_k_vs_exact, 1.0)

    def test_task_benchmark_requires_document_token_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires document token_ids"):
            benchmark_task_with_tachiom(
                _tiny_task(include_token_ids=False),
                engine="tachiom_tac",
                tac_config=TachiomTacConfig(centroid_count=4, candidate_k=2),
                warmup_iterations=0,
                measurement_iterations=1,
                exact_backend="numpy_reference",
            )


if __name__ == "__main__":
    unittest.main()
