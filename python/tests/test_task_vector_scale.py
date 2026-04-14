from __future__ import annotations

import unittest

from kayak_bridge.task_vector_scale import build_vector_scaled_task


class TaskVectorScaleTests(unittest.TestCase):
    def test_repeats_document_vectors_without_changing_queries(self) -> None:
        task = {
            "dataset_id": "toy",
            "model_name": "toy-model",
            "family": "toy-family",
            "slice_name": "toy-slice",
            "primary_metric": "ndcg@10",
            "k": 10,
            "vector_dim": 2,
            "nominal_document_vector_count": 2,
            "documents": [
                {
                    "doc_id": "d1",
                    "text": "doc one",
                    "vector_count": 2,
                    "vectors": [[1.0, 0.0], [0.0, 1.0]],
                },
                {
                    "doc_id": "d2",
                    "text": "doc two",
                    "vector_count": 1,
                    "vectors": [[0.5, 0.5]],
                },
            ],
            "queries": [
                {
                    "query_id": "q1",
                    "text": "query",
                    "vector_count": 1,
                    "vectors": [[1.0, 0.0]],
                    "relevant_doc_ids": ["d1"],
                }
            ],
        }

        scaled = build_vector_scaled_task(task, document_vector_multiplier=3)

        self.assertEqual(scaled.document_vector_multiplier, 3)
        self.assertEqual(scaled.task["queries"], task["queries"])
        self.assertEqual(scaled.task["documents"][0]["vector_count"], 6)
        self.assertEqual(
            scaled.task["documents"][0]["vectors"],
            [[1.0, 0.0], [0.0, 1.0]] * 3,
        )
        self.assertEqual(scaled.task["documents"][1]["vector_count"], 3)
        self.assertEqual(
            scaled.task["nominal_document_vector_count"],
            round((6 + 3) / 2.0),
        )

    def test_rejects_non_positive_multiplier(self) -> None:
        with self.assertRaises(ValueError):
            build_vector_scaled_task(
                {"documents": [], "queries": []},
                document_vector_multiplier=0,
            )


if __name__ == "__main__":
    unittest.main()
