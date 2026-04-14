from __future__ import annotations

import unittest

from kayak_bridge.task_scale import (
    build_scaled_task_with_document_copies,
    choose_repeatable_distractor_doc_ids,
    protected_doc_ids_for_scale_sweep,
)


def _tiny_task() -> dict[str, object]:
    return {
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
            {
                "doc_id": "doc-c",
                "text": "gamma",
                "vector_count": 1,
                "vectors": [[-1.0, 0.0]],
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


class TaskScaleTests(unittest.TestCase):
    def test_protected_doc_ids_include_relevant_and_ranked_hits(self) -> None:
        protected = protected_doc_ids_for_scale_sweep(
            _tiny_task(),
            ranked_doc_ids_by_query=[("doc-b", "doc-c")],
        )

        self.assertEqual(protected, frozenset({"doc-a", "doc-b", "doc-c"}))

    def test_scaled_task_duplicates_only_repeatable_docs(self) -> None:
        task = _tiny_task()
        repeatable_doc_ids = choose_repeatable_distractor_doc_ids(
            task,
            protected_doc_ids=frozenset({"doc-a", "doc-b"}),
        )

        scaled = build_scaled_task_with_document_copies(
            task,
            target_document_count=5,
            repeatable_doc_ids=repeatable_doc_ids,
        )

        self.assertEqual(repeatable_doc_ids, ("doc-c",))
        self.assertEqual(scaled.base_document_count, 3)
        self.assertEqual(scaled.duplicated_document_count, 2)
        self.assertEqual(len(scaled.task["documents"]), 5)
        self.assertEqual(
            [
                document["doc_id"]
                for document in scaled.task["documents"][-2:]
            ],
            [
                "doc-c::scale_copy::000001",
                "doc-c::scale_copy::000002",
            ],
        )


if __name__ == "__main__":
    unittest.main()
