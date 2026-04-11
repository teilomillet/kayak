from __future__ import annotations


def build_mock_task() -> dict:
    return {
        "family": "mock",
        "slice_name": "python_bridge",
        "why": "Minimal Python task spec used to verify Mojo-side conversion.",
        "primary_metric": "mrr",
        "k": 2,
        "nominal_query_vector_count": 2,
        "nominal_document_vector_count": 2,
        "vector_dim": 2,
        "documents": [
            {
                "doc_id": "doc-a",
                "vectors": [[1.0, 0.0], [0.0, 1.0]],
            },
            {
                "doc_id": "doc-b",
                "vectors": [[1.0, 0.0], [0.5, 0.5]],
            },
        ],
        "queries": [
            {
                "query_id": "q-1",
                "text": "mock query",
                "relevant_doc_ids": ["doc-a"],
                "vectors": [[1.0, 0.0], [0.0, 1.0]],
            }
        ],
    }
