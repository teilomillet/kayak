from kayak_bridge.task_storage_workflow import (
    build_task_storage_workflow_bundle,
    default_queries_per_loads,
)


def test_default_queries_per_loads_keeps_standard_points_and_task_size() -> None:
    assert default_queries_per_loads(8) == (1, 2, 4, 8, 16, 32)
    assert default_queries_per_loads(40) == (1, 2, 4, 8, 16, 32, 40)


def test_build_task_storage_workflow_bundle_detects_crossover() -> None:
    bundle = build_task_storage_workflow_bundle(
        [
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_le",
                "queries_per_load": 1,
                "mean_session_seconds": 1.0,
                "mean_session_seconds_per_query": 1.0,
                "query_count": 8,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
            },
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_f16_le",
                "queries_per_load": 1,
                "mean_session_seconds": 1.1,
                "mean_session_seconds_per_query": 1.1,
                "query_count": 8,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
            },
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_le",
                "queries_per_load": 16,
                "mean_session_seconds": 4.0,
                "mean_session_seconds_per_query": 0.25,
                "query_count": 8,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
            },
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_f16_le",
                "queries_per_load": 16,
                "mean_session_seconds": 3.8,
                "mean_session_seconds_per_query": 0.2375,
                "query_count": 8,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
            },
        ],
        queries_per_loads=(1, 16),
    )

    assert bundle.first_queries_per_load_with_f16_not_slower_total_session == 16
    assert bundle.first_queries_per_load_with_f16_not_slower_per_query == 16
    assert bundle.points[0].session_seconds_ratio_f16_vs_binary == 1.1
    assert bundle.points[1].session_seconds_ratio_f16_vs_binary == 3.8 / 4.0
