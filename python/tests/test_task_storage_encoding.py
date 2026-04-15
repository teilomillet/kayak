from kayak_bridge.task_storage_encoding import build_task_storage_encoding_bundle


def test_build_task_storage_encoding_bundle_computes_f16_ratios() -> None:
    bundle = build_task_storage_encoding_bundle(
        [
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_le",
                "primary_metric": "ndcg@10",
                "primary_value": 0.41,
                "mean_build_seconds": 1.2,
                "mean_load_seconds": 0.8,
                "mean_search_seconds": 0.2,
                "query_count": 10,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
                "artifact_byte_size": 960000,
                "artifact_bytes_per_document": 9600.0,
                "artifact_bytes_per_vector": 800.0,
            },
            {
                "dataset_id": "mock://dataset",
                "model_name": "mock-model",
                "family": "mock",
                "slice_name": "fixture",
                "encoding_kind": "binary_f16_le",
                "primary_metric": "ndcg@10",
                "primary_value": 0.405,
                "mean_build_seconds": 0.9,
                "mean_load_seconds": 0.5,
                "mean_search_seconds": 0.19,
                "query_count": 10,
                "document_count": 100,
                "vector_count": 1200,
                "vector_dim": 128,
                "artifact_byte_size": 520000,
                "artifact_bytes_per_document": 5200.0,
                "artifact_bytes_per_vector": 433.3333333,
            },
        ]
    )

    assert bundle.dataset_id == "mock://dataset"
    assert bundle.binary_le_artifact_byte_size == 960000
    assert bundle.binary_f16_le_artifact_byte_size == 520000
    assert bundle.artifact_byte_ratio_f16_vs_binary == 520000 / 960000
    assert bundle.build_seconds_ratio_f16_vs_binary == 0.9 / 1.2
    assert bundle.load_seconds_ratio_f16_vs_binary == 0.5 / 0.8
    assert bundle.search_seconds_ratio_f16_vs_binary == 0.19 / 0.2
    assert bundle.primary_value_delta_f16_minus_binary == 0.405 - 0.41
