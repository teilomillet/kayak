from kayak_bridge.task_storage_cold_workflow import (
    TaskStorageColdWorkflowBundle,
    benchmark_task_storage_cold_workflow,
    default_cold_queries_per_loads,
)


def test_default_cold_queries_per_loads_is_small_representative_ladder() -> None:
    assert default_cold_queries_per_loads() == (1, 8, 32)


def test_bundle_serialization_shape() -> None:
    bundle = TaskStorageColdWorkflowBundle(
        dataset_id="mock://dataset",
        model_name="mock-model",
        family="mock",
        slice_name="fixture",
        cold_policy="memory_pressure_percent_free",
        cold_policy_percent_free=60,
        cold_policy_sample_seconds=1,
        cold_policy_hysteresis_seconds=1,
        repetitions=3,
        first_queries_per_load_with_f16_not_slower_total_session=None,
        points=(),
        runs=(),
    )
    payload = bundle.to_json_ready()
    assert payload["cold_policy"] == "memory_pressure_percent_free"
    assert payload["cold_policy_percent_free"] == 60
    assert payload["repetitions"] == 3
