"""Owns small aggregate summaries for repeated benchmark runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any, Mapping, Sequence


def _require_runs(runs: Sequence[Mapping[str, Any]]) -> None:
    if not runs:
        raise ValueError("variance summary requires at least one run")


def _float_values(runs: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [float(run[key]) for run in runs]


@dataclass(frozen=True, slots=True)
class BenchmarkVarianceSummary:
    dataset_id: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    index_kind: str
    index_num_partitions: int | None
    index_num_sub_vectors: int | None
    index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None
    rebuild_count: int
    primary_value_min: float
    primary_value_max: float
    primary_value_mean: float
    mean_search_seconds_min: float
    mean_search_seconds_max: float
    mean_search_seconds_mean: float
    run_summaries: tuple[dict[str, Any], ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrozenBenchmarkSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    query_count: int
    document_count: int
    nominal_query_vector_count: int
    nominal_document_vector_count: int
    vector_dim: int
    engine: str
    engine_version: str
    index_kind: str
    index_num_partitions: int | None
    index_num_sub_vectors: int | None
    index_target_partition_size: int | None
    indexed_nprobes: int | None
    indexed_refine_factor: int | None
    vector_metric: str | None
    freeze_policy: str
    rebuild_count: int
    primary_value: float
    mean_ndcg_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    success_rate_at_k: float
    mean_search_seconds: float
    storage_byte_size: float | None
    bytes_per_document: float | None
    bytes_per_vector: float | None
    source_primary_value_min: float
    source_primary_value_max: float
    source_mean_search_seconds_min: float
    source_mean_search_seconds_max: float

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def summarize_benchmark_variance(
    runs: Sequence[Mapping[str, Any]],
) -> BenchmarkVarianceSummary:
    _require_runs(runs)
    first = runs[0]
    dataset_id = str(first["dataset_id"])
    family = str(first["family"])
    slice_name = str(first["slice_name"])
    primary_metric = str(first["primary_metric"])
    k = int(first["k"])
    index_kind = str(first.get("index_kind", "unknown"))
    index_num_partitions = _optional_int(first, "index_num_partitions")
    index_num_sub_vectors = _optional_int(first, "index_num_sub_vectors")
    index_target_partition_size = _optional_int(first, "index_target_partition_size")
    indexed_nprobes = _optional_int(first, "indexed_nprobes")
    indexed_refine_factor = _optional_int(first, "indexed_refine_factor")

    for run in runs[1:]:
        if str(run["dataset_id"]) != dataset_id:
            raise ValueError("all runs must share dataset_id")
        if str(run["family"]) != family:
            raise ValueError("all runs must share family")
        if str(run["slice_name"]) != slice_name:
            raise ValueError("all runs must share slice_name")
        if str(run["primary_metric"]) != primary_metric:
            raise ValueError("all runs must share primary_metric")
        if int(run["k"]) != k:
            raise ValueError("all runs must share k")
        if str(run.get("index_kind", "unknown")) != index_kind:
            raise ValueError("all runs must share index_kind")
        if _optional_int(run, "index_num_partitions") != index_num_partitions:
            raise ValueError("all runs must share index_num_partitions")
        if _optional_int(run, "index_num_sub_vectors") != index_num_sub_vectors:
            raise ValueError("all runs must share index_num_sub_vectors")
        if (
            _optional_int(run, "index_target_partition_size")
            != index_target_partition_size
        ):
            raise ValueError("all runs must share index_target_partition_size")
        if _optional_int(run, "indexed_nprobes") != indexed_nprobes:
            raise ValueError("all runs must share indexed_nprobes")
        if _optional_int(run, "indexed_refine_factor") != indexed_refine_factor:
            raise ValueError("all runs must share indexed_refine_factor")

    primary_values = _float_values(runs, "primary_value")
    search_seconds = _float_values(runs, "mean_search_seconds")

    return BenchmarkVarianceSummary(
        dataset_id=dataset_id,
        family=family,
        slice_name=slice_name,
        primary_metric=primary_metric,
        k=k,
        index_kind=index_kind,
        index_num_partitions=index_num_partitions,
        index_num_sub_vectors=index_num_sub_vectors,
        index_target_partition_size=index_target_partition_size,
        indexed_nprobes=indexed_nprobes,
        indexed_refine_factor=indexed_refine_factor,
        rebuild_count=len(runs),
        primary_value_min=min(primary_values),
        primary_value_max=max(primary_values),
        primary_value_mean=mean(primary_values),
        mean_search_seconds_min=min(search_seconds),
        mean_search_seconds_max=max(search_seconds),
        mean_search_seconds_mean=mean(search_seconds),
        run_summaries=tuple(dict(run) for run in runs),
    )


def freeze_benchmark_summary_mean(
    runs: Sequence[Mapping[str, Any]],
    *,
    freeze_policy: str = "mean_across_rebuilds",
) -> FrozenBenchmarkSummary:
    variance = summarize_benchmark_variance(runs)
    first = runs[0]

    def maybe_mean(key: str) -> float | None:
        values = [run.get(key) for run in runs]
        if any(value is None for value in values):
            return None
        return mean(float(value) for value in values)

    return FrozenBenchmarkSummary(
        dataset_id=variance.dataset_id,
        model_name=str(first.get("model_name", "")),
        family=variance.family,
        slice_name=variance.slice_name,
        primary_metric=variance.primary_metric,
        k=int(first["k"]),
        query_count=int(first["query_count"]),
        document_count=int(first["document_count"]),
        nominal_query_vector_count=int(first["nominal_query_vector_count"]),
        nominal_document_vector_count=int(first["nominal_document_vector_count"]),
        vector_dim=int(first["vector_dim"]),
        engine=str(first.get("engine", "")),
        engine_version=str(first.get("engine_version", "")),
        index_kind=variance.index_kind,
        index_num_partitions=variance.index_num_partitions,
        index_num_sub_vectors=variance.index_num_sub_vectors,
        index_target_partition_size=variance.index_target_partition_size,
        indexed_nprobes=variance.indexed_nprobes,
        indexed_refine_factor=variance.indexed_refine_factor,
        vector_metric=(
            None if first.get("vector_metric") is None else str(first["vector_metric"])
        ),
        freeze_policy=freeze_policy,
        rebuild_count=variance.rebuild_count,
        primary_value=mean(float(run["primary_value"]) for run in runs),
        mean_ndcg_at_k=mean(float(run["mean_ndcg_at_k"]) for run in runs),
        mean_recall_at_k=mean(float(run["mean_recall_at_k"]) for run in runs),
        mean_reciprocal_rank=mean(
            float(run["mean_reciprocal_rank"]) for run in runs
        ),
        success_rate_at_k=mean(float(run["success_rate_at_k"]) for run in runs),
        mean_search_seconds=variance.mean_search_seconds_mean,
        storage_byte_size=maybe_mean("storage_byte_size"),
        bytes_per_document=maybe_mean("bytes_per_document"),
        bytes_per_vector=maybe_mean("bytes_per_vector"),
        source_primary_value_min=variance.primary_value_min,
        source_primary_value_max=variance.primary_value_max,
        source_mean_search_seconds_min=variance.mean_search_seconds_min,
        source_mean_search_seconds_max=variance.mean_search_seconds_max,
    )


def _optional_int(summary: Mapping[str, Any], key: str) -> int | None:
    value = summary.get(key)
    if value is None:
        return None
    return int(value)
