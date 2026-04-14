"""Owns the filtered three-system benchmark contract for real LanceDB runs.

This module freezes the comparison shape that the threshold study justified:
- filter zero vectors once up front when LanceDB cosine search requires it
- measure Kayak exact on the same filtered task
- measure LanceDB scan and Kayak-from-LanceDB on the same LanceDB-stored corpus
- emit one auditable scorecard rather than ad hoc one-off numbers
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from .comparison_scorecard import build_comparison_scorecard
from .kayak_task_benchmark import benchmark_task_with_kayak_exact
from .lancedb_benchmark import _filter_zero_vectors
from .lancedb_storage_comparison import benchmark_task_with_lancedb_storage_compare


@dataclass(frozen=True, slots=True)
class FilteredContractBundle:
    contract_name: str
    zero_vector_policy: str
    original_zero_document_vector_count_filtered: int
    original_zero_query_vector_count_filtered: int
    original_stored_document_vector_count_total: int
    original_stored_query_vector_count_total: int
    filtered_task_path: str
    kayak_exact_path: str
    storage_compare_path: str
    scorecard_path: str

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _flatten_lancedb_scan_summary(
    storage_compare: Mapping[str, Any],
) -> dict[str, object]:
    document_count = int(storage_compare["document_count"])
    stored_document_vector_count_total = int(
        storage_compare["stored_document_vector_count_total"]
    )
    storage_byte_size = int(storage_compare["storage_byte_size"])
    lancedb_scan = storage_compare["lancedb_scan"]
    return {
        "dataset_id": storage_compare["dataset_id"],
        "model_name": storage_compare["model_name"],
        "family": storage_compare["family"],
        "slice_name": storage_compare["slice_name"],
        "primary_metric": storage_compare["primary_metric"],
        "primary_value": lancedb_scan["primary_value"],
        "mean_ndcg_at_k": lancedb_scan["mean_ndcg_at_k"],
        "mean_reciprocal_rank": lancedb_scan["mean_reciprocal_rank"],
        "mean_recall_at_k": lancedb_scan["mean_recall_at_k"],
        "success_rate_at_k": lancedb_scan["success_rate_at_k"],
        "mean_search_seconds": lancedb_scan["mean_search_seconds"],
        "k": storage_compare["k"],
        "query_count": storage_compare["query_count"],
        "document_count": document_count,
        "nominal_query_vector_count": storage_compare["nominal_query_vector_count"],
        "nominal_document_vector_count": (
            storage_compare["nominal_document_vector_count"]
        ),
        "vector_dim": storage_compare["vector_dim"],
        "engine": "lancedb",
        "engine_version": storage_compare["storage_engine_version"],
        "index_kind": lancedb_scan["index_kind"],
        "vector_metric": "cosine",
        "storage_byte_size": storage_byte_size,
        "bytes_per_document": storage_byte_size / float(document_count),
        "bytes_per_vector": storage_byte_size
        / float(stored_document_vector_count_total),
    }


def _flatten_kayak_from_lancedb_summary(
    storage_compare: Mapping[str, Any],
) -> dict[str, object]:
    document_count = int(storage_compare["document_count"])
    stored_document_vector_count_total = int(
        storage_compare["stored_document_vector_count_total"]
    )
    storage_byte_size = int(storage_compare["storage_byte_size"])
    kayak_summary = storage_compare["kayak_exact_from_lancedb"]
    return {
        "dataset_id": storage_compare["dataset_id"],
        "model_name": storage_compare["model_name"],
        "family": storage_compare["family"],
        "slice_name": storage_compare["slice_name"],
        "primary_metric": storage_compare["primary_metric"],
        "primary_value": kayak_summary["primary_value"],
        "mean_ndcg_at_k": kayak_summary["mean_ndcg_at_k"],
        "mean_reciprocal_rank": kayak_summary["mean_reciprocal_rank"],
        "mean_recall_at_k": kayak_summary["mean_recall_at_k"],
        "success_rate_at_k": kayak_summary["success_rate_at_k"],
        "mean_search_seconds": kayak_summary["mean_search_seconds"],
        "k": storage_compare["k"],
        "query_count": storage_compare["query_count"],
        "document_count": document_count,
        "nominal_query_vector_count": storage_compare["nominal_query_vector_count"],
        "nominal_document_vector_count": (
            storage_compare["nominal_document_vector_count"]
        ),
        "vector_dim": storage_compare["vector_dim"],
        "engine": "kayak",
        "engine_version": kayak_summary["engine_version"],
        "index_kind": kayak_summary["index_kind"],
        "vector_metric": "dot_product",
        "storage_byte_size": storage_byte_size,
        "bytes_per_document": storage_byte_size / float(document_count),
        "bytes_per_vector": storage_byte_size
        / float(stored_document_vector_count_total),
    }


def benchmark_filtered_contract(
    *,
    task: Mapping[str, Any],
    output_root: Path,
    artifact_prefix: str,
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
) -> dict[str, object]:
    (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        stored_document_vector_count_total,
        stored_query_vector_count_total,
    ) = _filter_zero_vectors(task)

    output_root.mkdir(parents=True, exist_ok=True)
    filtered_task_path = output_root / f"{artifact_prefix}_filtered_task.json"
    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact.json"
    storage_compare_path = output_root / f"{artifact_prefix}_storage_compare.json"
    scorecard_path = output_root / f"{artifact_prefix}_scorecard.json"
    bundle_path = output_root / f"{artifact_prefix}_bundle.json"

    _write_json(filtered_task_path, filtered_task)

    kayak_exact = benchmark_task_with_kayak_exact(
        filtered_task,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    ).to_json_ready()
    _write_json(kayak_exact_path, kayak_exact)

    storage_compare = benchmark_task_with_lancedb_storage_compare(
        task=filtered_task,
        database_root=output_root / f"{artifact_prefix}_lancedb",
        table_name=artifact_prefix,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    ).to_json_ready()
    _write_json(storage_compare_path, storage_compare)

    lancedb_scan = _flatten_lancedb_scan_summary(storage_compare)
    kayak_from_lancedb = _flatten_kayak_from_lancedb_summary(storage_compare)

    scorecard = build_comparison_scorecard(
        systems=[
            ("kayak_exact", str(kayak_exact_path), kayak_exact),
            ("lancedb_scan", str(storage_compare_path), lancedb_scan),
            ("kayak_exact_from_lancedb", str(storage_compare_path), kayak_from_lancedb),
        ],
        baseline_name="kayak_exact",
    ).to_json_ready()
    _write_json(scorecard_path, scorecard)

    bundle = FilteredContractBundle(
        contract_name="lancedb_filtered_three_system_contract",
        zero_vector_policy="filter_zero_vectors_once_before_all_branches",
        original_zero_document_vector_count_filtered=zero_document_vector_count,
        original_zero_query_vector_count_filtered=zero_query_vector_count,
        original_stored_document_vector_count_total=stored_document_vector_count_total,
        original_stored_query_vector_count_total=stored_query_vector_count_total,
        filtered_task_path=str(filtered_task_path),
        kayak_exact_path=str(kayak_exact_path),
        storage_compare_path=str(storage_compare_path),
        scorecard_path=str(scorecard_path),
    ).to_json_ready()
    _write_json(bundle_path, bundle)

    return {
        "bundle_path": str(bundle_path),
        "bundle": bundle,
        "scorecard": scorecard,
        "kayak_exact": kayak_exact,
        "lancedb_scan": lancedb_scan,
        "kayak_exact_from_lancedb": kayak_from_lancedb,
    }
