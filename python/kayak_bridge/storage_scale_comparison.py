"""Owns same-task storage comparisons for Kayak packed storage versus LanceDB.

This module keeps the storage question narrow and measurable:
- both engines consume the same filtered encoded task
- both scale that task with the same distractor-duplication policy
- Kayak storage is measured through the native Mojo packed-index store/load path
- LanceDB storage is measured through the Python LanceDB table build/open path
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence

from .judged_metrics import summarize_ranked_task
from .lancedb_benchmark import (
    _build_table_rows,
    _directory_byte_size,
    _filter_zero_vectors,
    _require_lancedb,
    _search_doc_ids,
    _validate_unit_norm_vectors,
)
from .kayak_task_benchmark import rank_task_with_kayak_exact
from .task_scale import (
    build_scaled_task_with_document_copies,
    choose_repeatable_distractor_doc_ids,
    protected_doc_ids_for_scale_sweep,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PIXI_BIN = shutil.which("pixi")
TASK_JSON_STORAGE_INPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_encoding_input.json"
)
TASK_JSON_STORAGE_OUTPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_encoding_output.json"
)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _write_task_json(path: Path, task: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_native_kayak_storage_benchmark(
    task: Mapping[str, Any],
) -> dict[str, object]:
    if PIXI_BIN is None:
        raise RuntimeError("expected `pixi` on PATH to run the native Mojo benchmark")

    _write_task_json(TASK_JSON_STORAGE_INPUT_PATH, task)
    pixi_env_root = REPO_ROOT / ".pixi" / "envs" / "default"
    subprocess_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("UV_")
    }
    subprocess_env.pop("PYTHONHOME", None)
    subprocess_env.pop("__PYVENV_LAUNCHER__", None)
    subprocess_env["PYTHONPATH"] = "python"
    subprocess_env["VIRTUAL_ENV"] = str(pixi_env_root)
    subprocess_env["CONDA_PREFIX"] = str(pixi_env_root)
    subprocess_env["PATH"] = (
        f"{pixi_env_root / 'bin'}:{subprocess_env.get('PATH', '')}"
    )
    subprocess_env["DYLD_LIBRARY_PATH"] = (
        f"{pixi_env_root / 'lib'}:{subprocess_env.get('DYLD_LIBRARY_PATH', '')}"
    )

    completed = subprocess.run(
        [
            PIXI_BIN,
            "run",
            "mojo",
            "-I",
            ".",
            "benchmarks/task_json_storage_encoding.mojo",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        env=subprocess_env,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "task_json_storage_encoding benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not TASK_JSON_STORAGE_OUTPUT_PATH.exists():
        raise RuntimeError(
            "task_json_storage_encoding benchmark did not write "
            f"{TASK_JSON_STORAGE_OUTPUT_PATH}"
        )

    with TASK_JSON_STORAGE_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True, slots=True)
class StorageScaleComparisonRow:
    target_document_count: int
    actual_document_count: int
    scale_factor_vs_base: float
    duplicated_document_count: int
    stored_document_vector_count_total: int
    kayak_storage_byte_size: int
    kayak_bytes_per_document: float
    kayak_bytes_per_vector: float
    kayak_build_seconds: float
    kayak_load_seconds: float
    kayak_search_seconds: float
    kayak_primary_value: float
    kayak_vector_payload_encoding: str
    lancedb_storage_byte_size: int
    lancedb_bytes_per_document: float
    lancedb_bytes_per_vector: float
    lancedb_build_seconds: float
    lancedb_open_table_seconds: float
    lancedb_search_seconds: float
    lancedb_primary_value: float
    lancedb_storage_byte_ratio_vs_kayak: float | None
    lancedb_build_seconds_ratio_vs_kayak: float | None
    lancedb_search_seconds_ratio_vs_kayak: float | None
    lancedb_primary_ratio_vs_kayak: float | None


@dataclass(frozen=True, slots=True)
class StorageScaleComparisonSummary:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    k: int
    vector_dim: int
    base_document_count: int
    base_nominal_document_vector_count: int
    base_zero_document_vector_count_filtered: int
    base_zero_query_vector_count_filtered: int
    protected_document_count: int
    repeatable_distractor_source_count: int
    inflation_policy: str
    kayak_storage_engine: str
    kayak_storage_format: str
    lancedb_storage_engine: str
    lancedb_storage_engine_version: str
    rows: tuple[StorageScaleComparisonRow, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _benchmark_lancedb_storage_profile(
    *,
    task: Mapping[str, Any],
    database_root: Path,
    table_name: str,
    warmup_iterations: int,
    measurement_iterations: int,
    unit_norm_tolerance: float = 1e-3,
) -> dict[str, object]:
    lancedb, pa = _require_lancedb()
    norm_error = _validate_unit_norm_vectors(task, tolerance=unit_norm_tolerance)

    if database_root.exists():
        shutil.rmtree(database_root)
    database_root.mkdir(parents=True, exist_ok=True)

    build_start = time.perf_counter()
    db = lancedb.connect(str(database_root))
    vector_dim = int(task["vector_dim"])
    schema = pa.schema(
        [
            pa.field("doc_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.list_(pa.float32(), vector_dim))),
        ]
    )
    table = db.create_table(
        table_name,
        data=_build_table_rows(task),
        schema=schema,
        mode="overwrite",
    )
    build_seconds = time.perf_counter() - build_start

    open_start = time.perf_counter()
    reopened_db = lancedb.connect(str(database_root))
    reopened_table = reopened_db.open_table(table_name)
    open_table_seconds = time.perf_counter() - open_start

    query_matrices = tuple(query["vectors"] for query in task["queries"])
    for _ in range(warmup_iterations):
        for query_matrix in query_matrices:
            _search_doc_ids(reopened_table, query_matrix, int(task["k"]))

    elapsed_seconds: list[float] = []
    ranked_doc_ids_by_query: list[tuple[str, ...]] = []
    for measurement_iteration in range(measurement_iterations):
        for query_index, query_matrix in enumerate(query_matrices):
            start = time.perf_counter()
            ranked_doc_ids = _search_doc_ids(
                reopened_table,
                query_matrix,
                int(task["k"]),
            )
            elapsed_seconds.append(time.perf_counter() - start)
            if measurement_iteration == 0:
                ranked_doc_ids_by_query.append(ranked_doc_ids)
            elif query_index >= len(ranked_doc_ids_by_query):
                raise AssertionError("ranked query capture drifted during measurement")

    metrics = summarize_ranked_task(
        task=task,
        ranked_doc_ids_by_query=ranked_doc_ids_by_query,
    )
    storage_byte_size = _directory_byte_size(database_root)
    stored_document_vector_count_total = sum(
        int(document["vector_count"]) for document in task["documents"]
    )

    return {
        "engine": "lancedb",
        "engine_version": str(lancedb.__version__),
        "primary_value": metrics.primary_value,
        "mean_search_seconds": float(sum(elapsed_seconds) / len(elapsed_seconds)),
        "storage_byte_size": storage_byte_size,
        "bytes_per_document": storage_byte_size / float(len(task["documents"])),
        "bytes_per_vector": storage_byte_size
        / float(stored_document_vector_count_total),
        "build_seconds": build_seconds,
        "open_table_seconds": open_table_seconds,
        "vector_unit_norm_max_error": norm_error,
        "stored_document_vector_count_total": stored_document_vector_count_total,
    }


def benchmark_storage_engine_scale_sweep(
    task: Mapping[str, Any],
    *,
    database_root: Path,
    table_prefix: str,
    target_document_counts: Sequence[int],
    warmup_iterations: int = 1,
    measurement_iterations: int = 3,
) -> StorageScaleComparisonSummary:
    if not target_document_counts:
        raise ValueError("target_document_counts must not be empty")

    (
        filtered_task,
        zero_document_vector_count,
        zero_query_vector_count,
        _stored_document_vector_count_total,
        _stored_query_vector_count_total,
    ) = _filter_zero_vectors(task)

    base_ranked_doc_ids_by_query = rank_task_with_kayak_exact(filtered_task)
    protected_doc_ids = protected_doc_ids_for_scale_sweep(
        filtered_task,
        ranked_doc_ids_by_query=base_ranked_doc_ids_by_query,
    )
    repeatable_doc_ids = choose_repeatable_distractor_doc_ids(
        filtered_task,
        protected_doc_ids=protected_doc_ids,
    )

    rows: list[StorageScaleComparisonRow] = []
    base_document_count = len(filtered_task["documents"])
    inflation_policy = "repeat_nonprotected_documents_with_unique_doc_ids"
    lancedb_engine_version = ""

    for target_document_count in sorted({int(value) for value in target_document_counts}):
        scaled = build_scaled_task_with_document_copies(
            filtered_task,
            target_document_count=target_document_count,
            repeatable_doc_ids=repeatable_doc_ids,
            inflation_policy=inflation_policy,
        )
        scaled_task = scaled.task
        kayak_summary = _run_native_kayak_storage_benchmark(scaled_task)
        lancedb_summary = _benchmark_lancedb_storage_profile(
            task=scaled_task,
            database_root=database_root / f"docs_{target_document_count}",
            table_name=f"{table_prefix}_{target_document_count}",
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        lancedb_engine_version = str(lancedb_summary["engine_version"])
        stored_document_vector_count_total = int(
            lancedb_summary["stored_document_vector_count_total"]
        )

        rows.append(
            StorageScaleComparisonRow(
                target_document_count=target_document_count,
                actual_document_count=len(scaled_task["documents"]),
                scale_factor_vs_base=float(len(scaled_task["documents"]))
                / float(base_document_count),
                duplicated_document_count=scaled.duplicated_document_count,
                stored_document_vector_count_total=stored_document_vector_count_total,
                kayak_storage_byte_size=int(kayak_summary["artifact_byte_size"]),
                kayak_bytes_per_document=float(
                    kayak_summary["artifact_bytes_per_document"]
                ),
                kayak_bytes_per_vector=float(
                    kayak_summary["artifact_bytes_per_vector"]
                ),
                kayak_build_seconds=float(kayak_summary["mean_build_seconds"]),
                kayak_load_seconds=float(kayak_summary["mean_load_seconds"]),
                kayak_search_seconds=float(kayak_summary["mean_search_seconds"]),
                kayak_primary_value=float(kayak_summary["primary_value"]),
                kayak_vector_payload_encoding=str(kayak_summary["encoding_kind"]),
                lancedb_storage_byte_size=int(lancedb_summary["storage_byte_size"]),
                lancedb_bytes_per_document=float(
                    lancedb_summary["bytes_per_document"]
                ),
                lancedb_bytes_per_vector=float(lancedb_summary["bytes_per_vector"]),
                lancedb_build_seconds=float(lancedb_summary["build_seconds"]),
                lancedb_open_table_seconds=float(
                    lancedb_summary["open_table_seconds"]
                ),
                lancedb_search_seconds=float(
                    lancedb_summary["mean_search_seconds"]
                ),
                lancedb_primary_value=float(lancedb_summary["primary_value"]),
                lancedb_storage_byte_ratio_vs_kayak=_safe_ratio(
                    float(lancedb_summary["storage_byte_size"]),
                    float(kayak_summary["artifact_byte_size"]),
                ),
                lancedb_build_seconds_ratio_vs_kayak=_safe_ratio(
                    float(lancedb_summary["build_seconds"]),
                    float(kayak_summary["mean_build_seconds"]),
                ),
                lancedb_search_seconds_ratio_vs_kayak=_safe_ratio(
                    float(lancedb_summary["mean_search_seconds"]),
                    float(kayak_summary["mean_search_seconds"]),
                ),
                lancedb_primary_ratio_vs_kayak=_safe_ratio(
                    float(lancedb_summary["primary_value"]),
                    float(kayak_summary["primary_value"]),
                ),
            )
        )

    return StorageScaleComparisonSummary(
        dataset_id=str(filtered_task["dataset_id"]),
        model_name=str(filtered_task["model_name"]),
        family=str(filtered_task["family"]),
        slice_name=str(filtered_task["slice_name"]),
        primary_metric=str(filtered_task["primary_metric"]),
        k=int(filtered_task["k"]),
        vector_dim=int(filtered_task["vector_dim"]),
        base_document_count=base_document_count,
        base_nominal_document_vector_count=int(
            filtered_task["nominal_document_vector_count"]
        ),
        base_zero_document_vector_count_filtered=zero_document_vector_count,
        base_zero_query_vector_count_filtered=zero_query_vector_count,
        protected_document_count=len(protected_doc_ids),
        repeatable_distractor_source_count=len(repeatable_doc_ids),
        inflation_policy=inflation_policy,
        kayak_storage_engine="kayak",
        kayak_storage_format="packed_index_binary_le",
        lancedb_storage_engine="lancedb",
        lancedb_storage_engine_version=lancedb_engine_version,
        rows=tuple(rows),
    )
