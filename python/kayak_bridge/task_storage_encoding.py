"""Owns task-json storage-encoding benchmarks for native Kayak packed storage.

This module keeps the storage-encoding question explicit and measurable:
- both encodings are benchmarked from the same filtered task JSON
- the native Mojo benchmark measures build, load, search, and artifact bytes
- the Python bundle reduces the result to direct f16-vs-native ratios
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PIXI_BIN = shutil.which("pixi")
MOJO_WITH_PIXI_PYTHON = REPO_ROOT / "scripts" / "run_mojo_with_pixi_python.sh"
TASK_JSON_STORAGE_COMPARE_INPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_encoding_compare_input.json"
)
TASK_JSON_STORAGE_COMPARE_OUTPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_encoding_compare_output.json"
)
VECTOR_PAYLOAD_ENCODING_BINARY_LE = "binary_le"
VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE = "binary_f16_le"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _write_task_json(path: Path, task: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _summary_by_encoding(
    summaries: Sequence[Mapping[str, object]],
    *,
    encoding_kind: str,
) -> Mapping[str, object]:
    for summary in summaries:
        if str(summary["encoding_kind"]) == encoding_kind:
            return summary
    raise ValueError(f"missing storage encoding summary for {encoding_kind}")


def benchmark_task_storage_encodings(
    task: Mapping[str, Any],
) -> list[dict[str, object]]:
    if PIXI_BIN is None:
        raise RuntimeError(
            "expected `pixi` on PATH to run the native Mojo benchmark"
        )
    if not MOJO_WITH_PIXI_PYTHON.exists():
        raise RuntimeError(
            "expected scripts/run_mojo_with_pixi_python.sh to run the native "
            "Mojo benchmark with the Pixi Python runtime"
        )

    _write_task_json(TASK_JSON_STORAGE_COMPARE_INPUT_PATH, task)
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
            "bash",
            str(MOJO_WITH_PIXI_PYTHON),
            "-I",
            ".",
            "benchmarks/task_json_storage_encoding_compare.mojo",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        env=subprocess_env,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "task_json_storage_encoding_compare benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not TASK_JSON_STORAGE_COMPARE_OUTPUT_PATH.exists():
        raise RuntimeError(
            "task_json_storage_encoding_compare benchmark did not write "
            f"{TASK_JSON_STORAGE_COMPARE_OUTPUT_PATH}"
        )

    with TASK_JSON_STORAGE_COMPARE_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("expected list payload from storage-encoding benchmark")
    return [dict(summary) for summary in payload]


@dataclass(frozen=True, slots=True)
class TaskStorageEncodingBundle:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    primary_metric: str
    query_count: int
    document_count: int
    vector_count: int
    vector_dim: int
    binary_le_primary_value: float
    binary_f16_le_primary_value: float
    primary_value_delta_f16_minus_binary: float
    binary_le_mean_build_seconds: float
    binary_f16_le_mean_build_seconds: float
    build_seconds_ratio_f16_vs_binary: float | None
    binary_le_mean_load_seconds: float
    binary_f16_le_mean_load_seconds: float
    load_seconds_ratio_f16_vs_binary: float | None
    binary_le_mean_search_seconds: float
    binary_f16_le_mean_search_seconds: float
    search_seconds_ratio_f16_vs_binary: float | None
    binary_le_artifact_byte_size: int
    binary_f16_le_artifact_byte_size: int
    artifact_byte_ratio_f16_vs_binary: float | None
    binary_le_artifact_bytes_per_document: float
    binary_f16_le_artifact_bytes_per_document: float
    bytes_per_document_ratio_f16_vs_binary: float | None
    binary_le_artifact_bytes_per_vector: float
    binary_f16_le_artifact_bytes_per_vector: float
    bytes_per_vector_ratio_f16_vs_binary: float | None

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def build_task_storage_encoding_bundle(
    summaries: Sequence[Mapping[str, object]],
) -> TaskStorageEncodingBundle:
    binary_summary = _summary_by_encoding(
        summaries,
        encoding_kind=VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )
    f16_summary = _summary_by_encoding(
        summaries,
        encoding_kind=VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    )

    return TaskStorageEncodingBundle(
        dataset_id=str(binary_summary["dataset_id"]),
        model_name=str(binary_summary["model_name"]),
        family=str(binary_summary["family"]),
        slice_name=str(binary_summary["slice_name"]),
        primary_metric=str(binary_summary["primary_metric"]),
        query_count=int(binary_summary["query_count"]),
        document_count=int(binary_summary["document_count"]),
        vector_count=int(binary_summary["vector_count"]),
        vector_dim=int(binary_summary["vector_dim"]),
        binary_le_primary_value=float(binary_summary["primary_value"]),
        binary_f16_le_primary_value=float(f16_summary["primary_value"]),
        primary_value_delta_f16_minus_binary=float(f16_summary["primary_value"])
        - float(binary_summary["primary_value"]),
        binary_le_mean_build_seconds=float(binary_summary["mean_build_seconds"]),
        binary_f16_le_mean_build_seconds=float(f16_summary["mean_build_seconds"]),
        build_seconds_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["mean_build_seconds"]),
            float(binary_summary["mean_build_seconds"]),
        ),
        binary_le_mean_load_seconds=float(binary_summary["mean_load_seconds"]),
        binary_f16_le_mean_load_seconds=float(f16_summary["mean_load_seconds"]),
        load_seconds_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["mean_load_seconds"]),
            float(binary_summary["mean_load_seconds"]),
        ),
        binary_le_mean_search_seconds=float(binary_summary["mean_search_seconds"]),
        binary_f16_le_mean_search_seconds=float(f16_summary["mean_search_seconds"]),
        search_seconds_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["mean_search_seconds"]),
            float(binary_summary["mean_search_seconds"]),
        ),
        binary_le_artifact_byte_size=int(binary_summary["artifact_byte_size"]),
        binary_f16_le_artifact_byte_size=int(f16_summary["artifact_byte_size"]),
        artifact_byte_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["artifact_byte_size"]),
            float(binary_summary["artifact_byte_size"]),
        ),
        binary_le_artifact_bytes_per_document=float(
            binary_summary["artifact_bytes_per_document"]
        ),
        binary_f16_le_artifact_bytes_per_document=float(
            f16_summary["artifact_bytes_per_document"]
        ),
        bytes_per_document_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["artifact_bytes_per_document"]),
            float(binary_summary["artifact_bytes_per_document"]),
        ),
        binary_le_artifact_bytes_per_vector=float(
            binary_summary["artifact_bytes_per_vector"]
        ),
        binary_f16_le_artifact_bytes_per_vector=float(
            f16_summary["artifact_bytes_per_vector"]
        ),
        bytes_per_vector_ratio_f16_vs_binary=_safe_ratio(
            float(f16_summary["artifact_bytes_per_vector"]),
            float(binary_summary["artifact_bytes_per_vector"]),
        ),
    )
