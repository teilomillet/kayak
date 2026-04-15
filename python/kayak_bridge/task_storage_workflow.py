"""Owns load-heavy task-json storage workflow benchmarks for native Kayak.

This module answers a narrower question than storage bytes:
- given an already persisted packed index
- how expensive is one session that loads the index and runs N exact searches
- how does that tradeoff differ between binary_le and binary_f16_le
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
TASK_JSON_STORAGE_WORKFLOW_INPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_workflow_compare_input.json"
)
TASK_JSON_STORAGE_WORKFLOW_QUERY_COUNTS_PATH = (
    REPO_ROOT
    / ".cache"
    / "kayak"
    / "task_json_storage_workflow_compare_query_counts.json"
)
TASK_JSON_STORAGE_WORKFLOW_OUTPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_workflow_compare_output.json"
)
VECTOR_PAYLOAD_ENCODING_BINARY_LE = "binary_le"
VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE = "binary_f16_le"


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_queries_per_load_json(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("queries_per_load payload must be a JSON list")
    values = [int(value) for value in payload]
    if any(value <= 0 for value in values):
        raise ValueError("queries_per_load values must be positive")
    return values


def default_queries_per_loads(task_query_count: int) -> tuple[int, ...]:
    if task_query_count <= 0:
        raise ValueError("task_query_count must be positive")
    counts: list[int] = []
    for value in (1, 2, 4, 8, 16, 32, task_query_count):
        value = int(value)
        if value <= 0 or value in counts:
            continue
        counts.append(value)
    return tuple(counts)


def _summary_by_key(
    summaries: Sequence[Mapping[str, object]],
    *,
    encoding_kind: str,
    queries_per_load: int,
) -> Mapping[str, object]:
    for summary in summaries:
        if (
            str(summary["encoding_kind"]) == encoding_kind
            and int(summary["queries_per_load"]) == queries_per_load
        ):
            return summary
    raise ValueError(
        "missing storage workflow summary for "
        f"{encoding_kind} queries_per_load={queries_per_load}"
    )


def benchmark_task_storage_workflow(
    task: Mapping[str, Any],
    *,
    queries_per_loads: Sequence[int],
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

    normalized_query_counts = [int(value) for value in queries_per_loads]
    if not normalized_query_counts:
        raise ValueError("queries_per_loads must not be empty")
    if any(value <= 0 for value in normalized_query_counts):
        raise ValueError("queries_per_loads must contain only positive integers")

    _write_json(TASK_JSON_STORAGE_WORKFLOW_INPUT_PATH, task)
    _write_json(TASK_JSON_STORAGE_WORKFLOW_QUERY_COUNTS_PATH, normalized_query_counts)
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
            "benchmarks/task_json_storage_workflow_compare.mojo",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        env=subprocess_env,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "task_json_storage_workflow_compare benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    if not TASK_JSON_STORAGE_WORKFLOW_OUTPUT_PATH.exists():
        raise RuntimeError(
            "task_json_storage_workflow_compare benchmark did not write "
            f"{TASK_JSON_STORAGE_WORKFLOW_OUTPUT_PATH}"
        )

    with TASK_JSON_STORAGE_WORKFLOW_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise TypeError("expected list payload from storage workflow benchmark")
    return [dict(summary) for summary in payload]


@dataclass(frozen=True, slots=True)
class TaskStorageWorkflowPoint:
    queries_per_load: int
    binary_le_mean_session_seconds: float
    binary_f16_le_mean_session_seconds: float
    session_seconds_ratio_f16_vs_binary: float | None
    session_seconds_delta_f16_minus_binary: float
    binary_le_mean_session_seconds_per_query: float
    binary_f16_le_mean_session_seconds_per_query: float
    session_seconds_per_query_ratio_f16_vs_binary: float | None


@dataclass(frozen=True, slots=True)
class TaskStorageWorkflowBundle:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    query_count: int
    document_count: int
    vector_count: int
    vector_dim: int
    first_queries_per_load_with_f16_not_slower_total_session: int | None
    first_queries_per_load_with_f16_not_slower_per_query: int | None
    points: tuple[TaskStorageWorkflowPoint, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def build_task_storage_workflow_bundle(
    summaries: Sequence[Mapping[str, object]],
    *,
    queries_per_loads: Sequence[int],
) -> TaskStorageWorkflowBundle:
    normalized_query_counts = tuple(int(value) for value in queries_per_loads)
    if not normalized_query_counts:
        raise ValueError("queries_per_loads must not be empty")

    first_total_session_crossover: int | None = None
    first_per_query_crossover: int | None = None
    points: list[TaskStorageWorkflowPoint] = []

    for queries_per_load in normalized_query_counts:
        binary_summary = _summary_by_key(
            summaries,
            encoding_kind=VECTOR_PAYLOAD_ENCODING_BINARY_LE,
            queries_per_load=queries_per_load,
        )
        f16_summary = _summary_by_key(
            summaries,
            encoding_kind=VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
            queries_per_load=queries_per_load,
        )
        total_ratio = _safe_ratio(
            float(f16_summary["mean_session_seconds"]),
            float(binary_summary["mean_session_seconds"]),
        )
        per_query_ratio = _safe_ratio(
            float(f16_summary["mean_session_seconds_per_query"]),
            float(binary_summary["mean_session_seconds_per_query"]),
        )
        if first_total_session_crossover is None and total_ratio is not None and total_ratio <= 1.0:
            first_total_session_crossover = queries_per_load
        if first_per_query_crossover is None and per_query_ratio is not None and per_query_ratio <= 1.0:
            first_per_query_crossover = queries_per_load

        points.append(
            TaskStorageWorkflowPoint(
                queries_per_load=queries_per_load,
                binary_le_mean_session_seconds=float(
                    binary_summary["mean_session_seconds"]
                ),
                binary_f16_le_mean_session_seconds=float(
                    f16_summary["mean_session_seconds"]
                ),
                session_seconds_ratio_f16_vs_binary=total_ratio,
                session_seconds_delta_f16_minus_binary=float(
                    f16_summary["mean_session_seconds"]
                )
                - float(binary_summary["mean_session_seconds"]),
                binary_le_mean_session_seconds_per_query=float(
                    binary_summary["mean_session_seconds_per_query"]
                ),
                binary_f16_le_mean_session_seconds_per_query=float(
                    f16_summary["mean_session_seconds_per_query"]
                ),
                session_seconds_per_query_ratio_f16_vs_binary=per_query_ratio,
            )
        )

    first_summary = summaries[0]
    return TaskStorageWorkflowBundle(
        dataset_id=str(first_summary["dataset_id"]),
        model_name=str(first_summary["model_name"]),
        family=str(first_summary["family"]),
        slice_name=str(first_summary["slice_name"]),
        query_count=int(first_summary["query_count"]),
        document_count=int(first_summary["document_count"]),
        vector_count=int(first_summary["vector_count"]),
        vector_dim=int(first_summary["vector_dim"]),
        first_queries_per_load_with_f16_not_slower_total_session=(
            first_total_session_crossover
        ),
        first_queries_per_load_with_f16_not_slower_per_query=(
            first_per_query_crossover
        ),
        points=tuple(points),
    )
