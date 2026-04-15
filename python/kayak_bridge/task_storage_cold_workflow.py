"""Owns cold-ish load-heavy storage workflow benchmarks for native Kayak.

This module answers a narrower question than the warm workflow trace:
- after explicit cache perturbation
- how expensive is one load + N exact searches session
- does binary_f16_le ever overtake binary_le under colder startup conditions

The cache boundary is explicit and user-visible through the selected policy.
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
MEMORY_PRESSURE_BIN = shutil.which("memory_pressure")
MOJO_WITH_PIXI_PYTHON = REPO_ROOT / "scripts" / "run_mojo_with_pixi_python.sh"
TASK_JSON_INPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_cold_workflow_input.json"
)
PREPARE_OUTPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_artifact_prepare_output.json"
)
SINGLE_SESSION_INPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_single_session_input.json"
)
SINGLE_SESSION_OUTPUT_PATH = (
    REPO_ROOT / ".cache" / "kayak" / "task_json_storage_single_session_output.json"
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


def _mojo_env() -> dict[str, str]:
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
    return subprocess_env


def load_single_session_config_json(path: str | Path) -> dict[str, int | str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("single session config payload must be a JSON object")
    return {
        "encoding_kind": str(payload["encoding_kind"]),
        "queries_per_load": int(payload["queries_per_load"]),
        "query_start": int(payload["query_start"]),
    }


def default_cold_queries_per_loads() -> tuple[int, ...]:
    return (1, 8, 32)


def _prepare_task_artifacts(task: Mapping[str, Any]) -> dict[str, object]:
    if PIXI_BIN is None:
        raise RuntimeError(
            "expected `pixi` on PATH to run the native Mojo benchmark"
        )
    if not MOJO_WITH_PIXI_PYTHON.exists():
        raise RuntimeError(
            "expected scripts/run_mojo_with_pixi_python.sh to run the native "
            "Mojo benchmark with the Pixi Python runtime"
        )

    _write_json(TASK_JSON_INPUT_PATH, task)
    completed = subprocess.run(
        [
            PIXI_BIN,
            "run",
            "bash",
            str(MOJO_WITH_PIXI_PYTHON),
            "-I",
            ".",
            "benchmarks/task_json_storage_artifact_prepare.mojo",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        env=_mojo_env(),
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "task_json_storage_artifact_prepare benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    with PREPARE_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _apply_memory_pressure_percent_free(
    *,
    percent_free: int,
    sample_seconds: int = 1,
    hysteresis_seconds: int = 1,
) -> None:
    if MEMORY_PRESSURE_BIN is None:
        raise RuntimeError(
            "memory_pressure is not available; cannot run the configured cold policy"
        )
    completed = subprocess.run(
        [
            MEMORY_PRESSURE_BIN,
            "-Q",
            "-p",
            str(int(percent_free)),
            "-s",
            str(int(sample_seconds)),
            "-y",
            str(int(hysteresis_seconds)),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "memory_pressure cold policy failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _run_single_session(
    task: Mapping[str, Any],
    *,
    encoding_kind: str,
    queries_per_load: int,
    query_start: int,
) -> dict[str, object]:
    if PIXI_BIN is None:
        raise RuntimeError(
            "expected `pixi` on PATH to run the native Mojo benchmark"
        )
    if not MOJO_WITH_PIXI_PYTHON.exists():
        raise RuntimeError(
            "expected scripts/run_mojo_with_pixi_python.sh to run the native "
            "Mojo benchmark with the Pixi Python runtime"
        )

    _write_json(TASK_JSON_INPUT_PATH, task)
    _write_json(
        SINGLE_SESSION_INPUT_PATH,
        {
            "encoding_kind": encoding_kind,
            "queries_per_load": int(queries_per_load),
            "query_start": int(query_start),
        },
    )
    completed = subprocess.run(
        [
            PIXI_BIN,
            "run",
            "bash",
            str(MOJO_WITH_PIXI_PYTHON),
            "-I",
            ".",
            "benchmarks/task_json_storage_single_session.mojo",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        env=_mojo_env(),
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "task_json_storage_single_session benchmark failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    with SINGLE_SESSION_OUTPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True, slots=True)
class ColdWorkflowRun:
    encoding_kind: str
    queries_per_load: int
    repetition: int
    query_start: int
    session_seconds: float


@dataclass(frozen=True, slots=True)
class ColdWorkflowPoint:
    queries_per_load: int
    repetitions: int
    binary_le_mean_session_seconds: float
    binary_f16_le_mean_session_seconds: float
    session_seconds_ratio_f16_vs_binary: float | None
    session_seconds_delta_f16_minus_binary: float


@dataclass(frozen=True, slots=True)
class TaskStorageColdWorkflowBundle:
    dataset_id: str
    model_name: str
    family: str
    slice_name: str
    cold_policy: str
    cold_policy_percent_free: int
    cold_policy_sample_seconds: int
    cold_policy_hysteresis_seconds: int
    repetitions: int
    first_queries_per_load_with_f16_not_slower_total_session: int | None
    points: tuple[ColdWorkflowPoint, ...]
    runs: tuple[ColdWorkflowRun, ...]

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def benchmark_task_storage_cold_workflow(
    task: Mapping[str, Any],
    *,
    queries_per_loads: Sequence[int],
    repetitions: int,
    percent_free: int = 60,
    sample_seconds: int = 1,
    hysteresis_seconds: int = 1,
) -> TaskStorageColdWorkflowBundle:
    normalized_query_counts = tuple(int(value) for value in queries_per_loads)
    if not normalized_query_counts:
        raise ValueError("queries_per_loads must not be empty")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    prepare_summary = _prepare_task_artifacts(task)
    query_count = len(task["queries"])
    if query_count <= 0:
        raise ValueError("task must contain at least one query")

    runs: list[ColdWorkflowRun] = []
    first_crossover: int | None = None
    points: list[ColdWorkflowPoint] = []
    for queries_per_load in normalized_query_counts:
        mean_by_encoding: dict[str, float] = {}
        for encoding_kind in (
            VECTOR_PAYLOAD_ENCODING_BINARY_LE,
            VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
        ):
            session_seconds: list[float] = []
            for repetition in range(repetitions):
                query_start = (repetition * queries_per_load) % query_count
                _apply_memory_pressure_percent_free(
                    percent_free=percent_free,
                    sample_seconds=sample_seconds,
                    hysteresis_seconds=hysteresis_seconds,
                )
                session_summary = _run_single_session(
                    task,
                    encoding_kind=encoding_kind,
                    queries_per_load=queries_per_load,
                    query_start=query_start,
                )
                session_seconds.append(float(session_summary["session_seconds"]))
                runs.append(
                    ColdWorkflowRun(
                        encoding_kind=encoding_kind,
                        queries_per_load=queries_per_load,
                        repetition=repetition,
                        query_start=query_start,
                        session_seconds=float(session_summary["session_seconds"]),
                    )
                )
            mean_by_encoding[encoding_kind] = sum(session_seconds) / float(
                len(session_seconds)
            )
        ratio = _safe_ratio(
            mean_by_encoding[VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE],
            mean_by_encoding[VECTOR_PAYLOAD_ENCODING_BINARY_LE],
        )
        if first_crossover is None and ratio is not None and ratio <= 1.0:
            first_crossover = queries_per_load
        points.append(
            ColdWorkflowPoint(
                queries_per_load=queries_per_load,
                repetitions=repetitions,
                binary_le_mean_session_seconds=mean_by_encoding[
                    VECTOR_PAYLOAD_ENCODING_BINARY_LE
                ],
                binary_f16_le_mean_session_seconds=mean_by_encoding[
                    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE
                ],
                session_seconds_ratio_f16_vs_binary=ratio,
                session_seconds_delta_f16_minus_binary=mean_by_encoding[
                    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE
                ]
                - mean_by_encoding[VECTOR_PAYLOAD_ENCODING_BINARY_LE],
            )
        )

    return TaskStorageColdWorkflowBundle(
        dataset_id=str(prepare_summary["dataset_id"]),
        model_name=str(prepare_summary["model_name"]),
        family=str(prepare_summary["family"]),
        slice_name=str(prepare_summary["slice_name"]),
        cold_policy="memory_pressure_percent_free",
        cold_policy_percent_free=int(percent_free),
        cold_policy_sample_seconds=int(sample_seconds),
        cold_policy_hysteresis_seconds=int(hysteresis_seconds),
        repetitions=repetitions,
        first_queries_per_load_with_f16_not_slower_total_session=first_crossover,
        points=tuple(points),
        runs=tuple(runs),
    )
