"""Internal loader for the GPU-targeted Mojo i8 rerank bridge."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Sequence

import numpy as np

from .plaid_approx import KayakPlaidI8PayloadSnapshot

from .cache_paths import PYTHON_MOJO_CACHE, REPO_ROOT, configure_local_caches
from .mojo_exact_cpu import (
    _compiled_extension_suffix,
    _detect_mojo_command,
    _diagnostics_hint,
    _hash_inputs,
    _mojo_sources,
    _mojo_subprocess_env,
    _build_mojopkg,
)


configure_local_caches()

MODULE_SHORT_NAME = "_mojo_gpu_i8_rerank_bindings"
MODULE_FULL_NAME = f"kayak_bridge.{MODULE_SHORT_NAME}"
BINDING_SOURCE = Path(__file__).with_name(f"{MODULE_SHORT_NAME}.mojo")


@dataclass(frozen=True, slots=True)
class MojoGpuI8RerankBridgeResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    host_to_device_mean_seconds: float
    kernel_mean_seconds: float
    device_to_host_mean_seconds: float
    score_delta_max_abs: float
    candidate_score_count: int
    scores: tuple[float, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "host_to_device_mean_seconds": self.host_to_device_mean_seconds,
            "kernel_mean_seconds": self.kernel_mean_seconds,
            "device_to_host_mean_seconds": self.device_to_host_mean_seconds,
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count": self.candidate_score_count,
            "score_count": len(self.scores),
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8PreparedSessionResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    prepare_host_to_device_mean_seconds: float
    score_host_to_device_mean_seconds: float
    kernel_mean_seconds: float
    device_to_host_mean_seconds: float
    score_delta_max_abs: float
    candidate_score_count: int
    scores: tuple[float, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "prepare_host_to_device_mean_seconds": (
                self.prepare_host_to_device_mean_seconds
            ),
            "score_host_to_device_mean_seconds": (
                self.score_host_to_device_mean_seconds
            ),
            "kernel_mean_seconds": self.kernel_mean_seconds,
            "device_to_host_mean_seconds": self.device_to_host_mean_seconds,
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count": self.candidate_score_count,
            "score_count": len(self.scores),
        }


def gpu_extension_device_probe(*, target_accelerator: str) -> str:
    module = load_module(target_accelerator=target_accelerator)
    return str(module.gpu_extension_device_probe())


def score_i8_real_payload_once(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
) -> MojoGpuI8RerankBridgeResult:
    marshalling_started_at = time.perf_counter()
    query_values = _float32_values(queries)
    token_codes = _int8_values(payload.token_codes)
    token_scales = _float32_values(payload.token_scales)
    doc_offsets = _int_values(payload.doc_offsets)
    candidate_positions = _flatten_int_rows(
        candidate_positions_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_query",
    )
    reference_scores = _flatten_float_rows(
        reference_scores_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_query",
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        query_values,
        token_codes,
        token_scales,
        doc_offsets,
        candidate_positions,
        reference_scores,
        int(shape.query_count),
        int(shape.query_vector_count),
        int(shape.document_count),
        int(shape.document_vector_count),
        int(candidate_k),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.score_i8_real_payload_once(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 6:
        raise RuntimeError("GPU i8 rerank bridge returned an unexpected result shape")

    scores = tuple(float(value) for value in raw_result[5])
    return MojoGpuI8RerankBridgeResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        host_to_device_mean_seconds=float(raw_result[0]),
        kernel_mean_seconds=float(raw_result[1]),
        device_to_host_mean_seconds=float(raw_result[2]),
        score_delta_max_abs=float(raw_result[3]),
        candidate_score_count=int(raw_result[4]),
        scores=scores,
    )


def profile_i8_prepared_payload_session(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8PreparedSessionResult:
    marshalling_started_at = time.perf_counter()
    query_values = _float32_values(queries)
    token_codes = _int8_values(payload.token_codes)
    token_scales = _float32_values(payload.token_scales)
    doc_offsets = _int_values(payload.doc_offsets)
    candidate_positions = _flatten_int_rows(
        candidate_positions_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_query",
    )
    reference_scores = _flatten_float_rows(
        reference_scores_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_query",
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        query_values,
        token_codes,
        token_scales,
        doc_offsets,
        candidate_positions,
        reference_scores,
        int(shape.query_count),
        int(shape.query_vector_count),
        int(shape.document_count),
        int(shape.document_vector_count),
        int(candidate_k),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_prepared_payload_session(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 7:
        raise RuntimeError(
            "GPU i8 prepared rerank bridge returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[6])
    return MojoGpuI8PreparedSessionResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        prepare_host_to_device_mean_seconds=float(raw_result[0]),
        score_host_to_device_mean_seconds=float(raw_result[1]),
        kernel_mean_seconds=float(raw_result[2]),
        device_to_host_mean_seconds=float(raw_result[3]),
        score_delta_max_abs=float(raw_result[4]),
        candidate_score_count=int(raw_result[5]),
        scores=scores,
    )


def profile_i8_prepared_payload_session_ndarray(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8PreparedSessionResult:
    marshalling_started_at = time.perf_counter()
    query_values = _float32_array(queries)
    token_codes = _int8_array(payload.token_codes)
    token_scales = _float32_array(payload.token_scales)
    doc_offsets = _int64_array(payload.doc_offsets)
    candidate_positions = _flatten_int_rows_array(
        candidate_positions_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_query",
    )
    reference_scores = _flatten_float_rows_array(
        reference_scores_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_query",
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        query_values,
        token_codes,
        token_scales,
        doc_offsets,
        candidate_positions,
        reference_scores,
        int(shape.query_count),
        int(shape.query_vector_count),
        int(shape.document_count),
        int(shape.document_vector_count),
        int(candidate_k),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_prepared_payload_session(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 7:
        raise RuntimeError(
            "GPU i8 prepared ndarray bridge returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[6])
    return MojoGpuI8PreparedSessionResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        prepare_host_to_device_mean_seconds=float(raw_result[0]),
        score_host_to_device_mean_seconds=float(raw_result[1]),
        kernel_mean_seconds=float(raw_result[2]),
        device_to_host_mean_seconds=float(raw_result[3]),
        score_delta_max_abs=float(raw_result[4]),
        candidate_score_count=int(raw_result[5]),
        scores=scores,
    )


def profile_i8_prepared_payload_session_addresses(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8PreparedSessionResult:
    marshalling_started_at = time.perf_counter()
    query_values = _float32_array(queries)
    token_codes = _int8_array(payload.token_codes)
    token_scales = _float32_array(payload.token_scales)
    doc_offsets = _int64_array(payload.doc_offsets)
    candidate_positions = _flatten_int_rows_array(
        candidate_positions_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_query",
    )
    reference_scores = _flatten_float_rows_array(
        reference_scores_by_query,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_query",
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        _array_address(query_values),
        _array_address(token_codes),
        _array_address(token_scales),
        _array_address(doc_offsets),
        _array_address(candidate_positions),
        _array_address(reference_scores),
        int(shape.query_count),
        int(shape.query_vector_count),
        int(shape.document_count),
        int(shape.document_vector_count),
        int(candidate_k),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_prepared_payload_session_addresses(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 7:
        raise RuntimeError(
            "GPU i8 prepared address bridge returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[6])
    return MojoGpuI8PreparedSessionResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        prepare_host_to_device_mean_seconds=float(raw_result[0]),
        score_host_to_device_mean_seconds=float(raw_result[1]),
        kernel_mean_seconds=float(raw_result[2]),
        device_to_host_mean_seconds=float(raw_result[3]),
        score_delta_max_abs=float(raw_result[4]),
        candidate_score_count=int(raw_result[5]),
        scores=scores,
    )


@lru_cache(maxsize=None)
def load_module(*, target_accelerator: str) -> ModuleType:
    if not target_accelerator:
        raise ValueError("target_accelerator must be explicit for GPU bridge loading")

    cache_key = _cache_key(target_accelerator)
    mojopkg_path = _build_mojopkg(cache_key)
    extension_path = _build_extension(
        cache_key=cache_key,
        mojopkg_path=mojopkg_path,
        target_accelerator=target_accelerator,
    )
    return _load_extension(extension_path)


def _cache_key(target_accelerator: str) -> str:
    return _hash_inputs(
        [BINDING_SOURCE, *_mojo_sources()],
        extra_tokens=(
            "gpu_i8_rerank_bridge_v1",
            f"target_accelerator:{target_accelerator}",
        ),
    )


def _build_extension(
    *,
    cache_key: str,
    mojopkg_path: Path,
    target_accelerator: str,
) -> Path:
    extension_path = (
        PYTHON_MOJO_CACHE
        / cache_key
        / f"{MODULE_SHORT_NAME}-{cache_key}{_compiled_extension_suffix()}"
    )
    extension_path.parent.mkdir(parents=True, exist_ok=True)
    if extension_path.exists():
        return extension_path

    command = [
        *_detect_mojo_command(),
        "build",
        str(BINDING_SOURCE),
        "--emit",
        "shared-lib",
        "--target-accelerator",
        target_accelerator,
        "-o",
        str(extension_path),
        "-I",
        str(mojopkg_path.parent),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=_mojo_subprocess_env(command),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to build Mojo GPU i8 rerank extension\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"{_diagnostics_hint()}"
        )

    return extension_path


def _load_extension(extension_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        MODULE_FULL_NAME,
        extension_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"failed to create Python import spec for {extension_path}"
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_FULL_NAME] = module
    spec.loader.exec_module(module)
    return module


def _float32_values(values: Any) -> list[float]:
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1).tolist()


def _float32_array(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1)


def _int8_values(values: Any) -> list[int]:
    return np.ascontiguousarray(values, dtype=np.int8).reshape(-1).tolist()


def _int8_array(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int8).reshape(-1)


def _int_values(values: Any) -> list[int]:
    return [int(value) for value in np.asarray(values).reshape(-1)]


def _int64_array(values: Any) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64).reshape(-1)


def _array_address(values: np.ndarray) -> int:
    return int(values.ctypes.data)


def _flatten_int_rows(
    rows: Sequence[Sequence[int]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> list[int]:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values: list[int] = []
    for row in rows:
        if len(row) != expected_cols:
            raise ValueError(f"{name} rows must match candidate_k")
        values.extend(int(value) for value in row)
    return values


def _flatten_int_rows_array(
    rows: Sequence[Sequence[int]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1)
    if values.size != expected_rows * expected_cols:
        raise ValueError(f"{name} rows must match candidate_k")
    return values


def _flatten_float_rows(
    rows: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> list[float]:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values: list[float] = []
    for row in rows:
        if len(row) != expected_cols:
            raise ValueError(f"{name} rows must match candidate_k")
        values.extend(float(value) for value in row)
    return values


def _flatten_float_rows_array(
    rows: Sequence[Sequence[float]],
    *,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} row count must match query_count")
    values = np.ascontiguousarray(rows, dtype=np.float32).reshape(-1)
    if values.size != expected_rows * expected_cols:
        raise ValueError(f"{name} rows must match candidate_k")
    return values
