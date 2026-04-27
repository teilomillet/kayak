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

from .plaid_approx import (
    KayakPlaidI8PayloadSnapshot,
    KayakPlaidI8SelectedCentroids,
)

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


@dataclass(frozen=True, slots=True)
class MojoGpuI8AddressServeResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    score_delta_max_abs: float
    candidate_score_count: int
    scores: tuple[float, ...]

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count": self.candidate_score_count,
            "score_count": len(self.scores),
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8AddressTopKResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    score_delta_max_abs: float
    candidate_score_count: int
    top_k: int
    topk_position_match_count: int
    positions: tuple[int, ...]
    scores: tuple[float, ...]

    @property
    def topk_position_count(self) -> int:
        return len(self.positions)

    @property
    def topk_position_agreement(self) -> float:
        if not self.positions:
            return 0.0
        return self.topk_position_match_count / float(len(self.positions))

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count": self.candidate_score_count,
            "top_k": self.top_k,
            "topk_position_count": self.topk_position_count,
            "topk_position_match_count": self.topk_position_match_count,
            "topk_position_agreement": self.topk_position_agreement,
            "score_count": len(self.scores),
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8AddressTopKNoReferenceResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    candidate_score_count: int
    top_k: int
    positions: tuple[int, ...]
    scores: tuple[float, ...]

    @property
    def topk_position_count(self) -> int:
        return len(self.positions)

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "candidate_score_count": self.candidate_score_count,
            "top_k": self.top_k,
            "topk_position_count": self.topk_position_count,
            "score_count": len(self.scores),
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8CandidateGenerationPayloadResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    mojo_host_ingest_mean_seconds: float
    host_to_device_mean_seconds: float
    device_to_host_mean_seconds: float
    copy_mismatch_count: int
    offset_violation_count: int
    doc_index_out_of_range_count: int
    centroid_count: int
    posting_count: int
    document_count: int
    byte_counts: dict[str, int]

    @property
    def total_payload_bytes(self) -> int:
        return sum(self.byte_counts.values())

    @property
    def payload_agreement_ok(self) -> bool:
        return self.copy_mismatch_count == 0

    @property
    def posting_invariants_ok(self) -> bool:
        return (
            self.offset_violation_count == 0
            and self.doc_index_out_of_range_count == 0
        )

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "mojo_host_ingest_mean_seconds": (
                self.mojo_host_ingest_mean_seconds
            ),
            "host_to_device_mean_seconds": self.host_to_device_mean_seconds,
            "device_to_host_mean_seconds": self.device_to_host_mean_seconds,
            "copy_mismatch_count": self.copy_mismatch_count,
            "offset_violation_count": self.offset_violation_count,
            "doc_index_out_of_range_count": self.doc_index_out_of_range_count,
            "payload_agreement_ok": self.payload_agreement_ok,
            "posting_invariants_ok": self.posting_invariants_ok,
            "centroid_count": self.centroid_count,
            "posting_count": self.posting_count,
            "document_count": self.document_count,
            "byte_counts": self.byte_counts,
            "total_payload_bytes": self.total_payload_bytes,
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8SelectedPostingTraversalResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    mojo_host_ingest_mean_seconds: float
    payload_host_to_device_mean_seconds: float
    selected_host_to_device_mean_seconds: float
    kernel_mean_seconds: float
    device_to_host_mean_seconds: float
    selected_position_out_of_range_count: int
    selected_offset_violation_count: int
    doc_mismatch_count: int
    doc_index_out_of_range_count: int
    score_delta_max_abs: float
    selected_centroid_count: int
    expanded_posting_count: int
    document_count: int

    @property
    def traversal_agreement_ok(self) -> bool:
        return (
            self.selected_position_out_of_range_count == 0
            and self.selected_offset_violation_count == 0
            and self.doc_mismatch_count == 0
            and self.doc_index_out_of_range_count == 0
            and self.score_delta_max_abs == 0.0
        )

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "mojo_host_ingest_mean_seconds": (
                self.mojo_host_ingest_mean_seconds
            ),
            "payload_host_to_device_mean_seconds": (
                self.payload_host_to_device_mean_seconds
            ),
            "selected_host_to_device_mean_seconds": (
                self.selected_host_to_device_mean_seconds
            ),
            "kernel_mean_seconds": self.kernel_mean_seconds,
            "device_to_host_mean_seconds": self.device_to_host_mean_seconds,
            "selected_position_out_of_range_count": (
                self.selected_position_out_of_range_count
            ),
            "selected_offset_violation_count": (
                self.selected_offset_violation_count
            ),
            "doc_mismatch_count": self.doc_mismatch_count,
            "doc_index_out_of_range_count": self.doc_index_out_of_range_count,
            "score_delta_max_abs": self.score_delta_max_abs,
            "traversal_agreement_ok": self.traversal_agreement_ok,
            "selected_centroid_count": self.selected_centroid_count,
            "expanded_posting_count": self.expanded_posting_count,
            "document_count": self.document_count,
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8SelectedPostingAccumulationResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    mojo_host_ingest_mean_seconds: float
    payload_host_to_device_mean_seconds: float
    selected_host_to_device_mean_seconds: float
    kernel_mean_seconds: float
    device_to_host_mean_seconds: float
    host_topk_mean_seconds: float
    selected_position_out_of_range_count: int
    doc_index_out_of_range_count: int
    score_mismatch_count: int
    score_delta_max_abs: float
    topk_position_mismatch_count: int
    top_k: int
    topk_position_count: int
    selected_centroid_count: int
    document_score_count: int
    document_count: int

    @property
    def accumulation_agreement_ok(self) -> bool:
        return (
            self.selected_position_out_of_range_count == 0
            and self.doc_index_out_of_range_count == 0
            and self.score_mismatch_count == 0
            and self.topk_position_mismatch_count == 0
        )

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "mojo_host_ingest_mean_seconds": (
                self.mojo_host_ingest_mean_seconds
            ),
            "payload_host_to_device_mean_seconds": (
                self.payload_host_to_device_mean_seconds
            ),
            "selected_host_to_device_mean_seconds": (
                self.selected_host_to_device_mean_seconds
            ),
            "kernel_mean_seconds": self.kernel_mean_seconds,
            "device_to_host_mean_seconds": self.device_to_host_mean_seconds,
            "host_topk_mean_seconds": self.host_topk_mean_seconds,
            "selected_position_out_of_range_count": (
                self.selected_position_out_of_range_count
            ),
            "doc_index_out_of_range_count": self.doc_index_out_of_range_count,
            "score_mismatch_count": self.score_mismatch_count,
            "score_delta_max_abs": self.score_delta_max_abs,
            "topk_position_mismatch_count": (
                self.topk_position_mismatch_count
            ),
            "top_k": self.top_k,
            "topk_position_count": self.topk_position_count,
            "accumulation_agreement_ok": self.accumulation_agreement_ok,
            "selected_centroid_count": self.selected_centroid_count,
            "document_score_count": self.document_score_count,
            "document_count": self.document_count,
        }


@dataclass(frozen=True, slots=True)
class _SelectedPostingTraversalReference:
    selected_positions: np.ndarray
    selected_scores: np.ndarray
    selected_posting_offsets: np.ndarray
    expected_doc_indices: np.ndarray
    expected_scores: np.ndarray

    @property
    def selected_centroid_count(self) -> int:
        return int(self.selected_positions.size)

    @property
    def expanded_posting_count(self) -> int:
        return int(self.expected_doc_indices.size)


@dataclass(frozen=True, slots=True)
class _SelectedPostingAccumulationReference:
    selected_positions: np.ndarray
    selected_scores: np.ndarray
    expected_document_scores: np.ndarray

    @property
    def selected_centroid_count(self) -> int:
        return int(self.selected_positions.size)

    @property
    def document_score_count(self) -> int:
        return int(self.expected_document_scores.size)


@dataclass(frozen=True, slots=True)
class MojoGpuI8AddressResidentSessionResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    score_delta_max_abs: float
    candidate_score_count: int
    session_iterations: int
    scores: tuple[float, ...]

    @property
    def extension_call_seconds_per_iteration(self) -> float:
        return self.extension_call_seconds / float(self.session_iterations)

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "extension_call_seconds_per_iteration": (
                self.extension_call_seconds_per_iteration
            ),
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count": self.candidate_score_count,
            "session_iterations": self.session_iterations,
            "score_count": len(self.scores),
        }


@dataclass(frozen=True, slots=True)
class MojoGpuI8AddressMultiWindowSessionResult:
    host_marshalling_seconds: float
    extension_call_seconds: float
    score_delta_max_abs: float
    candidate_score_count_per_window: int
    window_count: int
    scores: tuple[float, ...]

    @property
    def extension_call_seconds_per_window(self) -> float:
        return self.extension_call_seconds / float(self.window_count)

    @property
    def candidate_score_count_total(self) -> int:
        return self.candidate_score_count_per_window * self.window_count

    def to_json_ready(self) -> dict[str, object]:
        return {
            "host_marshalling_seconds": self.host_marshalling_seconds,
            "extension_call_seconds": self.extension_call_seconds,
            "extension_call_seconds_per_window": (
                self.extension_call_seconds_per_window
            ),
            "score_delta_max_abs": self.score_delta_max_abs,
            "candidate_score_count_per_window": (
                self.candidate_score_count_per_window
            ),
            "candidate_score_count_total": self.candidate_score_count_total,
            "window_count": self.window_count,
            "score_count": len(self.scores),
        }


@dataclass(slots=True)
class MojoGpuI8AddressSessionHandle:
    target_accelerator: str
    handle: int
    shape: Any
    candidate_k: int
    prepare_host_marshalling_seconds: float
    prepare_extension_call_seconds: float
    _closed: bool = False

    def score(
        self,
        *,
        queries: np.ndarray,
        candidate_positions_by_query: Sequence[Sequence[int]],
        reference_scores_by_query: Sequence[Sequence[float]],
    ) -> MojoGpuI8AddressServeResult:
        if self._closed or self.handle == 0:
            raise RuntimeError("GPU i8 address session handle is closed")

        marshalling_started_at = time.perf_counter()
        query_values = _float32_array(queries)
        expected_query_values = (
            int(self.shape.query_count)
            * int(self.shape.query_vector_count)
            * int(self.shape.vector_dim)
        )
        if query_values.size != expected_query_values:
            raise ValueError("queries shape must match the prepared handle shape")
        candidate_positions = _flatten_int_rows_array(
            candidate_positions_by_query,
            expected_rows=int(self.shape.query_count),
            expected_cols=int(self.candidate_k),
            name="candidate_positions_by_query",
        )
        reference_scores = _flatten_float_rows_array(
            reference_scores_by_query,
            expected_rows=int(self.shape.query_count),
            expected_cols=int(self.candidate_k),
            name="reference_scores_by_query",
        )
        host_marshalling_seconds = time.perf_counter() - marshalling_started_at

        module = load_module(target_accelerator=self.target_accelerator)
        request = [
            int(self.handle),
            _array_address(query_values),
            _array_address(candidate_positions),
            _array_address(reference_scores),
        ]
        extension_started_at = time.perf_counter()
        raw_result = module.score_i8_address_session_handle(request)
        extension_call_seconds = time.perf_counter() - extension_started_at

        if len(raw_result) != 3:
            raise RuntimeError(
                "GPU i8 address session handle returned an unexpected result shape"
            )

        scores = tuple(float(value) for value in raw_result[2])
        return MojoGpuI8AddressServeResult(
            host_marshalling_seconds=host_marshalling_seconds,
            extension_call_seconds=extension_call_seconds,
            score_delta_max_abs=float(raw_result[0]),
            candidate_score_count=int(raw_result[1]),
            scores=scores,
        )

    def score_topk(
        self,
        *,
        queries: np.ndarray,
        candidate_positions_by_query: Sequence[Sequence[int]],
        reference_scores_by_query: Sequence[Sequence[float]],
        top_k: int,
    ) -> MojoGpuI8AddressTopKResult:
        if self._closed or self.handle == 0:
            raise RuntimeError("GPU i8 address session handle is closed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")

        marshalling_started_at = time.perf_counter()
        query_values = _float32_array(queries)
        expected_query_values = (
            int(self.shape.query_count)
            * int(self.shape.query_vector_count)
            * int(self.shape.vector_dim)
        )
        if query_values.size != expected_query_values:
            raise ValueError("queries shape must match the prepared handle shape")
        candidate_positions = _flatten_int_rows_array(
            candidate_positions_by_query,
            expected_rows=int(self.shape.query_count),
            expected_cols=int(self.candidate_k),
            name="candidate_positions_by_query",
        )
        reference_scores = _flatten_float_rows_array(
            reference_scores_by_query,
            expected_rows=int(self.shape.query_count),
            expected_cols=int(self.candidate_k),
            name="reference_scores_by_query",
        )
        host_marshalling_seconds = time.perf_counter() - marshalling_started_at

        module = load_module(target_accelerator=self.target_accelerator)
        request = [
            int(self.handle),
            _array_address(query_values),
            _array_address(candidate_positions),
            _array_address(reference_scores),
            int(top_k),
        ]
        extension_started_at = time.perf_counter()
        raw_result = module.score_i8_address_session_handle_topk(request)
        extension_call_seconds = time.perf_counter() - extension_started_at

        if len(raw_result) != 5:
            raise RuntimeError(
                "GPU i8 address session top-k returned an unexpected result shape"
            )

        positions = tuple(int(value) for value in raw_result[3])
        scores = tuple(float(value) for value in raw_result[4])
        reference_positions = _rank_candidate_positions_by_score(
            candidate_positions_by_query,
            reference_scores_by_query,
            final_k=int(raw_result[2]),
        )
        expected_positions = tuple(
            int(position) for row in reference_positions for position in row
        )
        topk_position_match_count = sum(
            1
            for actual, expected in zip(positions, expected_positions)
            if actual == expected
        )
        return MojoGpuI8AddressTopKResult(
            host_marshalling_seconds=host_marshalling_seconds,
            extension_call_seconds=extension_call_seconds,
            score_delta_max_abs=float(raw_result[0]),
            candidate_score_count=int(raw_result[1]),
            top_k=int(raw_result[2]),
            topk_position_match_count=topk_position_match_count,
            positions=positions,
            scores=scores,
        )

    def score_topk_without_reference(
        self,
        *,
        queries: np.ndarray,
        candidate_positions_by_query: Sequence[Sequence[int]],
        top_k: int,
    ) -> MojoGpuI8AddressTopKNoReferenceResult:
        if self._closed or self.handle == 0:
            raise RuntimeError("GPU i8 address session handle is closed")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.candidate_k:
            raise ValueError("top_k must not exceed candidate_k")

        marshalling_started_at = time.perf_counter()
        query_values = _float32_array(queries)
        expected_query_values = (
            int(self.shape.query_count)
            * int(self.shape.query_vector_count)
            * int(self.shape.vector_dim)
        )
        if query_values.size != expected_query_values:
            raise ValueError("queries shape must match the prepared handle shape")
        candidate_positions = _flatten_int_rows_array(
            candidate_positions_by_query,
            expected_rows=int(self.shape.query_count),
            expected_cols=int(self.candidate_k),
            name="candidate_positions_by_query",
        )
        host_marshalling_seconds = time.perf_counter() - marshalling_started_at

        module = load_module(target_accelerator=self.target_accelerator)
        request = [
            int(self.handle),
            _array_address(query_values),
            _array_address(candidate_positions),
            int(top_k),
        ]
        extension_started_at = time.perf_counter()
        raw_result = module.score_i8_address_session_handle_topk_no_reference(
            request
        )
        extension_call_seconds = time.perf_counter() - extension_started_at

        if len(raw_result) != 4:
            raise RuntimeError(
                "GPU i8 address session no-reference top-k returned an "
                "unexpected result shape"
            )

        positions = tuple(int(value) for value in raw_result[2])
        scores = tuple(float(value) for value in raw_result[3])
        return MojoGpuI8AddressTopKNoReferenceResult(
            host_marshalling_seconds=host_marshalling_seconds,
            extension_call_seconds=extension_call_seconds,
            candidate_score_count=int(raw_result[0]),
            top_k=int(raw_result[1]),
            positions=positions,
            scores=scores,
        )

    def close(self) -> float:
        if self._closed or self.handle == 0:
            return 0.0
        module = load_module(target_accelerator=self.target_accelerator)
        extension_started_at = time.perf_counter()
        module.release_i8_address_session_handle(int(self.handle))
        extension_call_seconds = time.perf_counter() - extension_started_at
        self.handle = 0
        self._closed = True
        return extension_call_seconds

    def __enter__(self) -> "MojoGpuI8AddressSessionHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def to_json_ready(self) -> dict[str, object]:
        return {
            "handle_open": not self._closed,
            "prepare_host_marshalling_seconds": (
                self.prepare_host_marshalling_seconds
            ),
            "prepare_extension_call_seconds": self.prepare_extension_call_seconds,
            "document_count": int(self.shape.document_count),
            "document_vector_count": int(self.shape.document_vector_count),
            "query_count": int(self.shape.query_count),
            "query_vector_count": int(self.shape.query_vector_count),
            "candidate_k": int(self.candidate_k),
            "vector_dim": int(self.shape.vector_dim),
        }


def gpu_extension_device_probe(*, target_accelerator: str) -> str:
    module = load_module(target_accelerator=target_accelerator)
    return str(module.gpu_extension_device_probe())


def prepare_i8_address_session_handle(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    payload: KayakPlaidI8PayloadSnapshot,
) -> MojoGpuI8AddressSessionHandle:
    marshalling_started_at = time.perf_counter()
    token_codes = _int8_array(payload.token_codes)
    token_scales = _float32_array(payload.token_scales)
    doc_offsets = _int64_array(payload.doc_offsets)
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        _array_address(token_codes),
        _array_address(token_scales),
        _array_address(doc_offsets),
        int(shape.document_count),
        int(shape.document_vector_count),
        int(shape.query_count),
        int(shape.query_vector_count),
        int(candidate_k),
    ]
    extension_started_at = time.perf_counter()
    raw_handle = module.prepare_i8_address_session_handle(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    handle = int(raw_handle)
    if handle == 0:
        raise RuntimeError("GPU i8 address session returned a null handle")
    return MojoGpuI8AddressSessionHandle(
        target_accelerator=target_accelerator,
        handle=handle,
        shape=shape,
        candidate_k=int(candidate_k),
        prepare_host_marshalling_seconds=host_marshalling_seconds,
        prepare_extension_call_seconds=extension_call_seconds,
    )


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


def profile_i8_candidate_generation_payload_addresses(
    *,
    target_accelerator: str,
    shape: Any,
    payload: KayakPlaidI8PayloadSnapshot,
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8CandidateGenerationPayloadResult:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    marshalling_started_at = time.perf_counter()
    centroid_token_indices = _int64_array(payload.centroid_token_indices)
    centroid_doc_offsets = _int64_array(payload.centroid_doc_offsets)
    centroid_doc_indices = _int64_array(payload.centroid_doc_indices)
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        _array_address(centroid_token_indices),
        _array_address(centroid_doc_offsets),
        _array_address(centroid_doc_indices),
        int(payload.centroid_count),
        int(payload.posting_count),
        int(shape.document_count),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_candidate_generation_payload_addresses(
        request
    )
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 9:
        raise RuntimeError(
            "GPU i8 candidate-generation payload bridge returned an "
            "unexpected result shape"
        )

    return MojoGpuI8CandidateGenerationPayloadResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        mojo_host_ingest_mean_seconds=float(raw_result[0]),
        host_to_device_mean_seconds=float(raw_result[1]),
        device_to_host_mean_seconds=float(raw_result[2]),
        copy_mismatch_count=int(raw_result[3]),
        offset_violation_count=int(raw_result[4]),
        doc_index_out_of_range_count=int(raw_result[5]),
        centroid_count=int(raw_result[6]),
        posting_count=int(raw_result[7]),
        document_count=int(raw_result[8]),
        byte_counts=payload.candidate_generation_byte_counts(),
    )


def profile_i8_selected_posting_traversal_addresses(
    *,
    target_accelerator: str,
    shape: Any,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8SelectedPostingTraversalResult:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    marshalling_started_at = time.perf_counter()
    centroid_doc_offsets = _int64_array(payload.centroid_doc_offsets)
    centroid_doc_indices = _int64_array(payload.centroid_doc_indices)
    reference = _selected_posting_traversal_reference(
        payload=payload,
        selected=selected,
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    if reference.expanded_posting_count <= 0:
        raise ValueError("selected posting traversal requires at least one visit")

    module = load_module(target_accelerator=target_accelerator)
    request = [
        _array_address(centroid_doc_offsets),
        _array_address(centroid_doc_indices),
        _array_address(reference.selected_positions),
        _array_address(reference.selected_scores),
        _array_address(reference.selected_posting_offsets),
        _array_address(reference.expected_doc_indices),
        _array_address(reference.expected_scores),
        int(payload.centroid_count),
        int(payload.posting_count),
        int(shape.document_count),
        int(reference.selected_centroid_count),
        int(reference.expanded_posting_count),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_selected_posting_traversal_addresses(
        request
    )
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 13:
        raise RuntimeError(
            "GPU i8 selected-posting traversal bridge returned an "
            "unexpected result shape"
        )

    return MojoGpuI8SelectedPostingTraversalResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        mojo_host_ingest_mean_seconds=float(raw_result[0]),
        payload_host_to_device_mean_seconds=float(raw_result[1]),
        selected_host_to_device_mean_seconds=float(raw_result[2]),
        kernel_mean_seconds=float(raw_result[3]),
        device_to_host_mean_seconds=float(raw_result[4]),
        selected_position_out_of_range_count=int(raw_result[5]),
        selected_offset_violation_count=int(raw_result[6]),
        doc_mismatch_count=int(raw_result[7]),
        doc_index_out_of_range_count=int(raw_result[8]),
        score_delta_max_abs=float(raw_result[9]),
        selected_centroid_count=int(raw_result[10]),
        expanded_posting_count=int(raw_result[11]),
        document_count=int(raw_result[12]),
    )


def profile_i8_selected_posting_accumulation_addresses(
    *,
    target_accelerator: str,
    shape: Any,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
    warmup_iterations: int,
    measurement_iterations: int,
) -> MojoGpuI8SelectedPostingAccumulationResult:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")

    marshalling_started_at = time.perf_counter()
    centroid_doc_offsets = _int64_array(payload.centroid_doc_offsets)
    centroid_doc_indices = _int64_array(payload.centroid_doc_indices)
    reference = _selected_posting_accumulation_reference(
        payload=payload,
        selected=selected,
    )
    host_marshalling_seconds = time.perf_counter() - marshalling_started_at

    module = load_module(target_accelerator=target_accelerator)
    request = [
        _array_address(centroid_doc_offsets),
        _array_address(centroid_doc_indices),
        _array_address(reference.selected_positions),
        _array_address(reference.selected_scores),
        _array_address(reference.expected_document_scores),
        int(payload.centroid_count),
        int(payload.posting_count),
        int(shape.document_count),
        int(shape.query_count),
        int(shape.query_vector_count),
        int(selected.centroids_per_query_vector),
        int(shape.top_k),
        int(warmup_iterations),
        int(measurement_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.profile_i8_selected_posting_accumulation_addresses(
        request
    )
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 16:
        raise RuntimeError(
            "GPU i8 selected-posting accumulation bridge returned an "
            "unexpected result shape"
        )

    return MojoGpuI8SelectedPostingAccumulationResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        mojo_host_ingest_mean_seconds=float(raw_result[0]),
        payload_host_to_device_mean_seconds=float(raw_result[1]),
        selected_host_to_device_mean_seconds=float(raw_result[2]),
        kernel_mean_seconds=float(raw_result[3]),
        device_to_host_mean_seconds=float(raw_result[4]),
        host_topk_mean_seconds=float(raw_result[5]),
        selected_position_out_of_range_count=int(raw_result[6]),
        doc_index_out_of_range_count=int(raw_result[7]),
        score_mismatch_count=int(raw_result[8]),
        score_delta_max_abs=float(raw_result[9]),
        topk_position_mismatch_count=int(raw_result[10]),
        top_k=int(raw_result[11]),
        topk_position_count=int(raw_result[12]),
        selected_centroid_count=int(raw_result[13]),
        document_score_count=int(raw_result[14]),
        document_count=int(raw_result[15]),
    )


def score_i8_prepared_payload_session_addresses(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
) -> MojoGpuI8AddressServeResult:
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
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.score_i8_prepared_payload_session_addresses(request)
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 3:
        raise RuntimeError(
            "GPU i8 address serving bridge returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[2])
    return MojoGpuI8AddressServeResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        score_delta_max_abs=float(raw_result[0]),
        candidate_score_count=int(raw_result[1]),
        scores=scores,
    )


def score_i8_prepared_payload_session_addresses_repeated(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    queries: np.ndarray,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    session_iterations: int,
) -> MojoGpuI8AddressResidentSessionResult:
    if session_iterations <= 0:
        raise ValueError("session_iterations must be positive")
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
        int(session_iterations),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.score_i8_prepared_payload_session_addresses_repeated(
        request
    )
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 4:
        raise RuntimeError(
            "GPU i8 address resident session returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[3])
    return MojoGpuI8AddressResidentSessionResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        score_delta_max_abs=float(raw_result[0]),
        candidate_score_count=int(raw_result[1]),
        session_iterations=int(raw_result[2]),
        scores=scores,
    )


def score_i8_prepared_payload_session_addresses_multi_window(
    *,
    target_accelerator: str,
    shape: Any,
    candidate_k: int,
    query_windows: Any,
    payload: KayakPlaidI8PayloadSnapshot,
    candidate_positions_by_window: Sequence[Sequence[Sequence[int]]],
    reference_scores_by_window: Sequence[Sequence[Sequence[float]]],
    window_count: int,
) -> MojoGpuI8AddressMultiWindowSessionResult:
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    marshalling_started_at = time.perf_counter()
    query_values = _float32_array(query_windows)
    expected_query_values = (
        window_count
        * shape.query_count
        * shape.query_vector_count
        * shape.vector_dim
    )
    if query_values.size != expected_query_values:
        raise ValueError("query_windows shape must match window_count and shape")
    token_codes = _int8_array(payload.token_codes)
    token_scales = _float32_array(payload.token_scales)
    doc_offsets = _int64_array(payload.doc_offsets)
    candidate_positions = _flatten_int_windows_array(
        candidate_positions_by_window,
        expected_windows=window_count,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="candidate_positions_by_window",
    )
    reference_scores = _flatten_float_windows_array(
        reference_scores_by_window,
        expected_windows=window_count,
        expected_rows=shape.query_count,
        expected_cols=candidate_k,
        name="reference_scores_by_window",
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
        int(window_count),
    ]
    extension_started_at = time.perf_counter()
    raw_result = module.score_i8_prepared_payload_session_addresses_multi_window(
        request
    )
    extension_call_seconds = time.perf_counter() - extension_started_at

    if len(raw_result) != 4:
        raise RuntimeError(
            "GPU i8 address multi-window session returned an unexpected result shape"
        )

    scores = tuple(float(value) for value in raw_result[3])
    return MojoGpuI8AddressMultiWindowSessionResult(
        host_marshalling_seconds=host_marshalling_seconds,
        extension_call_seconds=extension_call_seconds,
        score_delta_max_abs=float(raw_result[0]),
        candidate_score_count_per_window=int(raw_result[1]),
        window_count=int(raw_result[2]),
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


def _selected_posting_traversal_reference(
    *,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
) -> _SelectedPostingTraversalReference:
    payload.validate()
    selected.validate()
    selected_positions = _int64_array(selected.positions_array())
    selected_scores = _float32_array(selected.scores_array())
    if selected_positions.size != selected_scores.size:
        raise ValueError("selected centroid positions and scores must align")

    centroid_doc_offsets = _int64_array(payload.centroid_doc_offsets)
    centroid_doc_indices = _int64_array(payload.centroid_doc_indices)
    selected_posting_offsets = np.empty(selected_positions.size + 1, dtype=np.int64)
    selected_posting_offsets[0] = 0
    for selected_index, centroid_position in enumerate(selected_positions):
        centroid = int(centroid_position)
        if centroid < 0 or centroid >= payload.centroid_count:
            raise ValueError("selected centroid position out of range")
        posting_start = int(centroid_doc_offsets[centroid])
        posting_stop = int(centroid_doc_offsets[centroid + 1])
        selected_posting_offsets[selected_index + 1] = (
            selected_posting_offsets[selected_index] + posting_stop - posting_start
        )

    expanded_posting_count = int(selected_posting_offsets[-1])
    expected_doc_indices = np.empty(expanded_posting_count, dtype=np.int64)
    expected_scores = np.empty(expanded_posting_count, dtype=np.float32)
    for selected_index, centroid_position in enumerate(selected_positions):
        centroid = int(centroid_position)
        posting_start = int(centroid_doc_offsets[centroid])
        posting_stop = int(centroid_doc_offsets[centroid + 1])
        output_start = int(selected_posting_offsets[selected_index])
        output_stop = int(selected_posting_offsets[selected_index + 1])
        expected_doc_indices[output_start:output_stop] = centroid_doc_indices[
            posting_start:posting_stop
        ]
        expected_scores[output_start:output_stop] = selected_scores[selected_index]

    return _SelectedPostingTraversalReference(
        selected_positions=selected_positions,
        selected_scores=selected_scores,
        selected_posting_offsets=np.ascontiguousarray(
            selected_posting_offsets,
            dtype=np.int64,
        ),
        expected_doc_indices=np.ascontiguousarray(
            expected_doc_indices,
            dtype=np.int64,
        ),
        expected_scores=np.ascontiguousarray(expected_scores, dtype=np.float32),
    )


def _selected_posting_accumulation_reference(
    *,
    payload: KayakPlaidI8PayloadSnapshot,
    selected: KayakPlaidI8SelectedCentroids,
) -> _SelectedPostingAccumulationReference:
    payload.validate()
    selected.validate()
    selected_positions = _int64_array(selected.positions_array())
    selected_scores = _float32_array(selected.scores_array())
    if selected_positions.size != selected_scores.size:
        raise ValueError("selected centroid positions and scores must align")

    centroid_doc_offsets = _int64_array(payload.centroid_doc_offsets)
    centroid_doc_indices = _int64_array(payload.centroid_doc_indices)
    document_scores = np.zeros(
        (selected.query_count, payload.document_count),
        dtype=np.float32,
    )
    selected_per_query = selected.selected_centroid_count_per_query
    for query_index in range(selected.query_count):
        query_base = query_index * selected_per_query
        for query_vector_index in range(selected.query_vector_count):
            selected_base = (
                query_base
                + query_vector_index * selected.centroids_per_query_vector
            )
            best_scores = np.empty(payload.document_count, dtype=np.float32)
            best_scores.fill(np.float32(-np.inf))
            seen = np.zeros(payload.document_count, dtype=bool)
            for selected_offset in range(selected.centroids_per_query_vector):
                selected_index = selected_base + selected_offset
                centroid = int(selected_positions[selected_index])
                if centroid < 0 or centroid >= payload.centroid_count:
                    raise ValueError("selected centroid position out of range")
                posting_start = int(centroid_doc_offsets[centroid])
                posting_stop = int(centroid_doc_offsets[centroid + 1])
                docs = centroid_doc_indices[posting_start:posting_stop]
                score = selected_scores[selected_index]
                best_scores[docs] = np.maximum(best_scores[docs], score)
                seen[docs] = True
            document_scores[query_index, seen] += best_scores[seen]

    return _SelectedPostingAccumulationReference(
        selected_positions=selected_positions,
        selected_scores=selected_scores,
        expected_document_scores=np.ascontiguousarray(
            document_scores.reshape(-1),
            dtype=np.float32,
        ),
    )


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


def _flatten_int_windows_array(
    windows: Sequence[Sequence[Sequence[int]]],
    *,
    expected_windows: int,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    values = np.ascontiguousarray(windows, dtype=np.int64).reshape(-1)
    expected_size = expected_windows * expected_rows * expected_cols
    if values.size != expected_size:
        raise ValueError(
            f"{name} shape must match window_count, query_count, and candidate_k"
        )
    return values


def _flatten_float_windows_array(
    windows: Sequence[Sequence[Sequence[float]]],
    *,
    expected_windows: int,
    expected_rows: int,
    expected_cols: int,
    name: str,
) -> np.ndarray:
    values = np.ascontiguousarray(windows, dtype=np.float32).reshape(-1)
    expected_size = expected_windows * expected_rows * expected_cols
    if values.size != expected_size:
        raise ValueError(
            f"{name} shape must match window_count, query_count, and candidate_k"
        )
    return values


def _rank_candidate_positions_by_score(
    candidate_positions_by_query: Sequence[Sequence[int]],
    scores_by_query: Sequence[Sequence[float]],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    if len(candidate_positions_by_query) != len(scores_by_query):
        raise ValueError("candidate and score query counts must match")
    ranked_rows: list[tuple[int, ...]] = []
    for candidate_positions, scores in zip(
        candidate_positions_by_query,
        scores_by_query,
    ):
        if len(candidate_positions) != len(scores):
            raise ValueError("candidate and score row lengths must match")
        ranked_offsets = sorted(
            range(len(candidate_positions)),
            key=lambda offset: float(scores[offset]),
            reverse=True,
        )
        ranked_rows.append(
            tuple(
                int(candidate_positions[offset])
                for offset in ranked_offsets[:final_k]
            )
        )
    return tuple(ranked_rows)
