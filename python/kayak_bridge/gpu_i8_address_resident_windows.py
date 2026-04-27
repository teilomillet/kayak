"""Resident-window helpers for the GPU i8 address serving sweep.

This module owns deterministic query-window construction and resident GPU probe
dispatch. It does not build indexes or assemble the final sweep report.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.gpu_i8_address_serve_sweep import (
    STATUS_OK,
    AddressServeSweepCase,
    AddressServeSweepControls,
)
from kayak_bridge.gpu_i8_score_agreement import gpu_i8_score_agreement_fields
from kayak_bridge.mojo_gpu_i8_rerank import (
    prepare_i8_address_session_handle,
    score_i8_prepared_payload_session_addresses_multi_window,
    score_i8_prepared_payload_session_addresses_repeated,
)

from bench_fastplaid_speed_track import SpeedTrackShape


def build_query_windows(
    *,
    base_queries: np.ndarray,
    shape: SpeedTrackShape,
    window_count: int,
    seed: int,
) -> np.ndarray:
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    query_windows = np.empty(
        (
            window_count,
            shape.query_count,
            shape.query_vector_count,
            shape.vector_dim,
        ),
        dtype=np.float32,
    )
    query_windows[0] = np.ascontiguousarray(base_queries, dtype=np.float32)
    if window_count == 1:
        return query_windows
    rng = np.random.default_rng(seed)
    query_windows[1:] = rng.standard_normal(
        (
            window_count - 1,
            shape.query_count,
            shape.query_vector_count,
            shape.vector_dim,
        )
    ).astype(np.float32)
    return query_windows


def reshape_windows(
    rows: Sequence[Sequence[Any]],
    *,
    window_count: int,
    query_count: int,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    expected_rows = window_count * query_count
    if len(rows) != expected_rows:
        raise ValueError("row count must match window_count * query_count")
    return tuple(
        tuple(tuple(row) for row in rows[offset : offset + query_count])
        for offset in range(0, expected_rows, query_count)
    )


def timing_payload_per_window(
    timing: Any,
    *,
    window_count: int,
) -> dict[str, object]:
    payload = timing.to_json_ready()
    return payload | {
        "mean_seconds_per_window": timing.mean_seconds / float(window_count),
        "window_count": window_count,
    }


def run_repeated_resident_probe(
    *,
    case: AddressServeSweepCase,
    shape: SpeedTrackShape,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
    queries: Any,
    payload: Any,
    candidate_positions: Sequence[Sequence[int]],
    reference_scores: Sequence[Sequence[float]],
) -> dict[str, object] | None:
    if not capability.available:
        return None
    try:
        result = score_i8_prepared_payload_session_addresses_repeated(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=case.candidate_k,
            queries=queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions,
            reference_scores_by_query=reference_scores,
            session_iterations=controls.resident_session_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {"status": "error", "parsed": {}, "error": str(exc)}

    parsed = {
        "bridge_scope": "single_extension_call_address_resident_session",
        "candidate_k": case.candidate_k,
        "candidate_score_count": result.candidate_score_count,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "extension_call_seconds": result.extension_call_seconds,
        "extension_call_seconds_per_iteration": (
            result.extension_call_seconds_per_iteration
        ),
        "host_marshalling_seconds": result.host_marshalling_seconds,
        "payload_source": "real_kayak_i8_snapshot",
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "session_iterations": result.session_iterations,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
    } | gpu_i8_score_agreement_fields(
        score_delta_max_abs=result.score_delta_max_abs,
        query_vector_count=shape.query_vector_count,
    )
    return {
        "status": STATUS_OK if parsed["score_agreement_ok"] else "error",
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def run_multi_window_resident_probe(
    *,
    case: AddressServeSweepCase,
    shape: SpeedTrackShape,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
    query_windows: np.ndarray,
    payload: Any,
    candidate_positions: Sequence[Sequence[Sequence[int]]],
    reference_scores: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object] | None:
    if not capability.available:
        return None
    try:
        result = score_i8_prepared_payload_session_addresses_multi_window(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=case.candidate_k,
            query_windows=query_windows,
            payload=payload,
            candidate_positions_by_window=candidate_positions,
            reference_scores_by_window=reference_scores,
            window_count=controls.resident_session_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {"status": "error", "parsed": {}, "error": str(exc)}

    parsed = {
        "bridge_scope": "single_extension_call_address_resident_multi_window",
        "candidate_k": case.candidate_k,
        "candidate_score_count_per_window": (
            result.candidate_score_count_per_window
        ),
        "candidate_score_count_total": result.candidate_score_count_total,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "extension_call_seconds": result.extension_call_seconds,
        "extension_call_seconds_per_window": (
            result.extension_call_seconds_per_window
        ),
        "host_marshalling_seconds": result.host_marshalling_seconds,
        "payload_source": "real_kayak_i8_snapshot",
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
        "window_count": result.window_count,
    } | gpu_i8_score_agreement_fields(
        score_delta_max_abs=result.score_delta_max_abs,
        query_vector_count=shape.query_vector_count,
    )
    return {
        "status": STATUS_OK if parsed["score_agreement_ok"] else "error",
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def run_cross_call_prepared_handle_probe(
    *,
    case: AddressServeSweepCase,
    shape: SpeedTrackShape,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
    query_windows: np.ndarray,
    payload: Any,
    candidate_positions: Sequence[Sequence[Sequence[int]]],
    reference_scores: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object] | None:
    if not capability.available:
        return None
    handle = None
    release_seconds = 0.0
    try:
        handle = prepare_i8_address_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=case.candidate_k,
            payload=payload,
        )
        for _ in range(controls.warmup_iterations):
            handle.score(
                queries=query_windows[0],
                candidate_positions_by_query=candidate_positions[0],
                reference_scores_by_query=reference_scores[0],
            )
        score_results = []
        for window_index in range(controls.resident_session_iterations):
            score_results.append(
                handle.score(
                    queries=query_windows[window_index],
                    candidate_positions_by_query=candidate_positions[window_index],
                    reference_scores_by_query=reference_scores[window_index],
                )
            )
        release_seconds = handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if handle is not None:
            try:
                release_seconds = handle.close()
            except Exception:
                pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    score_extension_total = sum(
        result.extension_call_seconds for result in score_results
    )
    score_host_marshalling_total = sum(
        result.host_marshalling_seconds for result in score_results
    )
    score_delta_max_abs = max(
        (result.score_delta_max_abs for result in score_results),
        default=0.0,
    )
    score_count_total = sum(result.candidate_score_count for result in score_results)
    parsed = {
        "bridge_scope": "explicit_handle_address_resident_multi_window",
        "candidate_k": case.candidate_k,
        "candidate_score_count_per_window": (
            shape.query_count * case.candidate_k
        ),
        "candidate_score_count_total": score_count_total,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "extension_call_seconds": score_extension_total,
        "extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "host_marshalling_seconds": score_host_marshalling_total,
        "host_marshalling_seconds_per_window": (
            score_host_marshalling_total
            / float(controls.resident_session_iterations)
        ),
        "payload_source": "real_kayak_i8_snapshot",
        "prepare_extension_call_seconds": handle.prepare_extension_call_seconds,
        "prepare_host_marshalling_seconds": (
            handle.prepare_host_marshalling_seconds
        ),
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "score_extension_call_seconds_total": score_extension_total,
        "score_extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
        "warmup_iterations": controls.warmup_iterations,
        "window_count": controls.resident_session_iterations,
    } | gpu_i8_score_agreement_fields(
        score_delta_max_abs=score_delta_max_abs,
        query_vector_count=shape.query_vector_count,
    )
    return {
        "status": STATUS_OK if parsed["score_agreement_ok"] else "error",
        "parsed": parsed,
        "measurements": [
            handle.to_json_ready(),
            *(result.to_json_ready() for result in score_results),
            {"release_extension_call_seconds": release_seconds},
        ],
    }


def run_cross_call_prepared_handle_topk_probe(
    *,
    case: AddressServeSweepCase,
    shape: SpeedTrackShape,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
    query_windows: np.ndarray,
    payload: Any,
    candidate_positions: Sequence[Sequence[Sequence[int]]],
    reference_scores: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object] | None:
    if not capability.available:
        return None
    handle = None
    release_seconds = 0.0
    try:
        handle = prepare_i8_address_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=case.candidate_k,
            payload=payload,
        )
        for _ in range(controls.warmup_iterations):
            handle.score_topk(
                queries=query_windows[0],
                candidate_positions_by_query=candidate_positions[0],
                reference_scores_by_query=reference_scores[0],
                top_k=shape.top_k,
            )
        topk_results = []
        for window_index in range(controls.resident_session_iterations):
            topk_results.append(
                handle.score_topk(
                    queries=query_windows[window_index],
                    candidate_positions_by_query=candidate_positions[window_index],
                    reference_scores_by_query=reference_scores[window_index],
                    top_k=shape.top_k,
                )
            )
        release_seconds = handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if handle is not None:
            try:
                release_seconds = handle.close()
            except Exception:
                pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    score_extension_total = sum(
        result.extension_call_seconds for result in topk_results
    )
    score_host_marshalling_total = sum(
        result.host_marshalling_seconds for result in topk_results
    )
    score_delta_max_abs = max(
        (result.score_delta_max_abs for result in topk_results),
        default=0.0,
    )
    topk_position_match_count = sum(
        result.topk_position_match_count for result in topk_results
    )
    topk_position_count = sum(result.topk_position_count for result in topk_results)
    candidate_score_count_total = sum(
        result.candidate_score_count for result in topk_results
    )
    parsed = {
        "bridge_scope": "explicit_handle_address_resident_topk",
        "candidate_k": case.candidate_k,
        "candidate_score_count_per_window": (
            shape.query_count * case.candidate_k
        ),
        "candidate_score_count_total": candidate_score_count_total,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "extension_call_seconds": score_extension_total,
        "extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "host_marshalling_seconds": score_host_marshalling_total,
        "host_marshalling_seconds_per_window": (
            score_host_marshalling_total
            / float(controls.resident_session_iterations)
        ),
        "payload_source": "real_kayak_i8_snapshot",
        "prepare_extension_call_seconds": handle.prepare_extension_call_seconds,
        "prepare_host_marshalling_seconds": (
            handle.prepare_host_marshalling_seconds
        ),
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "score_extension_call_seconds_total": score_extension_total,
        "score_extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "top_k": shape.top_k,
        "topk_position_agreement": (
            topk_position_match_count / float(topk_position_count)
            if topk_position_count
            else 0.0
        ),
        "topk_position_count": topk_position_count,
        "topk_position_match_count": topk_position_match_count,
        "topk_return_count_per_window": shape.query_count * shape.top_k,
        "topk_return_count_total": topk_position_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
        "warmup_iterations": controls.warmup_iterations,
        "window_count": controls.resident_session_iterations,
    } | gpu_i8_score_agreement_fields(
        score_delta_max_abs=score_delta_max_abs,
        query_vector_count=shape.query_vector_count,
    )
    status = (
        STATUS_OK
        if parsed["score_agreement_ok"]
        and parsed["topk_position_agreement"] == 1.0
        else "error"
    )
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [
            handle.to_json_ready(),
            *(result.to_json_ready() for result in topk_results),
            {"release_extension_call_seconds": release_seconds},
        ],
    }


def run_cross_call_prepared_handle_topk_no_reference_probe(
    *,
    case: AddressServeSweepCase,
    shape: SpeedTrackShape,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
    query_windows: np.ndarray,
    payload: Any,
    candidate_positions: Sequence[Sequence[Sequence[int]]],
    reference_scores: Sequence[Sequence[Sequence[float]]],
) -> dict[str, object] | None:
    if not capability.available:
        return None
    handle = None
    release_seconds = 0.0
    try:
        handle = prepare_i8_address_session_handle(
            target_accelerator=capability.target_accelerator or "",
            shape=shape,
            candidate_k=case.candidate_k,
            payload=payload,
        )
        for _ in range(controls.warmup_iterations):
            handle.score_topk_without_reference(
                queries=query_windows[0],
                candidate_positions_by_query=candidate_positions[0],
                top_k=shape.top_k,
            )
        topk_results = []
        for window_index in range(controls.resident_session_iterations):
            topk_results.append(
                handle.score_topk_without_reference(
                    queries=query_windows[window_index],
                    candidate_positions_by_query=candidate_positions[window_index],
                    top_k=shape.top_k,
                )
            )
        release_seconds = handle.close()
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        if handle is not None:
            try:
                release_seconds = handle.close()
            except Exception:
                pass
        return {"status": "error", "parsed": {}, "error": str(exc)}

    score_extension_total = sum(
        result.extension_call_seconds for result in topk_results
    )
    score_host_marshalling_total = sum(
        result.host_marshalling_seconds for result in topk_results
    )
    topk_position_match_count = 0
    topk_position_count = 0
    for window_index, result in enumerate(topk_results):
        expected_positions = _expected_topk_positions(
            candidate_positions[window_index],
            reference_scores[window_index],
            top_k=shape.top_k,
        )
        topk_position_count += len(expected_positions)
        topk_position_match_count += sum(
            1
            for actual, expected in zip(result.positions, expected_positions)
            if actual == expected
        )

    candidate_score_count_total = sum(
        result.candidate_score_count for result in topk_results
    )
    parsed = {
        "bridge_scope": "explicit_handle_address_resident_topk_no_reference",
        "candidate_k": case.candidate_k,
        "candidate_score_count_per_window": (
            shape.query_count * case.candidate_k
        ),
        "candidate_score_count_total": candidate_score_count_total,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "extension_call_seconds": score_extension_total,
        "extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "host_marshalling_seconds": score_host_marshalling_total,
        "host_marshalling_seconds_per_window": (
            score_host_marshalling_total
            / float(controls.resident_session_iterations)
        ),
        "payload_source": "real_kayak_i8_snapshot",
        "prepare_extension_call_seconds": handle.prepare_extension_call_seconds,
        "prepare_host_marshalling_seconds": (
            handle.prepare_host_marshalling_seconds
        ),
        "query_count_per_window": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "release_extension_call_seconds": release_seconds,
        "score_extension_call_seconds_total": score_extension_total,
        "score_extension_call_seconds_per_window": (
            score_extension_total / float(controls.resident_session_iterations)
        ),
        "top_k": shape.top_k,
        "topk_position_agreement": (
            topk_position_match_count / float(topk_position_count)
            if topk_position_count
            else 0.0
        ),
        "topk_position_count": topk_position_count,
        "topk_position_match_count": topk_position_match_count,
        "topk_return_count_per_window": shape.query_count * shape.top_k,
        "topk_return_count_total": topk_position_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "validation_reference_scores_sent_to_extension": False,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": controls.warmup_iterations,
        "window_count": controls.resident_session_iterations,
    }
    status = (
        STATUS_OK if parsed["topk_position_agreement"] == 1.0 else "error"
    )
    return {
        "status": status,
        "parsed": parsed,
        "measurements": [
            handle.to_json_ready(),
            *(result.to_json_ready() for result in topk_results),
            {"release_extension_call_seconds": release_seconds},
        ],
    }


def _expected_topk_positions(
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    *,
    top_k: int,
) -> tuple[int, ...]:
    expected: list[int] = []
    for positions, scores in zip(
        candidate_positions_by_query, reference_scores_by_query
    ):
        # The Mojo extension scans candidate offsets and only replaces the
        # winner on a strict score increase, so equal scores keep input order.
        ranked_offsets = sorted(
            range(len(positions)),
            key=lambda offset: (-scores[offset], offset),
        )
        expected.extend(int(positions[offset]) for offset in ranked_offsets[:top_k])
    return tuple(expected)
