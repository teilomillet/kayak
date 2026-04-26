"""Runs GPU i8 typed-address serving sweep cases.

This module owns index construction, CPU timing, GPU probe dispatch, and report
assembly for the address serving sweep. It does not own CLI parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability, probe_mojo_gpu
from kayak_bridge.gpu_i8_address_resident_windows import (
    build_query_windows,
    reshape_windows,
    run_multi_window_resident_probe,
    run_repeated_resident_probe,
    timing_payload_per_window,
)
from kayak_bridge.gpu_i8_address_serve_sweep import (
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
    AddressServeSweepCase,
    AddressServeSweepControls,
    comparison_payload,
    optional_float,
    report_status,
    summary_payload,
)
from kayak_bridge.plaid_approx import (
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
)

from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs
from profile_gpu_i8_real_payload_rerank import (
    parsed_payload,
    run_gpu_i8_address_serve_probe,
    time_candidate_generation,
    time_same_candidate_scores,
)


def build_report(
    *,
    cases: Sequence[AddressServeSweepCase],
    controls: AddressServeSweepControls,
) -> tuple[dict[str, Any], MojoGpuCapability]:
    controls.validate()
    capability = probe_mojo_gpu(controls.gpu_query_command)
    rows = [
        run_case(
            case,
            case_index=case_index,
            controls=controls,
            capability=capability,
        )
        for case_index, case in enumerate(cases)
    ]
    status = report_status(capability=capability, rows=rows)
    return (
        {
            "schema_version": 1,
            "benchmark": "gpu_i8_address_serve_shape_sweep",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "controls": controls.to_json_ready(),
            "mojo_gpu_capability": capability.to_json_ready(),
            "cases": rows,
            "summary": summary_payload(rows),
            "measurement_note": (
                "This sweep times the internal typed-address GPU i8 serving "
                "call and an in-call resident-session variant. Candidate "
                "generation remains CPU-side and top-k remains outside these "
                "GPU rows. The serving row still allocates and copies prepared "
                "index tensors inside each call; the resident-session row "
                "copies them once inside one extension call. The repeated row "
                "scores the same candidate window, while the multi-window row "
                "scores different query and candidate windows."
            ),
        },
        capability,
    )


def run_case(
    case: AddressServeSweepCase,
    *,
    case_index: int,
    controls: AddressServeSweepControls,
    capability: MojoGpuCapability,
) -> dict[str, Any]:
    shape = case.shape(vector_dim=controls.vector_dim, top_k=controls.top_k)
    inputs = build_synthetic_inputs(
        shape,
        seed=controls.seed + case_index,
        normalize_vectors=False,
    )
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=controls.kayak_plaid_centroid_count,
            centroids_per_query_vector=(
                controls.kayak_plaid_centroids_per_query_vector
            ),
            candidate_k=case.candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at
    candidate_positions = index.i8_candidate_positions_batch(inputs.queries)
    reference_scores = index.i8_score_candidate_positions_batch(
        inputs.queries,
        candidate_positions,
    )
    candidate_timing = time_candidate_generation(
        index,
        inputs.queries,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    score_timing = time_same_candidate_scores(
        index,
        inputs.queries,
        candidate_positions,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    payload = index.i8_payload_snapshot()
    query_windows = build_query_windows(
        base_queries=inputs.queries,
        shape=shape,
        window_count=controls.resident_session_iterations,
        seed=controls.seed + case_index + 100_000,
    )
    multi_queries = query_windows.reshape(
        controls.resident_session_iterations * shape.query_count,
        shape.query_vector_count,
        shape.vector_dim,
    )
    multi_candidate_positions = index.i8_candidate_positions_batch(multi_queries)
    multi_reference_scores = index.i8_score_candidate_positions_batch(
        multi_queries,
        multi_candidate_positions,
    )
    multi_candidate_timing = time_candidate_generation(
        index,
        multi_queries,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    multi_score_timing = time_same_candidate_scores(
        index,
        multi_queries,
        multi_candidate_positions,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    gpu_probe = _run_gpu_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        queries=inputs.queries,
        payload=payload,
        candidate_positions=candidate_positions,
        reference_scores=reference_scores,
    )
    resident_probe = run_repeated_resident_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        queries=inputs.queries,
        payload=payload,
        candidate_positions=candidate_positions,
        reference_scores=reference_scores,
    )
    multi_window_probe = run_multi_window_resident_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        query_windows=query_windows,
        payload=payload,
        candidate_positions=reshape_windows(
            multi_candidate_positions,
            window_count=controls.resident_session_iterations,
            query_count=shape.query_count,
        ),
        reference_scores=reshape_windows(
            multi_reference_scores,
            window_count=controls.resident_session_iterations,
            query_count=shape.query_count,
        ),
    )
    gpu_status = _gpu_status(gpu_probe)
    resident_status = _gpu_status(resident_probe)
    multi_window_status = _gpu_status(multi_window_probe)
    gpu_parsed = parsed_payload(gpu_probe)
    resident_parsed = parsed_payload(resident_probe)
    multi_window_parsed = parsed_payload(multi_window_probe)
    return _case_row(
        case=case,
        controls=controls,
        shape=shape,
        seed=controls.seed + case_index,
        build_seconds=build_seconds,
        candidate_timing=candidate_timing,
        score_timing=score_timing,
        multi_candidate_timing=multi_candidate_timing,
        multi_score_timing=multi_score_timing,
        capability=capability,
        gpu_probe=gpu_probe,
        gpu_status=gpu_status,
        gpu_parsed=gpu_parsed,
        resident_probe=resident_probe,
        resident_status=resident_status,
        resident_parsed=resident_parsed,
        multi_window_probe=multi_window_probe,
        multi_window_status=multi_window_status,
        multi_window_parsed=multi_window_parsed,
    )


def _run_gpu_probe(
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
    return run_gpu_i8_address_serve_probe(
        shape=shape,
        candidate_k=case.candidate_k,
        target_accelerator=capability.target_accelerator,
        queries=queries,
        payload=payload,
        candidate_positions_by_query=candidate_positions,
        reference_scores_by_query=reference_scores,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )


def _gpu_status(gpu_probe: dict[str, object] | None) -> object:
    if isinstance(gpu_probe, dict):
        return gpu_probe.get("status")
    return STATUS_PARTIAL_GPU_UNAVAILABLE


def _case_row(
    *,
    case: AddressServeSweepCase,
    controls: AddressServeSweepControls,
    shape: SpeedTrackShape,
    seed: int,
    build_seconds: float,
    candidate_timing: Any,
    score_timing: Any,
    multi_candidate_timing: Any,
    multi_score_timing: Any,
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
    gpu_status: object,
    gpu_parsed: dict[str, object],
    resident_probe: dict[str, object] | None,
    resident_status: object,
    resident_parsed: dict[str, object],
    multi_window_probe: dict[str, object] | None,
    multi_window_status: object,
    multi_window_parsed: dict[str, object],
) -> dict[str, Any]:
    status = (
        STATUS_OK
        if (
            gpu_status == STATUS_OK
            and resident_status == STATUS_OK
            and multi_window_status == STATUS_OK
        )
        else "error"
    )
    return {
        "name": case.name,
        "status": status,
        "shape": case.to_json_ready(
            vector_dim=controls.vector_dim,
            top_k=controls.top_k,
        ),
        "seed": seed,
        "cpu_i8_build_seconds": build_seconds,
        "cpu_i8_candidate_generation": candidate_timing.to_json_ready(),
        "cpu_i8_same_candidate_reference": score_timing.to_json_ready()
        | {"candidate_score_count_total": shape.query_count * case.candidate_k},
        "cpu_i8_multi_window_candidate_generation": (
            timing_payload_per_window(
                multi_candidate_timing,
                window_count=controls.resident_session_iterations,
            )
        ),
        "cpu_i8_multi_window_same_candidate_reference": (
            timing_payload_per_window(
                multi_score_timing,
                window_count=controls.resident_session_iterations,
            )
            | {
                "candidate_score_count_total": (
                    controls.resident_session_iterations
                    * shape.query_count
                    * case.candidate_k
                ),
                "candidate_score_count_per_window": (
                    shape.query_count * case.candidate_k
                ),
            }
        ),
        "gpu_address_serve": _gpu_payload(
            capability=capability,
            gpu_probe=gpu_probe,
            gpu_status=gpu_status,
            gpu_parsed=gpu_parsed,
        ),
        "gpu_address_resident_session": _gpu_payload(
            capability=capability,
            gpu_probe=resident_probe,
            gpu_status=resident_status,
            gpu_parsed=resident_parsed,
        ),
        "gpu_address_resident_multi_window_session": _gpu_payload(
            capability=capability,
            gpu_probe=multi_window_probe,
            gpu_status=multi_window_status,
            gpu_parsed=multi_window_parsed,
        ),
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=candidate_timing.mean_seconds,
            cpu_score_mean_seconds=score_timing.mean_seconds,
            gpu_parsed=gpu_parsed,
            resident_parsed=resident_parsed,
            cpu_multi_window_candidate_generation_mean_seconds_per_window=(
                multi_candidate_timing.mean_seconds
                / float(controls.resident_session_iterations)
            ),
            cpu_multi_window_score_mean_seconds_per_window=(
                multi_score_timing.mean_seconds
                / float(controls.resident_session_iterations)
            ),
            multi_window_parsed=multi_window_parsed,
        ),
    }


def _gpu_payload(
    *,
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
    gpu_status: object,
    gpu_parsed: dict[str, object],
) -> dict[str, object]:
    return {
        "status": gpu_status,
        "target_accelerator": capability.target_accelerator,
        "parsed": gpu_parsed,
        "derived": {
            "extension_call_seconds": optional_float(
                gpu_parsed.get("extension_call_seconds")
            ),
            "extension_call_seconds_per_iteration": optional_float(
                gpu_parsed.get("extension_call_seconds_per_iteration")
            ),
            "extension_call_seconds_per_window": optional_float(
                gpu_parsed.get("extension_call_seconds_per_window")
            ),
            "host_marshalling_seconds": optional_float(
                gpu_parsed.get("host_marshalling_seconds")
            ),
        },
        "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
    }
