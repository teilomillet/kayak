"""Runs GPU i8 typed-address serving sweep cases.

This module owns index construction, CPU timing, GPU probe dispatch, and report
assembly for the address serving sweep. It does not own CLI parsing.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability, probe_mojo_gpu
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
from kayak_bridge.mojo_gpu_i8_rerank import (
    score_i8_prepared_payload_session_addresses_repeated,
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
                "copies them once inside one extension call and repeats the "
                "same candidate window."
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
    resident_probe = _run_resident_probe(
        case=case,
        shape=shape,
        controls=controls,
        capability=capability,
        queries=inputs.queries,
        payload=payload,
        candidate_positions=candidate_positions,
        reference_scores=reference_scores,
    )
    gpu_status = _gpu_status(gpu_probe)
    resident_status = _gpu_status(resident_probe)
    gpu_parsed = parsed_payload(gpu_probe)
    resident_parsed = parsed_payload(resident_probe)
    return _case_row(
        case=case,
        controls=controls,
        shape=shape,
        seed=controls.seed + case_index,
        build_seconds=build_seconds,
        candidate_timing=candidate_timing,
        score_timing=score_timing,
        capability=capability,
        gpu_probe=gpu_probe,
        gpu_status=gpu_status,
        gpu_parsed=gpu_parsed,
        resident_probe=resident_probe,
        resident_status=resident_status,
        resident_parsed=resident_parsed,
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


def _run_resident_probe(
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
        "score_agreement_ok": result.score_delta_max_abs <= 0.0001,
        "score_delta_max_abs": result.score_delta_max_abs,
        "session_iterations": result.session_iterations,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "vector_dim": shape.vector_dim,
    }
    return {
        "status": STATUS_OK if parsed["score_agreement_ok"] else "error",
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


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
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
    gpu_status: object,
    gpu_parsed: dict[str, object],
    resident_probe: dict[str, object] | None,
    resident_status: object,
    resident_parsed: dict[str, object],
) -> dict[str, Any]:
    status = (
        STATUS_OK
        if gpu_status == STATUS_OK and resident_status == STATUS_OK
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
        "comparison": comparison_payload(
            cpu_candidate_generation_mean_seconds=candidate_timing.mean_seconds,
            cpu_score_mean_seconds=score_timing.mean_seconds,
            gpu_parsed=gpu_parsed,
            resident_parsed=resident_parsed,
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
            "host_marshalling_seconds": optional_float(
                gpu_parsed.get("host_marshalling_seconds")
            ),
        },
        "error": gpu_probe.get("error") if isinstance(gpu_probe, dict) else None,
    }
