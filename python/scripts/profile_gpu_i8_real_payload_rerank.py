from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.gpu_device_capability import (  # noqa: E402
    MojoGpuCapability,
    probe_mojo_gpu,
)
from kayak_bridge.gpu_i8_real_payload_score import (  # noqa: E402
    GPU_REAL_PAYLOAD_STATUS_OK,
    run_gpu_i8_real_payload_score_probe,
    write_gpu_i8_real_payload_probe_inputs,
)
from kayak_bridge.gpu_i8_rerank_contract import write_report  # noqa: E402
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    profile_i8_prepared_payload_session,
    profile_i8_prepared_payload_session_addresses,
    profile_i8_prepared_payload_session_ndarray,
    score_i8_real_payload_once,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import (  # noqa: E402
    SpeedTrackShape,
    benchmark_kayak_exact,
    build_synthetic_inputs,
    mean_recall_at_k,
)
from profile_gpu_i8_candidate_score import derive_candidate_score_metrics  # noqa: E402


STATUS_OK = "ok"
STATUS_PARTIAL_GPU_UNAVAILABLE = "partial_gpu_unavailable"
STATUS_BLOCKED_GPU_REAL_PAYLOAD_FAILED = "blocked_gpu_real_payload_failed"
STATUS_BLOCKED_GPU_BRIDGE_FAILED = "blocked_gpu_bridge_failed"
STATUS_BLOCKED_GPU_PREPARED_BRIDGE_FAILED = (
    "blocked_gpu_prepared_bridge_failed"
)
STATUS_BLOCKED_GPU_PREPARED_NDARRAY_BRIDGE_FAILED = (
    "blocked_gpu_prepared_ndarray_bridge_failed"
)
STATUS_BLOCKED_GPU_PREPARED_ADDRESS_BRIDGE_FAILED = (
    "blocked_gpu_prepared_address_bridge_failed"
)


@dataclass(frozen=True, slots=True)
class TimingSummary:
    min_seconds: float
    median_seconds: float
    mean_seconds: float
    max_seconds: float

    def to_json_ready(self) -> dict[str, float]:
        return {
            "min_seconds": self.min_seconds,
            "median_seconds": self.median_seconds,
            "mean_seconds": self.mean_seconds,
            "max_seconds": self.max_seconds,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile the GPU i8 candidate-score primitive on real Kayak i8 "
            "payload snapshots and CPU-generated candidate windows."
        )
    )
    parser.add_argument("--document-count", type=int, default=256)
    parser.add_argument("--document-vector-count", type=int, default=16)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-vector-count", type=int, default=8)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--kayak-plaid-centroids-per-query-vector",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--gpu-query-command",
        default="gpu-query",
        help="Executable used to probe Mojo-visible GPU devices.",
    )
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Exit successfully when no Mojo GPU is visible; the report stays partial.",
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_real_payload_rerank/input"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_real_payload_rerank/summary.json"),
    )
    return parser.parse_args(argv)


def build_shape(args: argparse.Namespace) -> SpeedTrackShape:
    shape = SpeedTrackShape(
        document_count=args.document_count,
        document_vector_count=args.document_vector_count,
        query_count=args.query_count,
        query_vector_count=args.query_vector_count,
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        update_document_count=0,
    )
    shape.validate()
    if shape.vector_dim != 128:
        raise ValueError("real payload GPU rerank profile requires vector_dim=128")
    if args.candidate_k < shape.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    return shape


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], MojoGpuCapability]:
    shape = build_shape(args)
    inputs = build_synthetic_inputs(
        shape,
        seed=args.seed,
        normalize_vectors=False,
    )
    exact_row, exact_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend="mojo_exact_cpu",
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=args.kayak_plaid_centroid_count,
            centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
            candidate_k=args.candidate_k,
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
    ranked_positions = rank_candidate_positions_by_score(
        candidate_positions,
        reference_scores,
        final_k=shape.top_k,
    )
    recall = mean_recall_at_k(
        candidate_positions_by_query=ranked_positions,
        reference_positions_by_query=exact_positions,
        k=shape.top_k,
    )

    candidate_timing = time_candidate_generation(
        index,
        inputs.queries,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    score_timing = time_same_candidate_scores(
        index,
        inputs.queries,
        candidate_positions,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    payload_export_started_at = time.perf_counter()
    payload = index.i8_payload_snapshot()
    input_manifest = write_gpu_i8_real_payload_probe_inputs(
        args.data_root,
        shape=shape,
        candidate_k=args.candidate_k,
        queries=inputs.queries,
        payload=payload,
        candidate_positions_by_query=candidate_positions,
        reference_scores_by_query=reference_scores,
    )
    payload_export_seconds = time.perf_counter() - payload_export_started_at

    capability = probe_mojo_gpu(args.gpu_query_command)
    gpu_probe: dict[str, object] | None = None
    bridge_probe: dict[str, object] | None = None
    prepared_bridge_probe: dict[str, object] | None = None
    prepared_ndarray_bridge_probe: dict[str, object] | None = None
    prepared_address_bridge_probe: dict[str, object] | None = None
    if capability.available:
        gpu_probe = run_gpu_i8_real_payload_score_probe(
            shape,
            candidate_k=args.candidate_k,
            data_root=args.data_root,
            target_accelerator=capability.target_accelerator,
        )
        bridge_probe = run_gpu_i8_bridge_probe(
            shape=shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
            queries=inputs.queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions,
            reference_scores_by_query=reference_scores,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
        prepared_bridge_probe = run_gpu_i8_prepared_bridge_probe(
            shape=shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
            queries=inputs.queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions,
            reference_scores_by_query=reference_scores,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
        prepared_ndarray_bridge_probe = run_gpu_i8_prepared_ndarray_bridge_probe(
            shape=shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
            queries=inputs.queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions,
            reference_scores_by_query=reference_scores,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
        prepared_address_bridge_probe = run_gpu_i8_prepared_address_bridge_probe(
            shape=shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
            queries=inputs.queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions,
            reference_scores_by_query=reference_scores,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
    status = report_status(
        capability=capability,
        gpu_probe=gpu_probe,
        bridge_probe=bridge_probe,
        prepared_bridge_probe=prepared_bridge_probe,
        prepared_ndarray_bridge_probe=prepared_ndarray_bridge_probe,
        prepared_address_bridge_probe=prepared_address_bridge_probe,
    )

    parsed = parsed_payload(gpu_probe)
    bridge_parsed = parsed_payload(bridge_probe)
    prepared_bridge_parsed = parsed_payload(prepared_bridge_probe)
    prepared_ndarray_bridge_parsed = parsed_payload(prepared_ndarray_bridge_probe)
    prepared_address_bridge_parsed = parsed_payload(prepared_address_bridge_probe)
    return (
        {
            "schema_version": 1,
            "benchmark": "gpu_i8_real_payload_rerank_profile",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "shape": shape_payload(shape, candidate_k=args.candidate_k),
            "controls": controls_payload(args),
            "mojo_gpu_capability": capability.to_json_ready(),
            "input_manifest": input_manifest,
            "cpu_reference": {
                "exact_reference": exact_row,
                "cpu_i8_candidate_generation": candidate_timing.to_json_ready()
                | {
                    "candidate_k": args.candidate_k,
                    "candidate_position_count_total": (
                        shape.query_count * args.candidate_k
                    ),
                },
                "cpu_i8_same_candidate_reference": score_timing.to_json_ready()
                | {
                    "candidate_score_count_total": (
                        shape.query_count * args.candidate_k
                    ),
                    "recall_at_k_vs_kayak_exact": recall,
                },
                "cpu_i8_build_seconds": build_seconds,
                "payload_export_seconds": payload_export_seconds,
            },
            "gpu_real_payload_probe": {
                "status": (
                    gpu_probe.get("status")
                    if gpu_probe is not None
                    else STATUS_PARTIAL_GPU_UNAVAILABLE
                ),
                "target_accelerator": capability.target_accelerator,
                "data_root": str(args.data_root),
                "parsed": parsed,
                "counts": gpu_probe.get("counts") if gpu_probe is not None else None,
                "derived": derive_candidate_score_metrics(parsed),
            },
            "gpu_real_payload_bridge": {
                "status": (
                    bridge_probe.get("status")
                    if bridge_probe is not None
                    else STATUS_PARTIAL_GPU_UNAVAILABLE
                ),
                "target_accelerator": capability.target_accelerator,
                "parsed": bridge_parsed,
                "derived": bridge_derived_metrics(bridge_parsed),
                "error": (
                    bridge_probe.get("error")
                    if isinstance(bridge_probe, dict)
                    else None
                ),
            },
            "gpu_real_payload_prepared_bridge": {
                "status": (
                    prepared_bridge_probe.get("status")
                    if prepared_bridge_probe is not None
                    else STATUS_PARTIAL_GPU_UNAVAILABLE
                ),
                "target_accelerator": capability.target_accelerator,
                "parsed": prepared_bridge_parsed,
                "derived": prepared_bridge_derived_metrics(
                    prepared_bridge_parsed
                ),
                "error": (
                    prepared_bridge_probe.get("error")
                    if isinstance(prepared_bridge_probe, dict)
                    else None
                ),
            },
            "gpu_real_payload_prepared_ndarray_bridge": {
                "status": (
                    prepared_ndarray_bridge_probe.get("status")
                    if prepared_ndarray_bridge_probe is not None
                    else STATUS_PARTIAL_GPU_UNAVAILABLE
                ),
                "target_accelerator": capability.target_accelerator,
                "parsed": prepared_ndarray_bridge_parsed,
                "derived": prepared_bridge_derived_metrics(
                    prepared_ndarray_bridge_parsed
                ),
                "error": (
                    prepared_ndarray_bridge_probe.get("error")
                    if isinstance(prepared_ndarray_bridge_probe, dict)
                    else None
                ),
            },
            "gpu_real_payload_prepared_address_bridge": {
                "status": (
                    prepared_address_bridge_probe.get("status")
                    if prepared_address_bridge_probe is not None
                    else STATUS_PARTIAL_GPU_UNAVAILABLE
                ),
                "target_accelerator": capability.target_accelerator,
                "parsed": prepared_address_bridge_parsed,
                "derived": prepared_bridge_derived_metrics(
                    prepared_address_bridge_parsed
                ),
                "error": (
                    prepared_address_bridge_probe.get("error")
                    if isinstance(prepared_address_bridge_probe, dict)
                    else None
                ),
            },
            "comparison": comparison_payload(
                parsed,
                bridge_parsed,
                prepared_bridge_parsed,
                prepared_ndarray_bridge_parsed,
                prepared_address_bridge_parsed,
                cpu_score_mean_seconds=score_timing.mean_seconds,
            ),
            "measurement_note": (
                "This is the first real-payload GPU profile. Candidate "
                "generation and top-k remain on CPU; the GPU row only scores "
                "the CPU-provided i8 candidate window. The prepared bridge "
                "keeps index buffers resident only inside one profiling "
                "session; it is not a reusable public backend object. The "
                "prepared ndarray row passes contiguous NumPy arrays directly "
                "to the same Mojo function to isolate Python list materialization. "
                "The prepared address row passes raw typed array addresses to "
                "avoid per-element PythonObject indexing inside Mojo."
            ),
        },
        capability,
    )


def time_candidate_generation(
    index: KayakPlaidApproxIndex,
    queries: Any,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> TimingSummary:
    for _ in range(warmup_iterations):
        index.i8_candidate_positions_batch(queries)

    durations: list[float] = []
    for _ in range(measurement_iterations):
        started_at = time.perf_counter()
        index.i8_candidate_positions_batch(queries)
        durations.append(time.perf_counter() - started_at)
    return timing_summary(durations)


def time_same_candidate_scores(
    index: KayakPlaidApproxIndex,
    queries: Any,
    candidate_positions: Sequence[Sequence[int]],
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> TimingSummary:
    for _ in range(warmup_iterations):
        index.i8_score_candidate_positions_batch(queries, candidate_positions)

    durations: list[float] = []
    for _ in range(measurement_iterations):
        started_at = time.perf_counter()
        index.i8_score_candidate_positions_batch(queries, candidate_positions)
        durations.append(time.perf_counter() - started_at)
    return timing_summary(durations)


def rank_candidate_positions_by_score(
    candidate_positions_by_query: Sequence[Sequence[int]],
    scores_by_query: Sequence[Sequence[float]],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    ranked_rows: list[tuple[int, ...]] = []
    for candidate_positions, scores in zip(
        candidate_positions_by_query,
        scores_by_query,
        strict=True,
    ):
        ranked_offsets = sorted(
            range(len(candidate_positions)),
            key=lambda offset: scores[offset],
            reverse=True,
        )
        ranked_rows.append(
            tuple(candidate_positions[offset] for offset in ranked_offsets[:final_k])
        )
    return tuple(ranked_rows)


def timing_summary(durations: Sequence[float]) -> TimingSummary:
    if not durations:
        raise ValueError("durations must not be empty")
    ordered = sorted(float(duration) for duration in durations)
    return TimingSummary(
        min_seconds=ordered[0],
        median_seconds=ordered[len(ordered) // 2],
        mean_seconds=float(sum(ordered) / len(ordered)),
        max_seconds=ordered[-1],
    )


def report_status(
    *,
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
    bridge_probe: dict[str, object] | None,
    prepared_bridge_probe: dict[str, object] | None,
    prepared_ndarray_bridge_probe: dict[str, object] | None,
    prepared_address_bridge_probe: dict[str, object] | None,
) -> str:
    if not capability.available:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if gpu_probe is None or gpu_probe.get("status") != GPU_REAL_PAYLOAD_STATUS_OK:
        return STATUS_BLOCKED_GPU_REAL_PAYLOAD_FAILED
    if bridge_probe is None or bridge_probe.get("status") != STATUS_OK:
        return STATUS_BLOCKED_GPU_BRIDGE_FAILED
    if (
        prepared_bridge_probe is None
        or prepared_bridge_probe.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_PREPARED_BRIDGE_FAILED
    if (
        prepared_ndarray_bridge_probe is None
        or prepared_ndarray_bridge_probe.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_PREPARED_NDARRAY_BRIDGE_FAILED
    if (
        prepared_address_bridge_probe is None
        or prepared_address_bridge_probe.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_PREPARED_ADDRESS_BRIDGE_FAILED
    return STATUS_OK


def shape_payload(shape: SpeedTrackShape, *, candidate_k: int) -> dict[str, int]:
    return shape.to_json_ready() | {
        "candidate_k": candidate_k,
        "candidate_score_count_total": shape.query_count * candidate_k,
    }


def controls_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "kayak_plaid_centroid_count": args.kayak_plaid_centroid_count,
        "kayak_plaid_centroids_per_query_vector": (
            args.kayak_plaid_centroids_per_query_vector
        ),
        "kayak_plaid_candidate_k": args.candidate_k,
        "kayak_plaid_payload": "i8",
        "warmup_iterations": args.warmup_iterations,
        "measurement_iterations": args.measurement_iterations,
    }


def comparison_payload(
    parsed: dict[str, object],
    bridge_parsed: dict[str, object],
    prepared_bridge_parsed: dict[str, object],
    prepared_ndarray_bridge_parsed: dict[str, object],
    prepared_address_bridge_parsed: dict[str, object],
    *,
    cpu_score_mean_seconds: float,
) -> dict[str, float | None]:
    kernel = optional_float(parsed.get("twopass_kernel_mean_seconds"))
    h2d = optional_float(parsed.get("host_to_device_mean_seconds"))
    d2h = optional_float(parsed.get("device_to_host_mean_seconds"))
    gpu_e2e = sum_optional(h2d, kernel, d2h)
    bridge_kernel = optional_float(bridge_parsed.get("twopass_kernel_mean_seconds"))
    bridge_h2d = optional_float(bridge_parsed.get("host_to_device_mean_seconds"))
    bridge_d2h = optional_float(bridge_parsed.get("device_to_host_mean_seconds"))
    bridge_e2e = sum_optional(bridge_h2d, bridge_kernel, bridge_d2h)
    prepared_prepare_h2d = optional_float(
        prepared_bridge_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_score_h2d = optional_float(
        prepared_bridge_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_kernel = optional_float(
        prepared_bridge_parsed.get("twopass_kernel_mean_seconds")
    )
    prepared_d2h = optional_float(
        prepared_bridge_parsed.get("device_to_host_mean_seconds")
    )
    prepared_score_e2e = sum_optional(
        prepared_score_h2d,
        prepared_kernel,
        prepared_d2h,
    )
    prepared_ndarray_prepare_h2d = optional_float(
        prepared_ndarray_bridge_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_ndarray_score_h2d = optional_float(
        prepared_ndarray_bridge_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_ndarray_kernel = optional_float(
        prepared_ndarray_bridge_parsed.get("twopass_kernel_mean_seconds")
    )
    prepared_ndarray_d2h = optional_float(
        prepared_ndarray_bridge_parsed.get("device_to_host_mean_seconds")
    )
    prepared_ndarray_score_e2e = sum_optional(
        prepared_ndarray_score_h2d,
        prepared_ndarray_kernel,
        prepared_ndarray_d2h,
    )
    prepared_address_prepare_h2d = optional_float(
        prepared_address_bridge_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_address_score_h2d = optional_float(
        prepared_address_bridge_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_address_kernel = optional_float(
        prepared_address_bridge_parsed.get("twopass_kernel_mean_seconds")
    )
    prepared_address_d2h = optional_float(
        prepared_address_bridge_parsed.get("device_to_host_mean_seconds")
    )
    prepared_address_score_e2e = sum_optional(
        prepared_address_score_h2d,
        prepared_address_kernel,
        prepared_address_d2h,
    )
    return {
        "gpu_twopass_kernel_mean_seconds": kernel,
        "gpu_h2d_kernel_d2h_mean_seconds": gpu_e2e,
        "gpu_bridge_twopass_kernel_mean_seconds": bridge_kernel,
        "gpu_bridge_h2d_kernel_d2h_mean_seconds": bridge_e2e,
        "gpu_bridge_extension_call_seconds": optional_float(
            bridge_parsed.get("extension_call_seconds")
        ),
        "gpu_bridge_host_marshalling_seconds": optional_float(
            bridge_parsed.get("host_marshalling_seconds")
        ),
        "gpu_prepared_bridge_prepare_h2d_mean_seconds": prepared_prepare_h2d,
        "gpu_prepared_bridge_twopass_kernel_mean_seconds": prepared_kernel,
        "gpu_prepared_bridge_score_h2d_kernel_d2h_mean_seconds": (
            prepared_score_e2e
        ),
        "gpu_prepared_bridge_extension_call_seconds": optional_float(
            prepared_bridge_parsed.get("extension_call_seconds")
        ),
        "gpu_prepared_bridge_host_marshalling_seconds": optional_float(
            prepared_bridge_parsed.get("host_marshalling_seconds")
        ),
        "gpu_prepared_ndarray_bridge_prepare_h2d_mean_seconds": (
            prepared_ndarray_prepare_h2d
        ),
        "gpu_prepared_ndarray_bridge_twopass_kernel_mean_seconds": (
            prepared_ndarray_kernel
        ),
        "gpu_prepared_ndarray_bridge_score_h2d_kernel_d2h_mean_seconds": (
            prepared_ndarray_score_e2e
        ),
        "gpu_prepared_ndarray_bridge_extension_call_seconds": optional_float(
            prepared_ndarray_bridge_parsed.get("extension_call_seconds")
        ),
        "gpu_prepared_ndarray_bridge_host_marshalling_seconds": optional_float(
            prepared_ndarray_bridge_parsed.get("host_marshalling_seconds")
        ),
        "gpu_prepared_address_bridge_prepare_h2d_mean_seconds": (
            prepared_address_prepare_h2d
        ),
        "gpu_prepared_address_bridge_twopass_kernel_mean_seconds": (
            prepared_address_kernel
        ),
        "gpu_prepared_address_bridge_score_h2d_kernel_d2h_mean_seconds": (
            prepared_address_score_e2e
        ),
        "gpu_prepared_address_bridge_extension_call_seconds": optional_float(
            prepared_address_bridge_parsed.get("extension_call_seconds")
        ),
        "gpu_prepared_address_bridge_host_marshalling_seconds": optional_float(
            prepared_address_bridge_parsed.get("host_marshalling_seconds")
        ),
        "cpu_i8_same_candidate_score_mean_seconds": cpu_score_mean_seconds,
        "gpu_kernel_seconds_per_cpu_score_second": ratio(
            kernel,
            cpu_score_mean_seconds,
        ),
        "gpu_h2d_kernel_d2h_seconds_per_cpu_score_second": ratio(
            gpu_e2e,
            cpu_score_mean_seconds,
        ),
        "gpu_bridge_kernel_seconds_per_cpu_score_second": ratio(
            bridge_kernel,
            cpu_score_mean_seconds,
        ),
        "gpu_bridge_h2d_kernel_d2h_seconds_per_cpu_score_second": ratio(
            bridge_e2e,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_bridge_kernel_seconds_per_cpu_score_second": ratio(
            prepared_kernel,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_bridge_score_h2d_kernel_d2h_seconds_per_cpu_score_second": ratio(
            prepared_score_e2e,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_ndarray_bridge_kernel_seconds_per_cpu_score_second": ratio(
            prepared_ndarray_kernel,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_ndarray_bridge_score_h2d_kernel_d2h_seconds_per_cpu_score_second": ratio(
            prepared_ndarray_score_e2e,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_address_bridge_kernel_seconds_per_cpu_score_second": ratio(
            prepared_address_kernel,
            cpu_score_mean_seconds,
        ),
        "gpu_prepared_address_bridge_score_h2d_kernel_d2h_seconds_per_cpu_score_second": ratio(
            prepared_address_score_e2e,
            cpu_score_mean_seconds,
        ),
    }


def print_quiet_sections(report: dict[str, Any]) -> None:
    cpu_reference = report.get("cpu_reference")
    if isinstance(cpu_reference, dict):
        score_row = cpu_reference.get("cpu_i8_same_candidate_reference")
        if isinstance(score_row, dict):
            mean_seconds = score_row.get("mean_seconds")
            if isinstance(mean_seconds, (float, int)):
                print("== cpu_i8_same_candidate_score_query_batch ==")
                print("Mean:", mean_seconds)

    parsed = parsed_payload(report.get("gpu_real_payload_probe"))
    kernel = parsed.get("twopass_kernel_mean_seconds")
    h2d = optional_float(parsed.get("host_to_device_mean_seconds"))
    d2h = optional_float(parsed.get("device_to_host_mean_seconds"))
    if isinstance(kernel, (float, int)):
        print("== gpu_i8_real_payload_twopass_kernel ==")
        print("Mean:", kernel)
    gpu_e2e = sum_optional(h2d, optional_float(kernel), d2h)
    if gpu_e2e is not None:
        print("== gpu_i8_real_payload_h2d_kernel_d2h ==")
        print("Mean:", gpu_e2e)

    bridge_parsed = parsed_payload(report.get("gpu_real_payload_bridge"))
    bridge_kernel = bridge_parsed.get("twopass_kernel_mean_seconds")
    bridge_h2d = optional_float(bridge_parsed.get("host_to_device_mean_seconds"))
    bridge_d2h = optional_float(bridge_parsed.get("device_to_host_mean_seconds"))
    if isinstance(bridge_kernel, (float, int)):
        print("== gpu_i8_real_payload_bridge_twopass_kernel ==")
        print("Mean:", bridge_kernel)
    bridge_e2e = sum_optional(bridge_h2d, optional_float(bridge_kernel), bridge_d2h)
    if bridge_e2e is not None:
        print("== gpu_i8_real_payload_bridge_h2d_kernel_d2h ==")
        print("Mean:", bridge_e2e)

    prepared_parsed = parsed_payload(report.get("gpu_real_payload_prepared_bridge"))
    prepared_prepare_h2d = optional_float(
        prepared_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_score_h2d = optional_float(
        prepared_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_kernel = prepared_parsed.get("twopass_kernel_mean_seconds")
    prepared_d2h = optional_float(prepared_parsed.get("device_to_host_mean_seconds"))
    if prepared_prepare_h2d is not None:
        print("== gpu_i8_real_payload_prepared_bridge_prepare_h2d ==")
        print("Mean:", prepared_prepare_h2d)
    if isinstance(prepared_kernel, (float, int)):
        print("== gpu_i8_real_payload_prepared_bridge_twopass_kernel ==")
        print("Mean:", prepared_kernel)
    prepared_e2e = sum_optional(
        prepared_score_h2d,
        optional_float(prepared_kernel),
        prepared_d2h,
    )
    if prepared_e2e is not None:
        print("== gpu_i8_real_payload_prepared_bridge_score_h2d_kernel_d2h ==")
        print("Mean:", prepared_e2e)

    prepared_ndarray_parsed = parsed_payload(
        report.get("gpu_real_payload_prepared_ndarray_bridge")
    )
    prepared_ndarray_prepare_h2d = optional_float(
        prepared_ndarray_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_ndarray_score_h2d = optional_float(
        prepared_ndarray_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_ndarray_kernel = prepared_ndarray_parsed.get(
        "twopass_kernel_mean_seconds"
    )
    prepared_ndarray_d2h = optional_float(
        prepared_ndarray_parsed.get("device_to_host_mean_seconds")
    )
    if prepared_ndarray_prepare_h2d is not None:
        print("== gpu_i8_real_payload_prepared_ndarray_bridge_prepare_h2d ==")
        print("Mean:", prepared_ndarray_prepare_h2d)
    if isinstance(prepared_ndarray_kernel, (float, int)):
        print("== gpu_i8_real_payload_prepared_ndarray_bridge_twopass_kernel ==")
        print("Mean:", prepared_ndarray_kernel)
    prepared_ndarray_e2e = sum_optional(
        prepared_ndarray_score_h2d,
        optional_float(prepared_ndarray_kernel),
        prepared_ndarray_d2h,
    )
    if prepared_ndarray_e2e is not None:
        print(
            "== "
            "gpu_i8_real_payload_prepared_ndarray_bridge_score_h2d_kernel_d2h"
            " =="
        )
        print("Mean:", prepared_ndarray_e2e)

    prepared_address_parsed = parsed_payload(
        report.get("gpu_real_payload_prepared_address_bridge")
    )
    prepared_address_prepare_h2d = optional_float(
        prepared_address_parsed.get("prepare_host_to_device_mean_seconds")
    )
    prepared_address_score_h2d = optional_float(
        prepared_address_parsed.get("score_host_to_device_mean_seconds")
    )
    prepared_address_kernel = prepared_address_parsed.get(
        "twopass_kernel_mean_seconds"
    )
    prepared_address_d2h = optional_float(
        prepared_address_parsed.get("device_to_host_mean_seconds")
    )
    if prepared_address_prepare_h2d is not None:
        print("== gpu_i8_real_payload_prepared_address_bridge_prepare_h2d ==")
        print("Mean:", prepared_address_prepare_h2d)
    if isinstance(prepared_address_kernel, (float, int)):
        print("== gpu_i8_real_payload_prepared_address_bridge_twopass_kernel ==")
        print("Mean:", prepared_address_kernel)
    prepared_address_e2e = sum_optional(
        prepared_address_score_h2d,
        optional_float(prepared_address_kernel),
        prepared_address_d2h,
    )
    if prepared_address_e2e is not None:
        print(
            "== "
            "gpu_i8_real_payload_prepared_address_bridge_score_h2d_kernel_d2h"
            " =="
        )
        print("Mean:", prepared_address_e2e)


def exit_code(
    report: dict[str, Any],
    capability: MojoGpuCapability,
    args: argparse.Namespace,
) -> int:
    status = report.get("status")
    if status == STATUS_OK:
        return 0
    if status == STATUS_PARTIAL_GPU_UNAVAILABLE and args.allow_missing_gpu:
        return 0
    if not capability.available and not args.allow_missing_gpu:
        return 2
    if status == STATUS_BLOCKED_GPU_REAL_PAYLOAD_FAILED:
        return 3
    if status == STATUS_BLOCKED_GPU_BRIDGE_FAILED:
        return 5
    if status == STATUS_BLOCKED_GPU_PREPARED_BRIDGE_FAILED:
        return 6
    if status == STATUS_BLOCKED_GPU_PREPARED_NDARRAY_BRIDGE_FAILED:
        return 7
    if status == STATUS_BLOCKED_GPU_PREPARED_ADDRESS_BRIDGE_FAILED:
        return 8
    return 4


def run_gpu_i8_bridge_probe(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    target_accelerator: str | None,
    queries: Any,
    payload: Any,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object]:
    if target_accelerator is None:
        return {
            "status": STATUS_PARTIAL_GPU_UNAVAILABLE,
            "parsed": {},
            "error": "target_accelerator was not available",
        }

    try:
        for _ in range(warmup_iterations):
            score_i8_real_payload_once(
                target_accelerator=target_accelerator,
                shape=shape,
                candidate_k=candidate_k,
                queries=queries,
                payload=payload,
                candidate_positions_by_query=candidate_positions_by_query,
                reference_scores_by_query=reference_scores_by_query,
            )

        results = [
            score_i8_real_payload_once(
                target_accelerator=target_accelerator,
                shape=shape,
                candidate_k=candidate_k,
                queries=queries,
                payload=payload,
                candidate_positions_by_query=candidate_positions_by_query,
                reference_scores_by_query=reference_scores_by_query,
            )
            for _ in range(measurement_iterations)
        ]
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {
            "status": "error",
            "parsed": {},
            "error": str(exc),
        }
    if not results:
        raise ValueError("measurement_iterations must be positive")

    candidate_score_count = results[0].candidate_score_count
    if any(
        result.candidate_score_count != candidate_score_count
        for result in results
    ):
        raise RuntimeError("GPU bridge candidate score count changed between runs")

    parsed = {
        "payload_source": "real_kayak_i8_snapshot",
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "candidate_k": candidate_k,
        "candidate_score_count": candidate_score_count,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "host_marshalling_seconds": mean_result(
            results, "host_marshalling_seconds"
        ),
        "extension_call_seconds": mean_result(results, "extension_call_seconds"),
        "host_to_device_mean_seconds": mean_result(
            results, "host_to_device_mean_seconds"
        ),
        "twopass_kernel_mean_seconds": mean_result(results, "kernel_mean_seconds"),
        "device_to_host_mean_seconds": mean_result(
            results, "device_to_host_mean_seconds"
        ),
        "score_delta_max_abs": max(
            result.score_delta_max_abs for result in results
        ),
        "score_agreement_ok": all(
            result.score_delta_max_abs <= 0.0001 for result in results
        ),
    }
    return {
        "status": (
            STATUS_OK
            if parsed["score_agreement_ok"]
            else "score_agreement_failed"
        ),
        "parsed": parsed,
        "measurements": [result.to_json_ready() for result in results],
    }


def run_gpu_i8_prepared_bridge_probe(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    target_accelerator: str | None,
    queries: Any,
    payload: Any,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object]:
    return _run_gpu_i8_prepared_bridge_probe(
        shape=shape,
        candidate_k=candidate_k,
        target_accelerator=target_accelerator,
        queries=queries,
        payload=payload,
        candidate_positions_by_query=candidate_positions_by_query,
        reference_scores_by_query=reference_scores_by_query,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        bridge_scope="single_extension_call_resident_index_session",
        profile_fn=profile_i8_prepared_payload_session,
    )


def run_gpu_i8_prepared_ndarray_bridge_probe(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    target_accelerator: str | None,
    queries: Any,
    payload: Any,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object]:
    return _run_gpu_i8_prepared_bridge_probe(
        shape=shape,
        candidate_k=candidate_k,
        target_accelerator=target_accelerator,
        queries=queries,
        payload=payload,
        candidate_positions_by_query=candidate_positions_by_query,
        reference_scores_by_query=reference_scores_by_query,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        bridge_scope="single_extension_call_resident_index_session_ndarray",
        profile_fn=profile_i8_prepared_payload_session_ndarray,
    )


def run_gpu_i8_prepared_address_bridge_probe(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    target_accelerator: str | None,
    queries: Any,
    payload: Any,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, object]:
    return _run_gpu_i8_prepared_bridge_probe(
        shape=shape,
        candidate_k=candidate_k,
        target_accelerator=target_accelerator,
        queries=queries,
        payload=payload,
        candidate_positions_by_query=candidate_positions_by_query,
        reference_scores_by_query=reference_scores_by_query,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        bridge_scope="single_extension_call_resident_index_session_address",
        profile_fn=profile_i8_prepared_payload_session_addresses,
    )


def _run_gpu_i8_prepared_bridge_probe(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    target_accelerator: str | None,
    queries: Any,
    payload: Any,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_scores_by_query: Sequence[Sequence[float]],
    warmup_iterations: int,
    measurement_iterations: int,
    bridge_scope: str,
    profile_fn: Any,
) -> dict[str, object]:
    if target_accelerator is None:
        return {
            "status": STATUS_PARTIAL_GPU_UNAVAILABLE,
            "parsed": {},
            "error": "target_accelerator was not available",
        }

    try:
        result = profile_fn(
            target_accelerator=target_accelerator,
            shape=shape,
            candidate_k=candidate_k,
            queries=queries,
            payload=payload,
            candidate_positions_by_query=candidate_positions_by_query,
            reference_scores_by_query=reference_scores_by_query,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
    except Exception as exc:  # pragma: no cover - exercised by GPU environments.
        return {
            "status": "error",
            "parsed": {},
            "error": str(exc),
        }

    parsed = {
        "payload_source": "real_kayak_i8_snapshot",
        "bridge_scope": bridge_scope,
        "query_count": shape.query_count,
        "query_vector_count": shape.query_vector_count,
        "document_count": shape.document_count,
        "document_vector_count": shape.document_vector_count,
        "total_document_vector_count": (
            shape.document_count * shape.document_vector_count
        ),
        "resident_index_token_code_count": (
            shape.document_count * shape.document_vector_count * shape.vector_dim
        ),
        "resident_index_token_scale_count": (
            shape.document_count * shape.document_vector_count
        ),
        "resident_index_doc_offset_count": shape.document_count + 1,
        "candidate_k": candidate_k,
        "candidate_score_count": result.candidate_score_count,
        "vector_dim": shape.vector_dim,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "host_marshalling_seconds": result.host_marshalling_seconds,
        "extension_call_seconds": result.extension_call_seconds,
        "prepare_host_to_device_mean_seconds": (
            result.prepare_host_to_device_mean_seconds
        ),
        "score_host_to_device_mean_seconds": (
            result.score_host_to_device_mean_seconds
        ),
        "twopass_kernel_mean_seconds": result.kernel_mean_seconds,
        "device_to_host_mean_seconds": result.device_to_host_mean_seconds,
        "score_delta_max_abs": result.score_delta_max_abs,
        "score_agreement_ok": result.score_delta_max_abs <= 0.0001,
    }
    return {
        "status": (
            STATUS_OK
            if parsed["score_agreement_ok"]
            else "score_agreement_failed"
        ),
        "parsed": parsed,
        "measurements": [result.to_json_ready()],
    }


def bridge_derived_metrics(parsed: dict[str, object]) -> dict[str, float | None]:
    kernel = optional_float(parsed.get("twopass_kernel_mean_seconds"))
    h2d = optional_float(parsed.get("host_to_device_mean_seconds"))
    d2h = optional_float(parsed.get("device_to_host_mean_seconds"))
    extension_call = optional_float(parsed.get("extension_call_seconds"))
    host_marshalling = optional_float(parsed.get("host_marshalling_seconds"))
    return {
        "h2d_kernel_d2h_mean_seconds": sum_optional(h2d, kernel, d2h),
        "host_marshalling_seconds": host_marshalling,
        "extension_call_seconds": extension_call,
    }


def prepared_bridge_derived_metrics(
    parsed: dict[str, object],
) -> dict[str, float | None]:
    prepare_h2d = optional_float(
        parsed.get("prepare_host_to_device_mean_seconds")
    )
    score_h2d = optional_float(parsed.get("score_host_to_device_mean_seconds"))
    kernel = optional_float(parsed.get("twopass_kernel_mean_seconds"))
    d2h = optional_float(parsed.get("device_to_host_mean_seconds"))
    extension_call = optional_float(parsed.get("extension_call_seconds"))
    host_marshalling = optional_float(parsed.get("host_marshalling_seconds"))
    return {
        "prepare_host_to_device_mean_seconds": prepare_h2d,
        "score_h2d_kernel_d2h_mean_seconds": sum_optional(
            score_h2d,
            kernel,
            d2h,
        ),
        "host_marshalling_seconds": host_marshalling,
        "extension_call_seconds": extension_call,
    }


def mean_result(results: Sequence[Any], field: str) -> float:
    return float(sum(float(getattr(result, field)) for result in results) / len(results))


def parsed_payload(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    parsed = value.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, capability = build_report(args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        print_quiet_sections(report)
    return exit_code(report, capability, args)


if __name__ == "__main__":
    raise SystemExit(main())
