from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak  # noqa: E402
from kayak_bridge.gpu_device_capability import (  # noqa: E402
    MojoGpuCapability,
    probe_mojo_gpu,
)
from kayak_bridge.gpu_i8_candidate_score import (  # noqa: E402
    GPU_CANDIDATE_SCORE_STATUS_OK,
    run_gpu_i8_candidate_score_probe,
)
from kayak_bridge.gpu_i8_fastplaid_fused_scope import (  # noqa: E402
    build_fused_centroid_posting_scope_row,
    build_missing_fused_centroid_posting_scope_row,
)
from kayak_bridge.gpu_i8_fastplaid_hybrid_metrics import (  # noqa: E402
    STATUS_BLOCKED_GPU_HYBRID_FAILED,
    build_gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_comparison,
)
from kayak_bridge.gpu_i8_fastplaid_hybrid_scope import (  # noqa: E402
    build_hybrid_shortlist_exact_rerank_scope_row,
    build_missing_hybrid_shortlist_exact_rerank_scope_row,
)
from kayak_bridge.gpu_i8_fastplaid_topk_compare import (  # noqa: E402
    build_missing_prepared_handle_topk_scope_row,
    build_prepared_handle_topk_scope_row,
)
from kayak_bridge.gpu_i8_fastplaid_topk_metrics import (  # noqa: E402
    STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED,
    STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED,
    build_gpu_fused_centroid_posting_vs_fastplaid_comparison,
    build_gpu_prepared_topk_no_reference_vs_fastplaid_comparison,
    build_gpu_prepared_topk_vs_fastplaid_comparison,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig  # noqa: E402

from bench_fastplaid_speed_track import (  # noqa: E402
    FASTPLAID_BLOG_URL,
    FASTPLAID_REPO_URL,
    SpeedTrackShape,
    SyntheticInputs,
    benchmark_fastplaid,
    benchmark_kayak_exact,
    benchmark_kayak_plaid,
    build_pairwise_rows,
    build_synthetic_inputs,
    write_report,
)
from profile_gpu_i8_candidate_score import (  # noqa: E402
    derive_candidate_score_metrics,
)


STATUS_OK = "ok"
STATUS_PARTIAL_GPU_UNAVAILABLE = "partial_gpu_unavailable"
STATUS_BLOCKED_GPU_CANDIDATE_SCORE_FAILED = "blocked_gpu_candidate_score_failed"
STATUS_BLOCKED_FASTPLAID_UNAVAILABLE = "blocked_fastplaid_unavailable"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the benchmark-only GPU i8 candidate-score primitive with "
            "the existing FastPlaid full-search speed track on one explicit "
            "dim128 synthetic shape."
        )
    )
    parser.add_argument("--document-count", type=int, default=256)
    parser.add_argument("--document-vector-count", type=int, default=16)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-vector-count", type=int, default=8)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--candidate-k", type=int, default=128)
    parser.add_argument(
        "--gpu-hybrid-shortlist-k",
        type=int,
        default=None,
        help=(
            "Optional fused-shortlist size for the hybrid GPU primitive. "
            "Defaults to candidate-k clipped to document-count."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize synthetic token vectors before indexing.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=1)
    parser.add_argument(
        "--gpu-topk-session-iterations",
        type=int,
        default=4,
        help=(
            "Prepared-handle top-k GPU score windows measured per report row. "
            "Each window keeps the same explicit query/document vector counts."
        ),
    )
    parser.add_argument(
        "--kayak-backend",
        choices=(kayak.MOJO_EXACT_CPU_BACKEND, kayak.NUMPY_REFERENCE_BACKEND),
        default=kayak.MOJO_EXACT_CPU_BACKEND,
    )
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--kayak-plaid-centroids-per-query-vector",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--kayak-i8-candidate-order",
        choices=("ordered", "unordered"),
        default="ordered",
        help=(
            "Internal GPU pipeline candidate-window order. Unordered keeps "
            "the retained candidate set but skips approximate-score ordering."
        ),
    )
    parser.add_argument(
        "--kayak-i8-positive-centroids-only",
        action="store_true",
        help=(
            "Benchmark-only: generate unordered i8 candidate windows from "
            "positive selected centroid postings only."
        ),
    )
    parser.add_argument(
        "--fastplaid-device",
        default="cpu",
        help='FastPlaid device string, e.g. "cpu", "cuda", or an empty auto value.',
    )
    parser.add_argument(
        "--fastplaid-low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to FastPlaid. On GPU this controls index residency.",
    )
    parser.add_argument("--fastplaid-kmeans-niters", type=int, default=4)
    parser.add_argument("--fastplaid-max-points-per-centroid", type=int, default=256)
    parser.add_argument("--fastplaid-nbits", type=int, default=4)
    parser.add_argument("--fastplaid-batch-size", type=int, default=25_000)
    parser.add_argument(
        "--fastplaid-use-triton-kmeans",
        choices=("auto", "true", "false"),
        default="auto",
    )
    parser.add_argument("--fastplaid-update-buffer-size", type=int, default=100)
    parser.add_argument(
        "--require-fastplaid",
        action="store_true",
        help="Fail instead of writing a skipped FastPlaid row when unavailable.",
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
        "--index-root",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_fastplaid_compare/indexes"),
    )
    parser.add_argument(
        "--overwrite-index-root",
        action="store_true",
        help="Remove existing FastPlaid index directories before benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_fastplaid_compare/summary.json"),
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
        raise ValueError(
            "GPU i8 FastPlaid comparison currently requires vector_dim=128"
        )
    if args.candidate_k < shape.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")
    if args.gpu_hybrid_shortlist_k is not None:
        if args.gpu_hybrid_shortlist_k < shape.top_k:
            raise ValueError(
                "gpu_hybrid_shortlist_k must be greater than or equal to top_k"
            )
        if args.gpu_hybrid_shortlist_k > shape.document_count:
            raise ValueError(
                "gpu_hybrid_shortlist_k must be no larger than document_count"
            )
    if args.gpu_topk_session_iterations <= 0:
        raise ValueError("gpu_topk_session_iterations must be positive")
    if (
        args.kayak_i8_positive_centroids_only
        and args.kayak_i8_candidate_order != "unordered"
    ):
        raise ValueError("positive centroid candidates require unordered order")
    return shape


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], MojoGpuCapability]:
    shape = build_shape(args)
    inputs = build_synthetic_inputs(
        shape,
        seed=args.seed,
        normalize_vectors=args.normalize_vectors,
    )
    capability = probe_mojo_gpu(args.gpu_query_command)
    systems, reference_positions = benchmark_system_rows(
        shape=shape,
        inputs=inputs,
        args=args,
    )

    gpu_probe: dict[str, object] | None = None
    prepared_handle_topk_row: dict[str, Any] | None = None
    fused_centroid_posting_row: dict[str, Any] | None = None
    hybrid_shortlist_rerank_row: dict[str, Any] | None = None
    if capability.available:
        gpu_probe = run_gpu_i8_candidate_score_probe(
            shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
        )
        prepared_handle_topk_row = build_prepared_handle_topk_scope_row(
            shape=shape,
            inputs=inputs,
            reference_positions=reference_positions,
            capability=capability,
            args=args,
        )
        fused_centroid_posting_row = build_fused_centroid_posting_scope_row(
            shape=shape,
            inputs=inputs,
            reference_positions=reference_positions,
            capability=capability,
            args=args,
        )
        hybrid_shortlist_rerank_row = (
            build_hybrid_shortlist_exact_rerank_scope_row(
                shape=shape,
                inputs=inputs,
                reference_positions=reference_positions,
                capability=capability,
                args=args,
            )
        )

    fastplaid_row = _system_by_name(systems, "fastplaid")
    gpu_scope_row = build_gpu_scope_row(
        shape=shape,
        candidate_k=args.candidate_k,
        capability=capability,
        gpu_probe=gpu_probe,
    )
    comparison = build_gpu_vs_fastplaid_comparison(
        gpu_probe=gpu_probe,
        fastplaid_row=fastplaid_row,
    )
    prepared_handle_topk_comparison = (
        build_gpu_prepared_topk_vs_fastplaid_comparison(
            prepared_handle_topk_row=prepared_handle_topk_row,
            fastplaid_row=fastplaid_row,
        )
    )
    prepared_handle_topk_no_reference_comparison = (
        build_gpu_prepared_topk_no_reference_vs_fastplaid_comparison(
            prepared_handle_topk_row=prepared_handle_topk_row,
            fastplaid_row=fastplaid_row,
        )
    )
    fused_centroid_posting_comparison = (
        build_gpu_fused_centroid_posting_vs_fastplaid_comparison(
            fused_row=fused_centroid_posting_row,
            fastplaid_row=fastplaid_row,
        )
    )
    hybrid_shortlist_rerank_comparison = (
        build_gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_comparison(
            hybrid_row=hybrid_shortlist_rerank_row,
            fastplaid_row=fastplaid_row,
        )
    )
    status = report_status(
        capability=capability,
        gpu_probe=gpu_probe,
        prepared_handle_topk_row=prepared_handle_topk_row,
        fused_centroid_posting_row=fused_centroid_posting_row,
        hybrid_shortlist_rerank_row=hybrid_shortlist_rerank_row,
        fastplaid_row=fastplaid_row,
    )

    return (
        {
            "schema_version": 1,
            "benchmark": "gpu_i8_fastplaid_scope_compare",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": status,
            "source_urls": {
                "fastplaid_blog": FASTPLAID_BLOG_URL,
                "fastplaid_repo": FASTPLAID_REPO_URL,
            },
            "shape": shape_payload(shape, candidate_k=args.candidate_k),
            "controls": controls_payload(args),
            "input_bytes": input_bytes_payload(inputs),
            "systems": systems,
            "pairwise_vs_kayak_exact": build_pairwise_rows(systems),
            "mojo_gpu_capability": capability.to_json_ready(),
            "gpu_candidate_score_primitive": gpu_scope_row,
            "gpu_prepared_handle_topk_primitive": (
                prepared_handle_topk_row
                if prepared_handle_topk_row is not None
                else build_missing_prepared_handle_topk_scope_row(
                    shape=shape,
                    candidate_k=args.candidate_k,
                    capability=capability,
                )
            ),
            "gpu_fused_centroid_posting_topk_primitive": (
                fused_centroid_posting_row
                if fused_centroid_posting_row is not None
                else build_missing_fused_centroid_posting_scope_row(
                    shape=shape,
                    candidate_k=args.candidate_k,
                    capability=capability,
                )
            ),
            "gpu_hybrid_shortlist_exact_rerank_primitive": (
                hybrid_shortlist_rerank_row
                if hybrid_shortlist_rerank_row is not None
                else build_missing_hybrid_shortlist_exact_rerank_scope_row(
                    shape=shape,
                    candidate_k=args.candidate_k,
                    capability=capability,
                )
            ),
            "gpu_vs_fastplaid_scope_comparison": comparison,
            "gpu_prepared_handle_topk_vs_fastplaid_scope_comparison": (
                prepared_handle_topk_comparison
            ),
            "gpu_prepared_handle_topk_no_reference_vs_fastplaid_scope_comparison": (
                prepared_handle_topk_no_reference_comparison
            ),
            "gpu_fused_centroid_posting_vs_fastplaid_scope_comparison": (
                fused_centroid_posting_comparison
            ),
            "gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_scope_comparison": (
                hybrid_shortlist_rerank_comparison
            ),
            "measurement_note": (
                "FastPlaid rows are full-search timings. The GPU row is a "
                "benchmark-only candidate-score primitive over deterministic "
                "flat i8 tensors. The prepared-handle top-k GPU row uses real "
                "Kayak i8 payload snapshots and CPU-provided candidate "
                "windows, but is still an internal rerank boundary rather than "
                "a full search backend. The fused centroid-posting row keeps "
                "candidate generation inside the GPU primitive and returns "
                "top-k positions from the prepared payload, but is also still "
                "an internal primitive. The hybrid row uses fused GPU scores "
                "only for shortlisting, then exact-reranks that shortlist with "
                "the GPU address scorer. The no-reference top-k comparisons "
                "keep CPU reference scores out of the Mojo serving calls and "
                "use them only for post-call validation. The positive-centroid "
                "candidate option applies only to the address-window row and "
                "is benchmark-only and explicitly opt-in. "
                "Ratios across those scopes are profiling context only, not "
                "production search speedup claims."
            ),
        },
        capability,
    )


def shape_payload(shape: SpeedTrackShape, *, candidate_k: int) -> dict[str, int]:
    return shape.to_json_ready() | {
        "candidate_k": candidate_k,
        "candidate_score_count_total": shape.query_count * candidate_k,
    }


def controls_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "normalize_vectors": args.normalize_vectors,
        "kayak_backend": args.kayak_backend,
        "kayak_plaid_centroid_count": args.kayak_plaid_centroid_count,
        "kayak_plaid_centroids_per_query_vector": (
            args.kayak_plaid_centroids_per_query_vector
        ),
        "kayak_plaid_candidate_k": args.candidate_k,
        "kayak_plaid_payload": "i8",
        "gpu_hybrid_shortlist_k": args.gpu_hybrid_shortlist_k,
        "kayak_i8_candidate_order": args.kayak_i8_candidate_order,
        "kayak_i8_positive_centroids_only": (
            args.kayak_i8_positive_centroids_only
        ),
        "fastplaid_device": args.fastplaid_device,
        "fastplaid_low_memory": args.fastplaid_low_memory,
        "fastplaid_kmeans_niters": args.fastplaid_kmeans_niters,
        "fastplaid_max_points_per_centroid": (
            args.fastplaid_max_points_per_centroid
        ),
        "fastplaid_nbits": args.fastplaid_nbits,
        "fastplaid_batch_size": args.fastplaid_batch_size,
        "fastplaid_use_triton_kmeans": _parse_optional_bool(
            args.fastplaid_use_triton_kmeans
        ),
        "fastplaid_update_buffer_size": args.fastplaid_update_buffer_size,
        "gpu_topk_session_iterations": args.gpu_topk_session_iterations,
    }


def input_bytes_payload(inputs: SyntheticInputs) -> dict[str, int]:
    return {
        "documents": int(inputs.documents.nbytes),
        "queries": int(inputs.queries.nbytes),
    }


def benchmark_system_rows(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], tuple[tuple[int, ...], ...]]:
    systems: list[dict[str, Any]] = []
    kayak_exact, reference_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend=args.kayak_backend,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    systems.append(kayak_exact)

    systems.append(
        benchmark_kayak_plaid(
            shape=shape,
            inputs=inputs,
            reference_positions_by_query=reference_positions,
            config=KayakPlaidApproxConfig(
                centroid_count=args.kayak_plaid_centroid_count,
                centroids_per_query_vector=(
                    args.kayak_plaid_centroids_per_query_vector
                ),
                candidate_k=args.candidate_k,
                payload="i8",
            ),
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
    )

    systems.append(
        benchmark_fastplaid(
            shape=shape,
            inputs=inputs,
            reference_positions_by_query=reference_positions,
            index_root=args.index_root,
            overwrite_index_root=args.overwrite_index_root,
            device=args.fastplaid_device,
            low_memory=args.fastplaid_low_memory,
            kmeans_niters=args.fastplaid_kmeans_niters,
            max_points_per_centroid=args.fastplaid_max_points_per_centroid,
            nbits=args.fastplaid_nbits,
            batch_size=args.fastplaid_batch_size,
            use_triton_kmeans=_parse_optional_bool(args.fastplaid_use_triton_kmeans),
            update_buffer_size=args.fastplaid_update_buffer_size,
            seed=args.seed,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            require_fastplaid=args.require_fastplaid,
        )
    )
    return systems, reference_positions


def build_gpu_scope_row(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
) -> dict[str, Any]:
    parsed = _parsed_payload(gpu_probe)
    return {
        "system_name": "kayak_gpu_i8_candidate_score_probe",
        "engine": "kayak_gpu_probe",
        "status": (
            gpu_probe.get("status")
            if gpu_probe is not None
            else STATUS_PARTIAL_GPU_UNAVAILABLE
        ),
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": "benchmark_only_flat_i8_dim128",
        "scope": "candidate_score_primitive_only",
        "shape": shape_payload(shape, candidate_k=candidate_k),
        "counts": gpu_probe.get("counts") if gpu_probe is not None else None,
        "parsed": parsed,
        "derived": derive_candidate_score_metrics(parsed),
    }


def build_gpu_vs_fastplaid_comparison(
    *,
    gpu_probe: dict[str, object] | None,
    fastplaid_row: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = _parsed_payload(gpu_probe)
    fastplaid_batch = _fastplaid_float(fastplaid_row, "query_batch_mean_seconds")
    fastplaid_query = _fastplaid_float(fastplaid_row, "query_mean_seconds")
    h2d = _optional_float(parsed.get("host_to_device_mean_seconds"))
    kernel = _optional_float(parsed.get("twopass_kernel_mean_seconds"))
    d2h = _optional_float(parsed.get("device_to_host_mean_seconds"))
    gpu_probe_e2e = _sum_optional(h2d, kernel, d2h)
    candidate_score_count = _optional_float(parsed.get("candidate_score_count"))

    return {
        "status": _comparison_status(gpu_probe, fastplaid_row),
        "scope": "gpu_candidate_score_primitive_vs_fastplaid_full_search",
        "scope_warning": (
            "This is not an apples-to-apples backend comparison: FastPlaid is "
            "timed as full search, while the GPU row is only Kayak's candidate "
            "score primitive before real payload integration and CPU top-k."
        ),
        "candidate_score_count_total": (
            int(candidate_score_count)
            if candidate_score_count is not None
            else None
        ),
        "gpu_twopass_kernel_mean_seconds": kernel,
        "gpu_h2d_kernel_d2h_mean_seconds": gpu_probe_e2e,
        "fastplaid_query_batch_mean_seconds": fastplaid_batch,
        "fastplaid_query_mean_seconds": fastplaid_query,
        "gpu_kernel_seconds_per_fastplaid_batch_second": _ratio(
            kernel,
            fastplaid_batch,
        ),
        "gpu_h2d_kernel_d2h_seconds_per_fastplaid_batch_second": _ratio(
            gpu_probe_e2e,
            fastplaid_batch,
        ),
        "gpu_kernel_candidate_scores_per_second": _rate(
            candidate_score_count,
            kernel,
        ),
    }


def report_status(
    *,
    capability: MojoGpuCapability,
    gpu_probe: dict[str, object] | None,
    prepared_handle_topk_row: dict[str, Any] | None,
    fused_centroid_posting_row: dict[str, Any] | None,
    hybrid_shortlist_rerank_row: dict[str, Any] | None,
    fastplaid_row: dict[str, Any] | None,
) -> str:
    if fastplaid_row is None or fastplaid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_FASTPLAID_UNAVAILABLE
    if not capability.available:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if (
        gpu_probe is None
        or gpu_probe.get("status") != GPU_CANDIDATE_SCORE_STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_CANDIDATE_SCORE_FAILED
    if (
        prepared_handle_topk_row is None
        or prepared_handle_topk_row.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED
    if prepared_handle_topk_row.get("no_reference_status") != STATUS_OK:
        return STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED
    if (
        fused_centroid_posting_row is None
        or fused_centroid_posting_row.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED
    if (
        hybrid_shortlist_rerank_row is None
        or hybrid_shortlist_rerank_row.get("status") != STATUS_OK
    ):
        return STATUS_BLOCKED_GPU_HYBRID_FAILED
    return STATUS_OK


def print_quiet_sections(report: dict[str, Any]) -> None:
    for system in report.get("systems", []):
        if not isinstance(system, dict) or system.get("status") != STATUS_OK:
            continue
        mean_seconds = system.get("query_batch_mean_seconds")
        if isinstance(mean_seconds, (float, int)):
            print("==", system.get("system_name"), "query_batch ==")
            print("Mean:", mean_seconds)

    gpu = report.get("gpu_candidate_score_primitive")
    if isinstance(gpu, dict):
        parsed = gpu.get("parsed")
        if isinstance(parsed, dict):
            kernel = parsed.get("twopass_kernel_mean_seconds")
            h2d = parsed.get("host_to_device_mean_seconds")
            d2h = parsed.get("device_to_host_mean_seconds")
            if isinstance(kernel, (float, int)):
                print("== kayak_gpu_i8_candidate_twopass_kernel ==")
                print("Mean:", kernel)
            gpu_probe_e2e = _sum_optional(
                _optional_float(h2d),
                _optional_float(kernel),
                _optional_float(d2h),
            )
            if gpu_probe_e2e is not None:
                print("== kayak_gpu_i8_candidate_h2d_kernel_d2h ==")
                print("Mean:", gpu_probe_e2e)
    topk = report.get("gpu_prepared_handle_topk_primitive")
    if isinstance(topk, dict):
        parsed = topk.get("parsed")
        if isinstance(parsed, dict):
            topk_per_window = parsed.get("score_extension_call_seconds_per_window")
            if isinstance(topk_per_window, (float, int)):
                print("== kayak_gpu_i8_prepared_handle_topk_per_window ==")
                print("Mean:", topk_per_window)
        no_reference_parsed = topk.get("no_reference_parsed")
        if isinstance(no_reference_parsed, dict):
            no_ref = no_reference_parsed.get(
                "score_extension_call_seconds_per_window"
            )
            if isinstance(no_ref, (float, int)):
                print("== kayak_gpu_i8_prepared_handle_topk_no_reference ==")
                print("Mean:", no_ref)
        comparison = report.get(
            "gpu_prepared_handle_topk_vs_fastplaid_scope_comparison"
        )
        if isinstance(comparison, dict):
            envelope = comparison.get(
                "cpu_candidate_generation_plus_gpu_topk_seconds_per_window"
            )
            if isinstance(envelope, (float, int)):
                print("== kayak_cpu_candidates_gpu_i8_topk_per_window ==")
                print("Mean:", envelope)
        no_reference_comparison = report.get(
            "gpu_prepared_handle_topk_no_reference_vs_fastplaid_scope_comparison"
        )
        if isinstance(no_reference_comparison, dict):
            no_reference_envelope = no_reference_comparison.get(
                "cpu_candidate_generation_plus_gpu_topk_seconds_per_window"
            )
            if isinstance(no_reference_envelope, (float, int)):
                print(
                    "== kayak_cpu_candidates_gpu_i8_topk_no_reference_per_window =="
                )
                print("Mean:", no_reference_envelope)
    fused = report.get("gpu_fused_centroid_posting_topk_primitive")
    if isinstance(fused, dict):
        parsed = fused.get("parsed")
        if isinstance(parsed, dict):
            host_topk = parsed.get("score_extension_call_seconds_per_window")
            device_topk = parsed.get(
                "device_topk_score_extension_call_seconds_per_window"
            )
            if isinstance(host_topk, (float, int)):
                print("== kayak_gpu_i8_fused_host_topk_per_window ==")
                print("Mean:", host_topk)
            if isinstance(device_topk, (float, int)):
                print("== kayak_gpu_i8_fused_device_topk_per_window ==")
                print("Mean:", device_topk)
    fused_comparison = report.get(
        "gpu_fused_centroid_posting_vs_fastplaid_scope_comparison"
    )
    if isinstance(fused_comparison, dict):
        ratio = fused_comparison.get(
            "gpu_fused_device_topk_seconds_per_fastplaid_batch_second"
        )
        if isinstance(ratio, (float, int)):
            print("== kayak_gpu_i8_fused_device_topk_per_fastplaid_batch ==")
            print("Mean:", ratio)
    hybrid = report.get("gpu_hybrid_shortlist_exact_rerank_primitive")
    if isinstance(hybrid, dict):
        parsed = hybrid.get("parsed")
        if isinstance(parsed, dict):
            hybrid_seconds = parsed.get("hybrid_extension_seconds_per_window")
            fused_seconds = parsed.get("fused_device_topk_seconds_per_window")
            exact_seconds = parsed.get("exact_rerank_topk_seconds_per_window")
            if isinstance(hybrid_seconds, (float, int)):
                print("== kayak_gpu_i8_hybrid_per_window ==")
                print("Mean:", hybrid_seconds)
            if isinstance(fused_seconds, (float, int)):
                print("== kayak_gpu_i8_hybrid_fused_shortlist_per_window ==")
                print("Mean:", fused_seconds)
            if isinstance(exact_seconds, (float, int)):
                print("== kayak_gpu_i8_hybrid_exact_rerank_per_window ==")
                print("Mean:", exact_seconds)
    hybrid_comparison = report.get(
        "gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_scope_comparison"
    )
    if isinstance(hybrid_comparison, dict):
        ratio = hybrid_comparison.get(
            "gpu_hybrid_seconds_per_fastplaid_batch_second"
        )
        if isinstance(ratio, (float, int)):
            print("== kayak_gpu_i8_hybrid_per_fastplaid_batch ==")
            print("Mean:", ratio)


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
    if status == STATUS_BLOCKED_FASTPLAID_UNAVAILABLE and not args.require_fastplaid:
        return 0
    if not capability.available and not args.allow_missing_gpu:
        return 2
    if status == STATUS_BLOCKED_GPU_CANDIDATE_SCORE_FAILED:
        return 3
    if status == STATUS_BLOCKED_FASTPLAID_UNAVAILABLE:
        return 4
    if status == STATUS_BLOCKED_GPU_PREPARED_TOPK_FAILED:
        return 6
    if status == STATUS_BLOCKED_GPU_FUSED_HANDLE_FAILED:
        return 7
    if status == STATUS_BLOCKED_GPU_HYBRID_FAILED:
        return 8
    return 5


def _comparison_status(
    gpu_probe: dict[str, object] | None,
    fastplaid_row: dict[str, Any] | None,
) -> str:
    if fastplaid_row is None or fastplaid_row.get("status") != STATUS_OK:
        return STATUS_BLOCKED_FASTPLAID_UNAVAILABLE
    if gpu_probe is None:
        return STATUS_PARTIAL_GPU_UNAVAILABLE
    if gpu_probe.get("status") != GPU_CANDIDATE_SCORE_STATUS_OK:
        return STATUS_BLOCKED_GPU_CANDIDATE_SCORE_FAILED
    return STATUS_OK


def _system_by_name(
    systems: Sequence[dict[str, Any]],
    system_name: str,
) -> dict[str, Any] | None:
    return next(
        (
            system
            for system in systems
            if system.get("system_name") == system_name
        ),
        None,
    )


def _parsed_payload(gpu_probe: dict[str, object] | None) -> dict[str, object]:
    parsed = None if gpu_probe is None else gpu_probe.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def _fastplaid_float(
    fastplaid_row: dict[str, Any] | None,
    key: str,
) -> float | None:
    return _optional_float(
        None if fastplaid_row is None else fastplaid_row.get(key)
    )


def _parse_optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError("expected auto, true, or false")


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _sum_optional(*values: float | None) -> float | None:
    if any(value is None for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _rate(count: float | None, seconds: float | None) -> float | None:
    if count is None or seconds is None or seconds <= 0.0:
        return None
    return count / seconds


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
