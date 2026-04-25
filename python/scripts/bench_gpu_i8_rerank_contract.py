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
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from kayak_bridge.gpu_device_capability import (
    MojoGpuCapability,
    host_gpu_inventory,
    probe_mojo_gpu,
    run_command,
)
from kayak_bridge.gpu_copy_roundtrip import (
    GPU_COPY_PROBE_STATUS_TARGET_MISSING,
    run_gpu_copy_roundtrip_probe,
)
from kayak_bridge.gpu_i8_candidate_score import (
    GPU_CANDIDATE_SCORE_STATUS_TARGET_MISSING,
    run_gpu_i8_candidate_score_probe,
)
from kayak_bridge.gpu_i8_rerank_contract import (
    BENCHMARK_NAME,
    exit_code_for_report,
    gpu_timing_contract,
    print_quiet_wrapper_section,
    report_status,
    shape_contract,
    tensor_contract,
    write_report,
)
from kayak_bridge.gpu_i8_profile_contract import (
    gpu_profile_contract,
    gpu_profile_row_template,
)
from kayak_bridge.gpu_i8_single_score import (
    GPU_SINGLE_SCORE_STATUS_TARGET_MISSING,
    run_gpu_i8_single_score_probe,
)

from bench_fastplaid_speed_track import (
    SpeedTrackShape,
)
from gpu_i8_cpu_reference import (
    cpu_reference_rows,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Mojo GPU readiness and write the i8 candidate-rerank "
            "benchmark contract. This does not implement a GPU kernel."
        )
    )
    parser.add_argument("--document-count", type=int, default=64)
    parser.add_argument("--document-vector-count", type=int, default=16)
    parser.add_argument("--query-count", type=int, default=2)
    parser.add_argument("--query-vector-count", type=int, default=8)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=32)
    parser.add_argument("--centroid-count", type=int, default=64)
    parser.add_argument("--centroids-per-query-vector", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=1)
    parser.add_argument(
        "--skip-cpu-reference",
        action="store_true",
        help="Only write the GPU capability and measurement contract.",
    )
    parser.add_argument(
        "--skip-gpu-copy-probe",
        action="store_true",
        help="Skip the Mojo DeviceBuffer allocation/copy roundtrip probe.",
    )
    parser.add_argument(
        "--skip-gpu-single-score-probe",
        action="store_true",
        help="Skip the one-document Mojo GPU i8 score-agreement probe.",
    )
    parser.add_argument(
        "--skip-gpu-candidate-score-probe",
        action="store_true",
        help="Skip the batched candidate-score Mojo GPU agreement probe.",
    )
    parser.add_argument(
        "--gpu-query-command",
        default="gpu-query",
        help="Executable used to probe Mojo-visible GPU devices.",
    )
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Exit successfully when no Mojo GPU is visible; the report stays blocked.",
    )
    parser.add_argument(
        "--allow-missing-kernel",
        action="store_true",
        help="Exit successfully while the GPU rerank kernel is intentionally absent.",
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print a run_bench_quiet-compatible CPU i8 Mean section after JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_rerank_contract/summary.json"),
    )
    return parser.parse_args(argv)


def build_shape(args: argparse.Namespace) -> SpeedTrackShape:
    return SpeedTrackShape(
        document_count=args.document_count,
        document_vector_count=args.document_vector_count,
        query_count=args.query_count,
        query_vector_count=args.query_vector_count,
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        update_document_count=0,
    )


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], MojoGpuCapability]:
    shape = build_shape(args)
    shape.validate()
    if args.vector_dim != 128:
        raise ValueError("GPU i8 rerank contract is currently dim128-only")
    if args.candidate_k < args.top_k:
        raise ValueError("candidate_k must be greater than or equal to top_k")

    capability = probe_mojo_gpu(args.gpu_query_command)
    inventory = host_gpu_inventory()
    if args.skip_gpu_copy_probe:
        copy_probe: dict[str, object] = {
            "status": "skipped",
            "reason": "--skip-gpu-copy-probe",
        }
    elif capability.available:
        copy_probe = run_gpu_copy_roundtrip_probe(
            shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
        )
    else:
        copy_probe = {
            "status": GPU_COPY_PROBE_STATUS_TARGET_MISSING,
            "reason": "Mojo GPU capability is not available",
        }
    copy_probe_status = str(copy_probe.get("status"))
    copy_probe_gate_status = (
        None if copy_probe_status == "skipped" else copy_probe_status
    )
    if args.skip_gpu_single_score_probe:
        single_score_probe: dict[str, object] = {
            "status": "skipped",
            "reason": "--skip-gpu-single-score-probe",
        }
    elif args.skip_gpu_copy_probe:
        single_score_probe = {
            "status": "skipped",
            "reason": "single-score probe skipped because copy probe was skipped",
        }
    elif capability.available and copy_probe_gate_status == "ok":
        single_score_probe = run_gpu_i8_single_score_probe(
            shape,
            target_accelerator=capability.target_accelerator,
        )
    else:
        single_score_probe = {
            "status": GPU_SINGLE_SCORE_STATUS_TARGET_MISSING,
            "reason": "Mojo GPU capability or copy probe is not available",
        }
    single_score_probe_status = str(single_score_probe.get("status"))
    single_score_gate_status = (
        None if single_score_probe_status == "skipped" else single_score_probe_status
    )
    if args.skip_gpu_candidate_score_probe:
        candidate_score_probe: dict[str, object] = {
            "status": "skipped",
            "reason": "--skip-gpu-candidate-score-probe",
        }
    elif args.skip_gpu_copy_probe or args.skip_gpu_single_score_probe:
        candidate_score_probe = {
            "status": "skipped",
            "reason": (
                "candidate-score probe skipped because an earlier GPU probe "
                "was skipped"
            ),
        }
    elif (
        capability.available
        and copy_probe_gate_status == "ok"
        and single_score_gate_status == "ok"
    ):
        candidate_score_probe = run_gpu_i8_candidate_score_probe(
            shape,
            candidate_k=args.candidate_k,
            target_accelerator=capability.target_accelerator,
        )
    else:
        candidate_score_probe = {
            "status": GPU_CANDIDATE_SCORE_STATUS_TARGET_MISSING,
            "reason": (
                "Mojo GPU capability, copy probe, or one-document score "
                "probe is not available"
            ),
        }
    candidate_score_probe_status = str(candidate_score_probe.get("status"))
    candidate_score_gate_status = (
        None
        if candidate_score_probe_status == "skipped"
        else candidate_score_probe_status
    )
    report: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "created_at": datetime.now(UTC).isoformat(),
        "status": report_status(
            capability,
            inventory,
            copy_probe_status=copy_probe_gate_status,
            single_score_probe_status=single_score_gate_status,
            candidate_score_probe_status=candidate_score_gate_status,
        ),
        "mojo_toolchain": run_command(("mojo", "--version")).to_json_ready(),
        "host_gpu_inventory": inventory,
        "mojo_gpu_capability": capability.to_json_ready(),
        "gpu_copy_roundtrip_probe": copy_probe,
        "gpu_i8_single_doc_score_probe": single_score_probe,
        "gpu_i8_candidate_score_probe": candidate_score_probe,
        "shape": shape_contract(shape, candidate_k=args.candidate_k),
        "tensor_contract": tensor_contract(shape, candidate_k=args.candidate_k),
        "gpu_timing_contract": gpu_timing_contract(),
        "gpu_profile_contract": gpu_profile_contract(
            shape,
            candidate_k=args.candidate_k,
        ),
        "gpu_profile_row_template": gpu_profile_row_template(
            shape,
            candidate_k=args.candidate_k,
        ),
        "comparison_contract": {
            "correctness_reference": "CPU i8 score for the same candidates",
            "reported_agreement": [
                "score_delta_max_abs",
                "candidate_order_agreement",
                "recall_at_k_vs_cpu_i8",
                "recall_at_k_vs_kayak_exact",
            ],
            "current_status": (
                "CPU same-candidate reference measured when not skipped; "
                "one-document GPU score agreement measured when available; "
                "batched benchmark-only GPU candidate scores measured when "
                "available; production backend integration is not implemented"
            ),
        },
        "boundary": {
            "primitive": "plaid_i8_gpu_rerank_candidates_dim128",
            "public_api": "none",
            "cpu_fallback": "forbidden for GPU rows",
            "top_k": "CPU top-k may consume GPU candidate scores first",
        },
    }
    if args.skip_cpu_reference:
        report["cpu_reference"] = {
            "status": "skipped",
            "reason": "--skip-cpu-reference",
        }
    else:
        report["cpu_reference"] = cpu_reference_rows(
            shape=shape,
            candidate_k=args.candidate_k,
            centroid_count=args.centroid_count,
            centroids_per_query_vector=args.centroids_per_query_vector,
            seed=args.seed,
            normalize_vectors=args.normalize_vectors,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
    return report, capability


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, capability = build_report(args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        print_quiet_wrapper_section(report)
    return exit_code_for_report(
        capability=capability,
        allow_missing_gpu=args.allow_missing_gpu,
        allow_missing_kernel=args.allow_missing_kernel,
        copy_probe_status=(
            None
            if report["gpu_copy_roundtrip_probe"].get("status") == "skipped"
            else str(report["gpu_copy_roundtrip_probe"].get("status"))
        ),
        single_score_probe_status=(
            None
            if report["gpu_i8_single_doc_score_probe"].get("status") == "skipped"
            else str(report["gpu_i8_single_doc_score_probe"].get("status"))
        ),
        candidate_score_probe_status=(
            None
            if report["gpu_i8_candidate_score_probe"].get("status") == "skipped"
            else str(report["gpu_i8_candidate_score_probe"].get("status"))
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
