from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    DEFAULT_CASES,
    STATUS_BLOCKED_GPU_ADDRESS_SERVE_FAILED,
    AddressServeSweepCase,
    AddressServeSweepControls,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_address_serve_sweep_runner import build_report  # noqa: E402
from profile_gpu_i8_real_payload_rerank import (  # noqa: E402
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the no-internal-benchmark GPU i8 typed-address serving "
            "call across explicit vector-count shapes."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        help=(
            "Repeatable case spec: "
            "name:documents=256,document_vectors=16,queries=2,"
            "query_vectors=8,candidate_k=128"
        ),
    )
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--resident-session-iterations", type=int, default=4)
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
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_address_serve_sweep/summary.json"),
    )
    return parser.parse_args(argv)


def cases_from_args(args: argparse.Namespace) -> tuple[AddressServeSweepCase, ...]:
    return tuple(args.case) if args.case else DEFAULT_CASES


def controls_from_args(args: argparse.Namespace) -> AddressServeSweepControls:
    return AddressServeSweepControls(
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        resident_session_iterations=args.resident_session_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        kayak_plaid_centroids_per_query_vector=(
            args.kayak_plaid_centroids_per_query_vector
        ),
        gpu_query_command=args.gpu_query_command,
    )


def print_quiet_sections(report: dict[str, Any]) -> None:
    rows = report.get("cases")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        comparison = row.get("comparison")
        if not isinstance(name, str) or not isinstance(comparison, dict):
            continue
        print_quiet_mean(
            f"gpu_i8_address_serve_sweep_{name}_cpu_score",
            comparison.get("cpu_i8_same_candidate_score_mean_seconds"),
        )
        print_quiet_mean(
            f"gpu_i8_address_serve_sweep_{name}_gpu_extension_call",
            comparison.get("gpu_address_serve_extension_call_seconds"),
        )
        print_quiet_mean(
            f"gpu_i8_address_serve_sweep_{name}_cpu_candidate_plus_gpu",
            comparison.get("cpu_candidate_generation_plus_gpu_address_serve_seconds"),
        )
        print_quiet_mean(
            f"gpu_i8_address_serve_sweep_{name}_resident_iteration",
            comparison.get(
                "gpu_address_resident_session_extension_call_seconds_per_iteration"
            ),
        )
        print_quiet_mean(
            f"gpu_i8_address_serve_sweep_{name}_resident_multi_window",
            comparison.get(
                "gpu_address_resident_multi_window_extension_call_seconds_per_window"
            ),
        )


def print_quiet_mean(section: str, value: object) -> None:
    if isinstance(value, (float, int)):
        print(f"== {section} ==")
        print("Mean:", value)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def exit_code(report: dict[str, Any], *, gpu_available: bool, allow_missing: bool) -> int:
    status = report.get("status")
    if status == STATUS_OK:
        return 0
    if status == STATUS_PARTIAL_GPU_UNAVAILABLE and allow_missing:
        return 0
    if not gpu_available and not allow_missing:
        return 2
    if status == STATUS_BLOCKED_GPU_ADDRESS_SERVE_FAILED:
        return 3
    return 4


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, capability = build_report(
        cases=cases_from_args(args),
        controls=controls_from_args(args),
    )
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        print_quiet_sections(report)
    return exit_code(
        report,
        gpu_available=capability.available,
        allow_missing=args.allow_missing_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
