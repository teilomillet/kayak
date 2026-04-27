from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    CASE_SETS,
    STATUS_OK,
    STATUS_PARTIAL_GPU_UNAVAILABLE,
    AddressServeSweepCase,
    AddressServeSweepControls,
    case_set_names,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_centroid_selection import (  # noqa: E402
    STATUS_BLOCKED_GPU_CENTROID_SELECTION_FAILED,
    build_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile GPU scoring of i8 sampled centroids plus host centroid "
            "top-k selection."
        )
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        help=(
            "Repeatable case spec: "
            "name:documents=512,document_vectors=16,queries=2,"
            "query_vectors=32,candidate_k=256"
        ),
    )
    parser.add_argument(
        "--case-set",
        choices=case_set_names(),
        default="wide_topk",
    )
    parser.add_argument("--vector-dim", type=int, default=128)
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
        "--centroid-budget-policy",
        default="shape_rule_v0",
        help=(
            "Optional benchmark-only centroid budget policy. Use 'none' to "
            "force the static --kayak-plaid-centroids-per-query-vector value."
        ),
    )
    parser.add_argument(
        "--gpu-query-command",
        default="gpu-query",
        help="Executable used to probe Mojo-visible GPU devices.",
    )
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Exit successfully when no Mojo GPU is visible; report stays partial.",
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_centroid_selection/summary.json"),
    )
    return parser.parse_args(argv)


def cases_from_args(args: argparse.Namespace) -> tuple[AddressServeSweepCase, ...]:
    return tuple(args.case) if args.case else CASE_SETS[args.case_set]


def controls_from_args(args: argparse.Namespace) -> AddressServeSweepControls:
    return AddressServeSweepControls(
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        kayak_plaid_centroids_per_query_vector=(
            args.kayak_plaid_centroids_per_query_vector
        ),
        gpu_query_command=args.gpu_query_command,
    )


def centroid_budget_policy_from_args(args: argparse.Namespace) -> str | None:
    policy = str(args.centroid_budget_policy)
    if policy.lower() == "none":
        return None
    return policy


def case_selection_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.case:
        return {"source": "custom", "case_count": len(args.case)}
    return {"source": args.case_set, "case_count": len(CASE_SETS[args.case_set])}


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
            f"gpu_i8_centroid_selection_{name}_cpu_centroid_scoring_plus_selection",
            comparison.get("cpu_i8_centroid_scoring_plus_selection_seconds"),
        )
        print_quiet_mean(
            f"gpu_i8_centroid_selection_{name}_kernel",
            comparison.get("gpu_i8_centroid_selection_kernel_mean_seconds"),
        )
        print_quiet_mean(
            f"gpu_i8_centroid_selection_{name}_resident_payload",
            comparison.get(
                "gpu_i8_centroid_selection_resident_payload_mean_seconds"
            ),
        )
        print_quiet_mean(
            f"gpu_i8_centroid_selection_{name}_cold_payload",
            comparison.get("gpu_i8_centroid_selection_cold_payload_mean_seconds"),
        )


def print_quiet_mean(section: str, value: object) -> None:
    if isinstance(value, (float, int)):
        print(f"== {section} ==")
        print("Mean:", value)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def exit_code(report: dict[str, Any], *, gpu_available: bool, allow_missing: bool) -> int:
    status = report.get("status")
    if status == STATUS_OK:
        return 0
    if status == STATUS_PARTIAL_GPU_UNAVAILABLE and allow_missing:
        return 0
    if not gpu_available and not allow_missing:
        return 2
    if status == STATUS_BLOCKED_GPU_CENTROID_SELECTION_FAILED:
        return 3
    return 4


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, capability = build_report(
        cases=cases_from_args(args),
        controls=controls_from_args(args),
        centroid_budget_policy=centroid_budget_policy_from_args(args),
    )
    report["case_selection"] = case_selection_from_args(args)
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
