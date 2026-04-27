from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
for root in (PYTHON_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.append(str(root))

import compare_gpu_i8_fastplaid as fastplaid_compare  # noqa: E402
from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    CASE_SETS,
    STATUS_OK,
    AddressServeSweepCase,
    case_set_names,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_centroid_budget_sweep import (  # noqa: E402
    non_full_candidate_cases,
)
from kayak_bridge.gpu_i8_fastplaid_policy_compare import (  # noqa: E402
    FastPlaidPolicyCompareControls,
    compare_argv_for_case,
    parse_fastplaid_devices,
    report_status,
    summarize_case_device_report,
    summary_payload,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a shape-only GPU i8 centroid-budget policy against "
            "FastPlaid CPU/CUDA on explicit dim128 synthetic cases."
        )
    )
    parser.add_argument("--case-set", choices=case_set_names(), default="wide_topk")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        default=None,
        help="Explicit name:key=value,key=value case. Overrides --case-set.",
    )
    parser.add_argument("--include-full-window", action="store_true")
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=1)
    parser.add_argument("--gpu-topk-session-iterations", type=int, default=4)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--kayak-i8-candidate-order",
        choices=("ordered", "unordered"),
        default="unordered",
    )
    parser.add_argument(
        "--kayak-i8-positive-centroids-only",
        action="store_true",
        help=(
            "Benchmark-only: use unordered candidate windows built from "
            "positive selected centroid postings only."
        ),
    )
    parser.add_argument("--policy-name", default="shape_rule_v0")
    parser.add_argument(
        "--fastplaid-devices",
        type=parse_fastplaid_devices,
        default=("cpu", "cuda"),
    )
    parser.add_argument(
        "--require-fastplaid",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--allow-missing-gpu", action="store_true")
    parser.add_argument("--overwrite-index-root", action="store_true")
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_fastplaid_policy_compare/reports"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json"),
    )
    return parser.parse_args(argv)


def selected_cases(args: argparse.Namespace) -> tuple[AddressServeSweepCase, ...]:
    cases = tuple(args.case) if args.case else CASE_SETS[args.case_set]
    return non_full_candidate_cases(
        cases,
        include_full_window=args.include_full_window,
    )


def controls_from_args(args: argparse.Namespace) -> FastPlaidPolicyCompareControls:
    controls = FastPlaidPolicyCompareControls(
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        gpu_topk_session_iterations=args.gpu_topk_session_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        kayak_i8_candidate_order=args.kayak_i8_candidate_order,
        kayak_i8_positive_centroids_only=(
            args.kayak_i8_positive_centroids_only
        ),
        policy_name=args.policy_name,
        fastplaid_devices=tuple(args.fastplaid_devices),
    )
    controls.validate()
    return controls


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cases = selected_cases(args)
    controls = controls_from_args(args)
    rows = [
        run_case_device(
            case=case,
            case_index=case_index,
            controls=controls,
            fastplaid_device=fastplaid_device,
            args=args,
        )
        for case_index, case in enumerate(cases)
        for fastplaid_device in controls.fastplaid_devices
    ]
    return {
        "schema_version": 1,
        "benchmark": "gpu_i8_fastplaid_policy_compare",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": report_status(rows),
        "case_selection": {
            "source": "explicit" if args.case else args.case_set,
            "case_count": len(cases),
            "include_full_window": bool(args.include_full_window),
        },
        "controls": controls.to_json_ready(),
        "rows": rows,
        "summary": summary_payload(rows),
        "measurement_note": (
            "FastPlaid rows are full-search timings. Kayak policy rows use a "
            "shape-only centroid-budget policy, CPU candidate generation, and "
            "the GPU no-reference top-k primitive. The report also includes "
            "the fused centroid-posting GPU primitive as a separate internal "
            "scope. This is not a public backend speedup claim."
        ),
    }


def run_case_device(
    *,
    case: AddressServeSweepCase,
    case_index: int,
    controls: FastPlaidPolicyCompareControls,
    fastplaid_device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report_path = row_report_path(
        root=args.report_root,
        case=case,
        device=fastplaid_device,
        policy_name=controls.policy_name,
        seed=controls.seed + case_index,
    )
    compare_args = fastplaid_compare.parse_args(
        compare_argv_for_case(
            case=case,
            case_index=case_index,
            controls=controls,
            fastplaid_device=fastplaid_device,
            index_root=row_index_root(args.report_root, case, fastplaid_device),
            output=report_path,
            allow_missing_gpu=args.allow_missing_gpu,
            require_fastplaid=args.require_fastplaid,
            overwrite_index_root=args.overwrite_index_root,
        )
    )
    report, _capability = fastplaid_compare.build_report(compare_args)
    fastplaid_compare.write_report(report_path, report)
    return summarize_case_device_report(
        case=case,
        case_index=case_index,
        controls=controls,
        fastplaid_device=fastplaid_device,
        report=report,
        report_path=report_path,
    )


def row_report_path(
    *,
    root: Path,
    case: AddressServeSweepCase,
    device: str,
    policy_name: str,
    seed: int,
) -> Path:
    safe_device = device.replace("/", "_").replace(":", "_")
    return root / f"{case.name}_{policy_name}_{safe_device}_seed{seed}.json"


def row_index_root(
    root: Path,
    case: AddressServeSweepCase,
    device: str,
) -> Path:
    safe_device = device.replace("/", "_").replace(":", "_")
    return root / "indexes" / f"{case.name}_{safe_device}"


def emit_quiet_means(report: dict[str, Any]) -> None:
    for row in report["rows"]:
        if row.get("status") != STATUS_OK:
            continue
        prefix = (
            "gpu_i8_fastplaid_policy_"
            f"{row['name']}_{row['fastplaid_device']}"
        )
        print_quiet_mean(
            f"{prefix}_envelope_per_window",
            row[
                "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_window"
            ],
        )
        print_quiet_mean(
            f"{prefix}_cpu_candidate_generation_per_window",
            row["cpu_candidate_generation_seconds_per_window"],
        )
        print_quiet_mean(
            f"{prefix}_gpu_topk_no_reference_per_window",
            row["gpu_topk_no_reference_seconds_per_window"],
        )
        print_quiet_mean(
            f"{prefix}_gpu_fused_device_topk_per_window",
            row["gpu_fused_device_topk_seconds_per_window"],
        )
        print_quiet_mean(
            f"{prefix}_envelope_per_fastplaid_batch",
            row[
                "cpu_candidate_generation_plus_gpu_topk_no_reference_seconds_per_fastplaid_batch_second"
            ],
        )
        print_quiet_mean(
            f"{prefix}_gpu_fused_device_topk_per_fastplaid_batch",
            row["gpu_fused_device_topk_seconds_per_fastplaid_batch_second"],
        )
        print_quiet_mean(
            f"{prefix}_kayak_recall",
            row["kayak_i8_recall_at_k_vs_kayak_exact"],
        )
        print_quiet_mean(
            f"{prefix}_fastplaid_recall",
            row["fastplaid_recall_at_k_vs_kayak_exact"],
        )
        print_quiet_mean(
            f"{prefix}_gpu_fused_recall",
            row["gpu_fused_recall_at_k_vs_kayak_exact"],
        )


def print_quiet_mean(section: str, value: object) -> None:
    if isinstance(value, (float, int)):
        print(f"== {section} ==")
        print("Mean:", value)


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        emit_quiet_means(report)
    return 0 if report["status"] == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
