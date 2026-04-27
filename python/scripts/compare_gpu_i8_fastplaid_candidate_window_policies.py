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

import compare_gpu_i8_fastplaid_policy as single_policy  # noqa: E402
from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    STATUS_OK,
    case_set_names,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_candidate_window_policy import (  # noqa: E402
    INPUT_CANDIDATE_K_POLICY,
)
from kayak_bridge.gpu_i8_candidate_window_policy_matrix import (  # noqa: E402
    summarize_policy_matrix,
)
from kayak_bridge.gpu_i8_candidate_window_policy_matrix_runner import (  # noqa: E402
    DEFAULT_CANDIDATE_WINDOW_POLICIES,
    DEFAULT_CASE_SOURCE,
    candidate_window_policies,
    case_sources,
    controls_payload,
    policy_summary_path,
    report_status,
    single_policy_argv,
)
from kayak_bridge.gpu_i8_fastplaid_policy_compare import (  # noqa: E402
    parse_fastplaid_devices,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GPU i8 FastPlaid comparisons for multiple candidate-window "
            "policies on the same explicit dim128 shape families."
        )
    )
    parser.add_argument(
        "--case-set",
        action="append",
        choices=case_set_names(),
        default=None,
        help=f"Case set to run. Defaults to {DEFAULT_CASE_SOURCE}.",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        default=None,
        help="Explicit name:key=value,key=value case. Overrides --case-set.",
    )
    parser.add_argument("--include-full-window", action="store_true")
    parser.add_argument(
        "--candidate-window-policy",
        action="append",
        default=None,
        help=(
            "Candidate-window policy to run. Defaults to "
            f"{', '.join(DEFAULT_CANDIDATE_WINDOW_POLICIES)}."
        ),
    )
    parser.add_argument(
        "--baseline-candidate-window-policy",
        default=INPUT_CANDIDATE_K_POLICY,
    )
    add_single_policy_controls(parser)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path(
            ".cache/kayak/gpu_i8_fastplaid_candidate_window_policies/reports"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".cache/kayak/gpu_i8_fastplaid_candidate_window_policies/summary.json"
        ),
    )
    return parser.parse_args(argv)


def add_single_policy_controls(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--fixed-case-seed",
        action="store_true",
        help="Use the same seed for every case in each policy report.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=1)
    parser.add_argument("--gpu-topk-session-iterations", type=int, default=4)
    parser.add_argument("--gpu-hybrid-shortlist-k", type=int, default=None)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--kayak-i8-candidate-order",
        choices=("ordered", "unordered"),
        default="unordered",
    )
    parser.add_argument("--kayak-i8-positive-centroids-only", action="store_true")
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


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    refs: list[dict[str, object]] = []
    for source_name, cases in case_sources(args):
        for policy_name in candidate_window_policies(args):
            path = policy_summary_path(
                args.report_root,
                case_source=source_name,
                candidate_window_policy=policy_name,
            )
            report = run_single_policy(
                args,
                case_source=source_name,
                cases=cases,
                candidate_window_policy=policy_name,
                output=path,
            )
            reports.append(report)
            refs.append(
                {
                    "case_source": source_name,
                    "candidate_window_policy": policy_name,
                    "status": report.get("status"),
                    "path": str(path),
                    "row_count": len(report.get("rows", [])),
                }
            )
    return {
        "schema_version": 1,
        "benchmark": "gpu_i8_fastplaid_candidate_window_policies",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": report_status(reports),
        "case_sources": [
            {"name": source_name, "case_count": len(cases)}
            for source_name, cases in case_sources(args)
        ],
        "candidate_window_policies": list(candidate_window_policies(args)),
        "controls": controls_payload(args),
        "policy_reports": refs,
        "matrix_summary": summarize_policy_matrix(
            reports,
            baseline_policy=args.baseline_candidate_window_policy,
        ),
        "measurement_note": (
            "This benchmark-only matrix compares our resident selected-posting "
            "exact-rerank row against FastPlaid on explicit dim128 synthetic "
            "shapes and does not change public search defaults."
        ),
    }


def run_single_policy(
    args: argparse.Namespace,
    *,
    case_source: str,
    cases: Sequence[object],
    candidate_window_policy: str,
    output: Path,
) -> dict[str, Any]:
    argv = single_policy_argv(
        args,
        case_source=case_source,
        cases=cases,
        candidate_window_policy=candidate_window_policy,
        output=output,
    )
    policy_args = single_policy.parse_args(argv)
    report = single_policy.build_report(policy_args)
    single_policy.write_report(report, output)
    return report


def emit_quiet_means(report: dict[str, Any]) -> None:
    matrix = report["matrix_summary"]
    for policy_summary in matrix["policy_summaries"]:
        prefix = (
            "gpu_i8_candidate_window_policy_matrix_"
            f"{policy_summary['case_source']}_"
            f"{policy_summary['candidate_window_policy']}"
        )
        print_quiet_mean(
            f"{prefix}_resident_recall_delta_min",
            policy_summary[
                "min_gpu_resident_selected_recall_delta_vs_fastplaid"
            ],
        )
        print_quiet_mean(
            f"{prefix}_resident_fastplaid_ratio_mean",
            policy_summary[
                "mean_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
            ],
        )
        print_quiet_mean(
            f"{prefix}_resident_fastplaid_ratio_max",
            policy_summary[
                "max_gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
            ],
        )
    comparison = matrix["baseline_comparison_summary"]
    print_quiet_mean(
        "gpu_i8_candidate_window_policy_matrix_recall_delta_vs_baseline_min",
        comparison["min_gpu_resident_selected_recall_delta_vs_baseline"],
    )
    print_quiet_mean(
        "gpu_i8_candidate_window_policy_matrix_fastplaid_ratio_vs_baseline_max",
        comparison["max_gpu_resident_selected_fastplaid_ratio_vs_baseline"],
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
