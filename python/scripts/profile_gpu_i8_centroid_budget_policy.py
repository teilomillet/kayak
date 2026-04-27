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
    CASE_SETS,
    STATUS_OK,
    AddressServeSweepCase,
    case_set_names,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_centroid_budget_policy import (  # noqa: E402
    DEFAULT_POLICY_NAMES,
    parse_policy_names,
)
from kayak_bridge.gpu_i8_centroid_budget_policy_runner import (  # noqa: E402
    build_report,
)
from kayak_bridge.gpu_i8_centroid_budget_sweep import (  # noqa: E402
    DEFAULT_CENTROID_BUDGETS,
    CentroidBudgetSweepControls,
    non_full_candidate_cases,
    parse_centroid_budgets,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay centroid-budget policies over measured CPU i8 candidate "
            "generation rows for the GPU top-k pipeline."
        )
    )
    parser.add_argument(
        "--case-set",
        choices=case_set_names(),
        default="wide_topk",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        default=None,
        help="Explicit name:key=value,key=value case. Overrides --case-set.",
    )
    parser.add_argument(
        "--include-full-window",
        action="store_true",
        help="Also include cases where candidate_k >= document_count.",
    )
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--centroid-budgets",
        type=parse_centroid_budgets,
        default=DEFAULT_CENTROID_BUDGETS,
        help="Comma-separated swept centroids-per-query-vector budgets.",
    )
    parser.add_argument(
        "--baseline-centroids-per-query-vector",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--policies",
        type=parse_policy_names,
        default=DEFAULT_POLICY_NAMES,
        help="Comma-separated policy names, e.g. static4,shape_rule_v0.",
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_centroid_budget_policy/summary.json"),
    )
    return parser.parse_args(argv)


def selected_cases(args: argparse.Namespace) -> tuple[AddressServeSweepCase, ...]:
    cases = tuple(args.case) if args.case else CASE_SETS[args.case_set]
    return non_full_candidate_cases(
        cases,
        include_full_window=args.include_full_window,
    )


def controls_from_args(args: argparse.Namespace) -> CentroidBudgetSweepControls:
    controls = CentroidBudgetSweepControls(
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        centroid_budgets=tuple(args.centroid_budgets),
        baseline_centroids_per_query_vector=(
            args.baseline_centroids_per_query_vector
        ),
    )
    controls.validate()
    return controls


def case_selection_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source": "explicit" if args.case else args.case_set,
        "case_count": len(selected_cases(args)),
        "include_full_window": bool(args.include_full_window),
    }


def emit_quiet_means(report: dict[str, Any]) -> None:
    for row in report["policy_cases"]:
        for policy_row in row["policy_rows"]:
            if policy_row.get("status") != STATUS_OK:
                continue
            prefix = (
                "gpu_i8_centroid_budget_policy_"
                f"{row['name']}_{policy_row['policy_name']}"
            )
            comparison = policy_row["comparison"]
            print_quiet_mean(
                f"{prefix}_candidate_plus_score",
                comparison["cpu_candidate_generation_plus_score_seconds"],
            )
            print_quiet_mean(
                f"{prefix}_candidate_plus_score_vs_static32",
                comparison["candidate_plus_score_seconds_vs_baseline_budget"],
            )
            print_quiet_mean(
                f"{prefix}_final_recall",
                policy_row["recall_at_k_vs_kayak_exact"],
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
    report = build_report(
        cases=selected_cases(args),
        controls=controls_from_args(args),
        policy_names=tuple(args.policies),
        case_selection=case_selection_from_args(args),
    )
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        emit_quiet_means(report)
    return 0 if report["status"] == STATUS_OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
