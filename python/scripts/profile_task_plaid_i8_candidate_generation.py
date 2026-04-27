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

from kayak_bridge.json_task_loader import load_task_json  # noqa: E402
from kayak_bridge.late_ops import MOJO_EXACT_CPU_BACKEND  # noqa: E402
from kayak_bridge.plaid_task_candidate_profile import (  # noqa: E402
    PlaidTaskCandidateProfileControls,
    profile_task_plaid_i8_candidate_generation,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Kayak PLAID i8 candidate-generation substeps on an "
            "encoded task JSON. This is the real-task bridge between the "
            "synthetic GPU/FastPlaid matrix and corpus-scale serving tests."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".cache/kayak/plaid_i8_task_candidate_generation_profile/summary.json"
        ),
    )
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--centroid-count", type=int, default=128)
    parser.add_argument("--centroids-per-query-vector", type=int, default=32)
    parser.add_argument("--candidate-k", type=int, default=256)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--exact-backend", default=MOJO_EXACT_CPU_BACKEND)
    parser.add_argument(
        "--skip-exact-reference",
        action="store_true",
        help=(
            "Skip exact-reference recall. Use this on corpus-scale runs where "
            "full exact search is intentionally not part of the timing."
        ),
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    task = load_task_json(str(args.task))
    report = profile_task_plaid_i8_candidate_generation(
        task,
        PlaidTaskCandidateProfileControls(
            centroid_count=args.centroid_count,
            centroids_per_query_vector=args.centroids_per_query_vector,
            candidate_k=args.candidate_k,
            query_limit=args.query_limit,
            measurement_iterations=args.measurement_iterations,
            exact_reference=not args.skip_exact_reference,
            exact_backend=args.exact_backend,
        ),
    )
    report["created_at_utc"] = datetime.now(UTC).isoformat()
    report["task_path"] = str(args.task)
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def emit_quiet_means(report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    quiet_fields = (
        "full_candidate_mean_seconds_mean",
        "posting_accumulation_mean_seconds_mean",
        "final_topk_mean_seconds_mean",
        "candidate_document_vector_count_mean",
        "candidate_recall_at_k_vs_exact_mean",
    )
    for field in quiet_fields:
        value = aggregate.get(field)
        if isinstance(value, (float, int)):
            print(f"== plaid_i8_task_candidate_generation_{field} ==")
            print("Mean:", value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    write_report(report, args.output)
    print(f"wrote {args.output}")
    if args.emit_quiet_mean:
        emit_quiet_means(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
