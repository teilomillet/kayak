from __future__ import annotations

import argparse
from dataclasses import replace
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

from kayak_bridge.json_task_loader import load_task_json  # noqa: E402
from kayak_bridge.late_ops import MOJO_EXACT_CPU_BACKEND, search  # noqa: E402
from kayak_bridge.plaid_task_candidate_profile import (  # noqa: E402
    PlaidTaskCandidateProfileControls,
    build_late_index_from_task,
    build_late_queries_from_task,
    profile_prepared_task_plaid_i8_candidate_generation,
)
from kayak_bridge.plaid_approx import (  # noqa: E402
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep Kayak PLAID i8 candidate-window and centroid-budget knobs "
            "on one encoded task JSON."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".cache/kayak/plaid_i8_task_candidate_generation_sweep/summary.json"
        ),
    )
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--centroid-count", type=int, default=128)
    parser.add_argument(
        "--centroids-per-query-vector",
        type=int,
        action="append",
        dest="centroid_budgets",
        required=True,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        action="append",
        dest="candidate_windows",
        required=True,
    )
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--exact-backend", default=MOJO_EXACT_CPU_BACKEND)
    parser.add_argument("--skip-exact-reference", action="store_true")
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    task = load_task_json(str(args.task))
    final_k = int(task["k"])
    candidate_windows = dedupe_ints(args.candidate_windows)
    centroid_budgets = dedupe_ints(args.centroid_budgets)

    started_at = time.perf_counter()
    late_index = build_late_index_from_task(task)
    late_queries = build_late_queries_from_task(task)
    selected_queries = (
        late_queries
        if args.query_limit is None
        else late_queries[: args.query_limit]
    )
    index_build_seconds = time.perf_counter() - started_at

    if not selected_queries:
        raise ValueError("task must provide at least one selected query")

    prepare_started_at = time.perf_counter()
    base_plaid_index = KayakPlaidApproxIndex.from_late_index(
        late_index,
        config=KayakPlaidApproxConfig(
            centroid_count=args.centroid_count,
            centroids_per_query_vector=centroid_budgets[0],
            candidate_k=max(candidate_windows),
            payload="i8",
        ),
        final_k=final_k,
    )
    plaid_prepare_seconds = time.perf_counter() - prepare_started_at
    exact_doc_ids_by_query = None
    if not args.skip_exact_reference:
        exact_doc_ids_by_query = [
            tuple(
                hit.doc_id
                for hit in search(
                    late_query,
                    late_index,
                    k=final_k,
                    backend=args.exact_backend,
                )
            )
            for late_query in selected_queries
        ]

    reports = []
    for centroid_budget in centroid_budgets:
        for candidate_k in candidate_windows:
            row_config = KayakPlaidApproxConfig(
                centroid_count=args.centroid_count,
                centroids_per_query_vector=centroid_budget,
                candidate_k=candidate_k,
                payload="i8",
            )
            report = profile_prepared_task_plaid_i8_candidate_generation(
                task,
                late_index=late_index,
                selected_queries=selected_queries,
                plaid_index=replace(base_plaid_index, config=row_config),
                controls=PlaidTaskCandidateProfileControls(
                    centroid_count=args.centroid_count,
                    centroids_per_query_vector=centroid_budget,
                    candidate_k=candidate_k,
                    query_limit=args.query_limit,
                    measurement_iterations=args.measurement_iterations,
                    exact_reference=not args.skip_exact_reference,
                    exact_backend=args.exact_backend,
                ),
                index_build_seconds=index_build_seconds,
                plaid_prepare_seconds=plaid_prepare_seconds,
                exact_doc_ids_by_query=exact_doc_ids_by_query,
            )
            reports.append(
                {
                    "candidate_k": candidate_k,
                    "centroids_per_query_vector": centroid_budget,
                    "status": report["status"],
                    "task": report["task"],
                    "index": report["index"],
                    "queries": report["queries"],
                    "aggregate": report["aggregate"],
                }
            )
    return {
        "schema_version": 1,
        "benchmark": "plaid_i8_task_candidate_generation_sweep",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "ok" if all(row["status"] == "ok" for row in reports) else "error",
        "task_path": str(args.task),
        "controls": {
            "centroid_count": args.centroid_count,
            "centroids_per_query_vector": centroid_budgets,
            "candidate_k": candidate_windows,
            "query_limit": args.query_limit,
            "measurement_iterations": args.measurement_iterations,
            "exact_reference": not args.skip_exact_reference,
            "exact_backend": args.exact_backend,
            "exact_reference_cached": exact_doc_ids_by_query is not None,
        },
        "rows": reports,
        "measurement_note": (
            "The sweep builds the public task index and internal PLAID i8 "
            "prepared index once, then reuses that prepared payload across "
            "candidate-window and centroid-budget rows. Row aggregates are "
            "the candidate-generation policy evidence; setup timings are "
            "shared context. Exact-reference top-k doc ids are also cached "
            "once per selected query when exact reference is enabled."
        ),
    }


def dedupe_ints(values: Sequence[int]) -> list[int]:
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def emit_quiet_means(report: dict[str, Any]) -> None:
    for row in report["rows"]:
        prefix = (
            "plaid_i8_task_candidate_generation_sweep_"
            f"k{row['candidate_k']}_c{row['centroids_per_query_vector']}"
        )
        aggregate = row["aggregate"]
        print_quiet_mean(
            f"{prefix}_full_candidate_seconds",
            aggregate.get("full_candidate_mean_seconds_mean"),
        )
        print_quiet_mean(
            f"{prefix}_candidate_recall",
            aggregate.get("candidate_recall_at_k_vs_exact_mean"),
        )


def print_quiet_mean(section: str, value: object) -> None:
    if isinstance(value, (float, int)):
        print(f"== {section} ==")
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
