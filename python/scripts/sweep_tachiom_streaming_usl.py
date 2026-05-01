from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    benchmark_streaming_tachiom_index,
)
from kayak_bridge.usl_scaling import UslObservation, fit_usl  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep streaming Tachiom same-shape query batch caps and fit the "
            "Universal Scalability Law to measured throughput."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--engine",
        choices=(
            "streaming_tac_pq",
            "streaming_tac_pq_mojo",
            "streaming_tac_hnsw_pq",
            "streaming_tac_hnsw_pq_mojo",
            "streaming_tac_hnsw_pq_mojo_address",
        ),
        default="streaming_tac_hnsw_pq_mojo",
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--batch-sizes",
        required=True,
        help="Comma-separated positive same-shape query batch caps, e.g. 1,2,4,8,16.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--sweep-repeats",
        type=int,
        default=1,
        help=(
            "Repeat every batch-size measurement and fit USL to median QPS. "
            "This makes the JSON self-contained instead of relying on the "
            "outer quiet wrapper to aggregate repeats."
        ),
    )
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--max-exact-vector-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sweep_repeats <= 0:
        raise ValueError("sweep_repeats must be positive")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    measurement_rows = []
    for repeat_index in range(args.sweep_repeats):
        for batch_size in batch_sizes:
            summary = benchmark_streaming_tachiom_index(
                snapshot_root=args.snapshot,
                index_root=args.index,
                query_limit=args.query_limit,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                run_exact=args.run_exact,
                max_exact_vector_count=args.max_exact_vector_count,
                engine=args.engine,
                graph_root=args.graph,
                max_query_batch_size=batch_size,
            )
            measurement_rows.append(
                {
                    "repeat_index": repeat_index,
                    "summary": summary,
                }
            )
    aggregate_rows = _aggregate_rows(
        batch_sizes=batch_sizes,
        measurement_rows=measurement_rows,
    )

    fit = fit_usl(
        [
            UslObservation(
                scale=int(row["max_query_batch_size"]),
                throughput=float(row["query_qps_median"]),
            )
            for row in aggregate_rows
        ],
        max_predicted_scale=max(batch_sizes),
    )
    best_row = max(aggregate_rows, key=lambda row: float(row["query_qps_median"]))
    payload = {
        "engine": args.engine,
        "snapshot": str(args.snapshot),
        "index": str(args.index),
        "graph": None if args.graph is None else str(args.graph),
        "batch_sizes": batch_sizes,
        "sweep_repeats": args.sweep_repeats,
        "usl_fit": fit.to_json_ready(),
        "recommendation": {
            "max_query_batch_size": best_row["max_query_batch_size"],
            "query_qps_median": best_row["query_qps_median"],
            "query_batch_median_seconds": best_row["query_batch_median_seconds"],
            "decision_rule": (
                "Use the best measured median batch cap, not an extrapolated USL peak."
            ),
            "reason": (
                "The sweep changes only query-call batching. Retrieval rankings, "
                "vector counts, index bytes, and candidate budgets remain fixed."
            ),
        },
        "rows": aggregate_rows,
        "measurements": [
            {
                "repeat_index": int(row["repeat_index"]),
                "summary": row["summary"].to_json_ready(),
            }
            for row in measurement_rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.emit_quiet_mean:
        for row in aggregate_rows:
            print(f"== batch_size_{row['max_query_batch_size']} ==")
            print("Mean:", row["query_batch_median_seconds"])
        print("== best_observed ==")
        print("Mean:", best_row["query_batch_median_seconds"])
        print(
            "quiet_context "
            f"best_batch_size={best_row['max_query_batch_size']} "
            f"best_qps={float(best_row['query_qps_median']):.6f} "
            f"usl_alpha={fit.alpha:.9f} "
            f"usl_beta={fit.beta:.9f} "
            f"usl_r2={fit.r_squared:.6f}"
        )
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_batch_sizes(value: str) -> list[int]:
    sizes: list[int] = []
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped:
            continue
        size = int(stripped)
        if size <= 0:
            raise ValueError("batch sizes must be positive")
        if size not in sizes:
            sizes.append(size)
    if len(sizes) < 3:
        raise ValueError("USL sweep requires at least three unique batch sizes")
    return sizes


def _aggregate_rows(
    *,
    batch_sizes: Sequence[int],
    measurement_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        summaries = [
            row["summary"]
            for row in measurement_rows
            if row["summary"].max_query_batch_size == batch_size
        ]
        if not summaries:
            raise RuntimeError(f"missing measurements for batch size {batch_size}")
        durations = [row.query_batch_mean_seconds for row in summaries]
        median_seconds = float(median(durations))
        query_count = summaries[0].query_count
        aggregate_rows.append(
            {
                "max_query_batch_size": batch_size,
                "query_batch_min_seconds": float(min(durations)),
                "query_batch_median_seconds": median_seconds,
                "query_batch_mean_seconds": float(mean(durations)),
                "query_batch_max_seconds": float(max(durations)),
                "query_qps_median": float(query_count) / median_seconds,
                "repeat_count": len(summaries),
                "primary_value": summaries[0].primary_value,
                "candidate_recall_at_k_vs_exact": (
                    summaries[0].candidate_recall_at_k_vs_exact
                ),
                "final_recall_at_k_vs_exact": summaries[0].final_recall_at_k_vs_exact,
                "query_count": query_count,
                "document_count": summaries[0].document_count,
                "document_vector_count_total": summaries[
                    0
                ].document_vector_count_total,
                "query_vector_count_total": summaries[0].query_vector_count_total,
                "query_vector_count_mean": summaries[0].query_vector_count_mean,
                "vector_dim": summaries[0].vector_dim,
                "centroid_count": summaries[0].centroid_count,
                "posting_count": summaries[0].posting_count,
                "candidate_window_mean_count": summaries[
                    0
                ].candidate_window_mean_count,
                "index_bytes": summaries[0].index_bytes,
            }
        )
    return aggregate_rows


if __name__ == "__main__":
    raise SystemExit(main())
