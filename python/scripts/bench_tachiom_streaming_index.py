from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    benchmark_streaming_tachiom_index,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a materialized streaming TAC/PQ index against snapshot "
            "query sidecars."
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
        default="streaming_tac_pq",
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--max-query-batch-size",
        type=int,
        help=(
            "Cap same-shape query batches for scaling sweeps. The default keeps "
            "the benchmark's full same-shape batch."
        ),
    )
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--max-exact-vector-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
        max_query_batch_size=args.max_query_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary.to_json_ready(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.emit_quiet_mean:
        print("Mean:", summary.query_batch_mean_seconds)
        print(
            "quiet_context "
            f"qps={summary.query_qps:.6f} "
            f"primary={summary.primary_value:.6f} "
            f"candidate_recall={summary.candidate_recall_at_k_vs_exact} "
            f"final_recall={summary.final_recall_at_k_vs_exact}"
        )
    else:
        print(json.dumps(summary.to_json_ready(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
