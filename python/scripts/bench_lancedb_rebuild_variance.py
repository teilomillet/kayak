from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.benchmark_variance import summarize_benchmark_variance
from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb
from kayak_bridge.lancedb_index_controls import (
    LanceDbIndexBuildControls,
    LanceDbIndexedQueryControls,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure LanceDB rebuild variance over repeated benchmark runs."
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db-root-stem", type=Path, required=True)
    parser.add_argument("--table-name", type=str, required=True)
    parser.add_argument("--rebuild-count", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--index-num-partitions", type=int)
    parser.add_argument("--index-num-sub-vectors", type=int)
    parser.add_argument("--index-target-partition-size", type=int)
    parser.add_argument("--indexed-nprobes", type=int)
    parser.add_argument("--indexed-refine-factor", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_count <= 0:
        raise ValueError("rebuild-count must be positive")

    task = load_task_json(str(args.task))
    runs = []
    for rebuild_index in range(args.rebuild_count):
        summary = benchmark_task_with_lancedb(
            task=task,
            database_root=Path(f"{args.db_root_stem}-{rebuild_index:02d}"),
            table_name=args.table_name,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            build_index=args.build_index,
            index_build_controls=LanceDbIndexBuildControls(
                num_partitions=args.index_num_partitions,
                num_sub_vectors=args.index_num_sub_vectors,
                target_partition_size=args.index_target_partition_size,
            ),
            indexed_query_controls=LanceDbIndexedQueryControls(
                nprobes=args.indexed_nprobes,
                refine_factor=args.indexed_refine_factor,
            ),
        )
        runs.append(summary.to_json_ready())

    variance = summarize_benchmark_variance(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(variance.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"Mean: {variance.mean_search_seconds_mean}")


if __name__ == "__main__":
    main()
