from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark LanceDB multivector search on an encoded Kayak task JSON."
        )
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to a compatible encoded task JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the machine-readable benchmark summary JSON.",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        required=True,
        help="Directory where the temporary LanceDB database should be created.",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        required=True,
        help="LanceDB table name for the benchmark run.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=2,
        help="Full-query warmup passes excluded from latency measurement.",
    )
    parser.add_argument(
        "--measurement-iterations",
        type=int,
        default=25,
        help="Full-query measurement passes used for mean per-query latency.",
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build a LanceDB IVF_PQ index before searching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    summary = benchmark_task_with_lancedb(
        task=task,
        database_root=args.db_root,
        table_name=args.table_name,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        build_index=args.build_index,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"Mean: {summary.mean_search_seconds}")


if __name__ == "__main__":
    main()
