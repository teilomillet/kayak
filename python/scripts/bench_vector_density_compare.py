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
from kayak_bridge.vector_density_comparison import benchmark_vector_density_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kayak exact versus LanceDB scan while keeping document "
            "count fixed and increasing document vector density."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--table-prefix", type=str, default="vector_density")
    parser.add_argument(
        "--document-vector-multiplier",
        type=int,
        action="append",
        dest="document_vector_multipliers",
        help="Repeat this flag to add more vector-density sweep points.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    return parser.parse_args()


def _default_document_vector_multipliers() -> list[int]:
    return [1, 2, 4, 8]


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    multipliers = args.document_vector_multipliers
    if not multipliers:
        multipliers = _default_document_vector_multipliers()

    summary = benchmark_vector_density_sweep(
        task,
        database_root=args.db_root,
        table_prefix=args.table_prefix,
        document_vector_multipliers=multipliers,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    ).to_json_ready()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(
        "Largest-point ratio (LanceDB/Kayak): "
        f"{summary['rows'][-1]['lancedb_scan_latency_ratio_vs_kayak']}"
    )


if __name__ == "__main__":
    main()
