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
from kayak_bridge.scale_comparison import benchmark_same_task_scale_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kayak exact versus LanceDB scan over scaled versions of "
            "the same encoded task."
        )
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the encoded task JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the scale sweep JSON should be written.",
    )
    parser.add_argument(
        "--db-root",
        type=Path,
        required=True,
        help="Directory where temporary LanceDB databases should be created.",
    )
    parser.add_argument(
        "--table-prefix",
        type=str,
        default="scale_sweep",
        help="LanceDB table prefix used for sweep runs.",
    )
    parser.add_argument(
        "--target-document-count",
        type=int,
        action="append",
        dest="target_document_counts",
        help=(
            "Target document count for one sweep point. Repeat this flag to add "
            "multiple points."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Warmup passes for each benchmark run.",
    )
    parser.add_argument(
        "--measurement-iterations",
        type=int,
        default=3,
        help="Measured passes for each benchmark run.",
    )
    return parser.parse_args()


def _default_target_document_counts(task: dict[str, object]) -> list[int]:
    base_document_count = len(task["documents"])
    return [
        base_document_count,
        base_document_count * 2,
        base_document_count * 4,
        base_document_count * 8,
    ]


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    target_document_counts = args.target_document_counts
    if not target_document_counts:
        target_document_counts = _default_target_document_counts(task)

    summary = benchmark_same_task_scale_sweep(
        task,
        database_root=args.db_root,
        table_prefix=args.table_prefix,
        target_document_counts=target_document_counts,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    ).to_json_ready()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(
        "Mean: "
        f"{summary['rows'][-1]['lancedb_scan_mean_search_seconds']}"
    )


if __name__ == "__main__":
    main()
