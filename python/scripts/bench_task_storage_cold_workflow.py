from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.task_storage_cold_workflow import (
    benchmark_task_storage_cold_workflow,
    default_cold_queries_per_loads,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "task"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark cold-ish load-heavy Kayak storage workflows for a task JSON."
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--artifact-prefix", type=str, default=None)
    parser.add_argument(
        "--queries-per-load",
        type=int,
        action="append",
        dest="queries_per_loads",
        help="Queries to execute per cold-ish session. Repeat to add more points.",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--cold-percent-free",
        type=int,
        default=60,
        help="Target free-memory percentage passed to memory_pressure -p.",
    )
    parser.add_argument("--cold-sample-seconds", type=int, default=1)
    parser.add_argument("--cold-hysteresis-seconds", type=int, default=1)
    args = parser.parse_args()

    task = load_task_json(args.task)
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))
    queries_per_loads = (
        tuple(int(value) for value in args.queries_per_loads)
        if args.queries_per_loads
        else default_cold_queries_per_loads()
    )

    bundle = benchmark_task_storage_cold_workflow(
        task,
        queries_per_loads=queries_per_loads,
        repetitions=args.repetitions,
        percent_free=args.cold_percent_free,
        sample_seconds=args.cold_sample_seconds,
        hysteresis_seconds=args.cold_hysteresis_seconds,
    )

    bundle_path = output_root / f"{artifact_prefix}_storage_cold_workflow_bundle.json"
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {bundle_path}")
    for point in bundle.points:
        print(
            "queries_per_load="
            f"{point.queries_per_load} "
            f"binary_session_s={point.binary_le_mean_session_seconds} "
            f"f16_session_s={point.binary_f16_le_mean_session_seconds} "
            f"ratio={point.session_seconds_ratio_f16_vs_binary}"
        )
        print(f"Mean: {point.binary_f16_le_mean_session_seconds}")


if __name__ == "__main__":
    main()
