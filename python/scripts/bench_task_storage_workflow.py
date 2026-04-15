from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.task_storage_workflow import (
    benchmark_task_storage_workflow,
    build_task_storage_workflow_bundle,
    default_queries_per_loads,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "task"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark load-heavy Kayak storage workflows for a task JSON."
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to an encoded task JSON file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the task JSON parent.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default=None,
        help="Optional artifact filename prefix. Defaults to a slugified slice name.",
    )
    parser.add_argument(
        "--queries-per-load",
        type=int,
        action="append",
        dest="queries_per_loads",
        help="Queries to execute per load-heavy session. Repeat to add more points.",
    )
    args = parser.parse_args()

    task = load_task_json(args.task)
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))
    queries_per_loads = (
        tuple(int(value) for value in args.queries_per_loads)
        if args.queries_per_loads
        else default_queries_per_loads(len(task["queries"]))
    )

    summaries = benchmark_task_storage_workflow(
        task,
        queries_per_loads=queries_per_loads,
    )
    bundle = build_task_storage_workflow_bundle(
        summaries,
        queries_per_loads=queries_per_loads,
    )

    summaries_path = output_root / f"{artifact_prefix}_storage_workflow_summaries.json"
    bundle_path = output_root / f"{artifact_prefix}_storage_workflow_bundle.json"
    with summaries_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with bundle_path.open("w", encoding="utf-8") as handle:
        json.dump(bundle.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {summaries_path}")
    print(f"wrote {bundle_path}")
    for summary in summaries:
        print(
            "encoding="
            f"{summary['encoding_kind']} "
            f"queries_per_load={summary['queries_per_load']} "
            f"session_s={summary['mean_session_seconds']} "
            f"session_s_per_query={summary['mean_session_seconds_per_query']}"
        )
        print(f"Mean: {summary['mean_session_seconds']}")


if __name__ == "__main__":
    main()
