from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.task_storage_encoding import (
    benchmark_task_storage_encodings,
    build_task_storage_encoding_bundle,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "task"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Kayak native packed storage encodings for a task JSON."
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
    args = parser.parse_args()

    task = load_task_json(args.task)
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))

    summaries = benchmark_task_storage_encodings(task)
    bundle = build_task_storage_encoding_bundle(summaries)

    summaries_path = output_root / f"{artifact_prefix}_storage_encoding_summaries.json"
    bundle_path = output_root / f"{artifact_prefix}_storage_encoding_bundle.json"
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
            f"bytes={summary['artifact_byte_size']} "
            f"build_s={summary['mean_build_seconds']} "
            f"load_s={summary['mean_load_seconds']} "
            f"search_s={summary['mean_search_seconds']} "
            f"primary={summary['primary_value']}"
        )
        print(f"Mean: {summary['mean_search_seconds']}")


if __name__ == "__main__":
    main()
