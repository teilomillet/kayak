from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.task_json_catalog import (
    build_named_task_json,
    default_task_json_output_path,
    load_task_json,
    named_task_json_keys,
    write_task_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a named encoded task JSON for cross-engine benchmarking."
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        required=True,
        choices=named_task_json_keys(),
        help="Stable task key to build.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to the task key's cache path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the output JSON already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or default_task_json_output_path(args.dataset_key)

    if output_path.exists() and not args.force:
        task = load_task_json(output_path)
        print(f"using cached task json: {output_path}")
        print(f"slice: {task['slice_name']}")
        return

    task = build_named_task_json(args.dataset_key)
    write_task_json(output_path, task)
    print(f"wrote task json: {output_path}")
    print(f"slice: {task['slice_name']}")


if __name__ == "__main__":
    main()
