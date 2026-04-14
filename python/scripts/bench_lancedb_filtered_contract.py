from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.lancedb_filtered_contract import benchmark_filtered_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the filtered three-system LanceDB contract on one encoded task."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        required=True,
        help="Prefix for all emitted artifacts under output-root.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = benchmark_filtered_contract(
        task=load_task_json(str(args.task)),
        output_root=args.output_root,
        artifact_prefix=args.artifact_prefix,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    print(f"wrote {result['bundle_path']}")
    print(
        "LanceDB/Kayak ratio: "
        f"{result['scorecard']['pairwise'][0]['mean_search_seconds_ratio_vs_baseline']}"
    )


if __name__ == "__main__":
    main()
