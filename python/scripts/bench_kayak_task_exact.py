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
from kayak_bridge.kayak_task_benchmark import benchmark_task_with_kayak_exact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Kayak exact search on an encoded task JSON."
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measurement-iterations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    summary = benchmark_task_with_kayak_exact(
        task,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    ).to_json_ready()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"Mean: {summary['mean_search_seconds']}")


if __name__ == "__main__":
    main()
