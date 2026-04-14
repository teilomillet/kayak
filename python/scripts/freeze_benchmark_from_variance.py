from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.benchmark_variance import freeze_benchmark_summary_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze one benchmark summary from a rebuild-variance artifact."
    )
    parser.add_argument("--variance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--freeze-policy",
        type=str,
        default="mean_across_rebuilds",
        help="Label recorded in the frozen summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.variance.open("r", encoding="utf-8") as handle:
        variance = json.load(handle)

    frozen = freeze_benchmark_summary_mean(
        variance["run_summaries"],
        freeze_policy=args.freeze_policy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(frozen.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
