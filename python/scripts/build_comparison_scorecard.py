from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.comparison_scorecard import build_comparison_scorecard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a comparison scorecard from benchmark summary JSON files."
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        required=True,
        help="System name to use as the pairwise baseline.",
    )
    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help=(
            "System spec in the form name=path/to/summary.json. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the scorecard JSON output.",
    )
    return parser.parse_args()


def _parse_system_spec(spec: str) -> tuple[str, str, dict]:
    if "=" not in spec:
        raise ValueError(f"invalid --system spec: {spec}")
    name, path = spec.split("=", 1)
    summary_path = Path(path)
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return name, str(summary_path), summary


def main() -> None:
    args = parse_args()
    systems = [_parse_system_spec(spec) for spec in args.system]
    scorecard = build_comparison_scorecard(
        systems=systems,
        baseline_name=args.baseline_name,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(scorecard.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
