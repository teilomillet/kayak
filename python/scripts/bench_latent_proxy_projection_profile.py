from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.latent_proxy_projection_profile import (
    benchmark_latent_proxy_projection_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark native latent-proxy query projection and proxy scan "
            "microkernels on one task JSON and one exported artifact."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = benchmark_latent_proxy_projection_profile(
        task_path=args.task,
        artifact_root=args.artifact_root,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"Mean projection: {summary['mean_projection_seconds']}")


if __name__ == "__main__":
    main()
