from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from kayak_bridge.release_wheel_check import inspect_wheel

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that a built Kayak release wheel ships only the bundled "
            "mojopkg artifact, not engine source code."
        )
    )
    parser.add_argument("wheel", type=Path, help="path to a built .whl file")
    args = parser.parse_args(argv)

    summary = inspect_wheel(args.wheel)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
