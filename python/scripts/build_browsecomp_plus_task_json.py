from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.browsecomp_plus_subset import build_browsecomp_plus_colbert_subset


OUTPUT_PATH = (
    REPO_ROOT
    / ".cache"
    / "kayak"
    / "browsecomp_plus_real_subset"
    / "python_task.json"
)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"using cached task json: {OUTPUT_PATH}")
        return

    task = build_browsecomp_plus_colbert_subset()
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(task, handle)

    print(f"wrote task json: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
