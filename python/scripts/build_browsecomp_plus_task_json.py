from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.browsecomp_plus_subset import (
    build_browsecomp_plus_encoded_slice,
    build_browsecomp_plus_task_from_encoded_slice,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / ".cache"
    / "kayak"
    / "browsecomp_plus_real_subset"
)
EVIDENCE_OUTPUT_PATH = OUTPUT_ROOT / "python_task_evidence.json"
GOLD_OUTPUT_PATH = OUTPUT_ROOT / "python_task_gold.json"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if EVIDENCE_OUTPUT_PATH.exists() and GOLD_OUTPUT_PATH.exists():
        print(f"using cached task json: {EVIDENCE_OUTPUT_PATH}")
        print(f"using cached task json: {GOLD_OUTPUT_PATH}")
        return

    encoded_slice = build_browsecomp_plus_encoded_slice()
    evidence_task = build_browsecomp_plus_task_from_encoded_slice(
        encoded_slice, relevance_kind="evidence"
    )
    gold_task = build_browsecomp_plus_task_from_encoded_slice(
        encoded_slice, relevance_kind="gold"
    )

    with EVIDENCE_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(evidence_task, handle)

    with GOLD_OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(gold_task, handle)

    print(f"wrote task json: {EVIDENCE_OUTPUT_PATH}")
    print(f"wrote task json: {GOLD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
