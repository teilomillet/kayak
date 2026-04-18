from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)

import kayak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained LEMUR checkpoint pair into Kayak's native latent-proxy sidecar format."
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--mlp-path", type=Path)
    parser.add_argument("--w-path", type=Path)
    parser.add_argument("--query-divisor", type=float, default=32.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    index = kayak.documents(
        [document["doc_id"] for document in task["documents"]],
        [document["vectors"] for document in task["documents"]],
        texts=[document.get("text", "") for document in task["documents"]],
    ).pack()
    export_trained_reference_lemur_artifact(
        index,
        args.output_root,
        index_root=args.index_root,
        mlp_path=args.mlp_path,
        w_path=args.w_path,
        query_divisor=args.query_divisor,
        model_name=str(task.get("model_name", "")),
    )
    print(f"wrote {args.output_root}")


if __name__ == "__main__":
    main()
