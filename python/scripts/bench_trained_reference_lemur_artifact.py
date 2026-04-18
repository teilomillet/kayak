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
from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.latent_proxy_task_benchmark import (
    benchmark_task_with_latent_proxy_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one exported trained LEMUR latent-proxy artifact on an "
            "encoded task JSON."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help=(
            "Directory containing one latent-proxy artifact. If checkpoint paths "
            "are also provided, the artifact is (re)exported there first."
        ),
    )
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--mlp-path", type=Path)
    parser.add_argument("--w-path", type=Path)
    parser.add_argument("--query-divisor", type=float, default=32.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="Candidate shortlist size. Defaults to the task final k.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measurement-iterations", type=int, default=25)
    parser.add_argument("--rerank-backend", type=str, default="numpy_reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    candidate_k = int(task["k"]) if args.candidate_k is None else args.candidate_k

    if (
        args.index_root is not None
        or args.mlp_path is not None
        or args.w_path is not None
    ):
        import kayak

        index = kayak.documents(
            [document["doc_id"] for document in task["documents"]],
            [document["vectors"] for document in task["documents"]],
            texts=[document["text"] for document in task["documents"]],
        ).pack()
        export_trained_reference_lemur_artifact(
            index,
            args.artifact_root,
            index_root=args.index_root,
            mlp_path=args.mlp_path,
            w_path=args.w_path,
            query_divisor=args.query_divisor,
            model_name=str(task["model_name"]),
        )

    summary = benchmark_task_with_latent_proxy_artifact(
        task,
        artifact_root=args.artifact_root,
        candidate_k=candidate_k,
        device=args.device,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        rerank_backend=args.rerank_backend,
    ).to_json_ready()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"Mean: {summary['mean_search_seconds']}")


if __name__ == "__main__":
    main()
