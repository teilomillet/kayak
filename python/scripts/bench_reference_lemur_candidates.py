from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.lemur_candidate_contract import (
    benchmark_reference_lemur_contract,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def _parse_int_list(value: str, *, label: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"{label} must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reference LEMUR exact-vs-candidate sweep contract on one "
            "encoded task JSON."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory where artifacts should be written. Defaults to task parent.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        help="Artifact filename prefix. Defaults to the task slice name.",
    )
    parser.add_argument(
        "--latent-dims",
        type=str,
        required=True,
        help="Comma-separated latent dimensions such as `16,32,64`.",
    )
    parser.add_argument(
        "--candidate-ks",
        type=str,
        required=True,
        help="Comma-separated candidate window sizes such as `10,20,40`.",
    )
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--query-divisor", type=float, default=32.0)
    parser.add_argument("--landmark-count", type=int)
    parser.add_argument(
        "--disable-layer-norm",
        action="store_true",
        help="Disable row-wise layer normalization in the reference feature map.",
    )
    parser.add_argument("--layer-norm-eps", type=float, default=1e-5)
    parser.add_argument("--pinv-rcond", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-backend", type=str, default=kayak.NUMPY_REFERENCE_BACKEND)
    parser.add_argument("--rerank-backend", type=str, default=kayak.NUMPY_REFERENCE_BACKEND)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--measurement-iterations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))

    result = benchmark_reference_lemur_contract(
        task=task,
        output_root=output_root,
        artifact_prefix=artifact_prefix,
        latent_dims=_parse_int_list(args.latent_dims, label="latent-dims"),
        candidate_ks=_parse_int_list(args.candidate_ks, label="candidate-ks"),
        activation=args.activation,
        query_divisor=args.query_divisor,
        landmark_count=args.landmark_count,
        apply_layer_norm=not args.disable_layer_norm,
        layer_norm_eps=args.layer_norm_eps,
        pinv_rcond=args.pinv_rcond,
        seed=args.seed,
        exact_backend=args.exact_backend,
        rerank_backend=args.rerank_backend,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    print(f"wrote {result['bundle_path']}")
    print(
        "Best quality candidate: "
        f"{result['lemur_bundle']['best_quality_candidate_name']}"
    )


if __name__ == "__main__":
    main()
