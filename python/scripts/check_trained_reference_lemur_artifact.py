from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.latent_proxy_reference import (
    compute_query_latent_proxy_features,
    latent_proxy_similarity_scores,
    load_latent_proxy_artifact,
)
from kayak_bridge.trained_reference_lemur import load_trained_reference_lemur

import kayak


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained LEMUR checkpoint to the latent-proxy artifact "
            "contract and compare artifact scores to the original checkpoint."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--mlp-path", type=Path)
    parser.add_argument("--w-path", type=Path)
    parser.add_argument("--query-divisor", type=float, default=32.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--topk", type=int)
    return parser.parse_args()


def _overlap_at_k(
    reference_doc_ids: tuple[str, ...],
    artifact_doc_ids: tuple[str, ...],
    *,
    k: int,
) -> float:
    reference_topk = set(reference_doc_ids[:k])
    artifact_topk = set(artifact_doc_ids[:k])
    if not reference_topk:
        return 1.0
    return len(reference_topk & artifact_topk) / float(len(reference_topk))


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    index = kayak.documents(
        [document["doc_id"] for document in task["documents"]],
        [document["vectors"] for document in task["documents"]],
        texts=[document["text"] for document in task["documents"]],
    ).pack()
    queries = tuple(
        kayak.query(query["vectors"], text=query["text"])
        for query in task["queries"]
    )
    reference = load_trained_reference_lemur(
        index,
        index_root=args.index_root,
        mlp_path=args.mlp_path,
        w_path=args.w_path,
        query_divisor=args.query_divisor,
        device=args.device,
    )
    export_trained_reference_lemur_artifact(
        index,
        args.artifact_root,
        index_root=args.index_root,
        mlp_path=args.mlp_path,
        w_path=args.w_path,
        query_divisor=args.query_divisor,
        model_name=str(task.get("model_name", "")),
    )
    artifact = load_latent_proxy_artifact(args.artifact_root)

    topk = int(task["k"]) if args.topk is None else int(args.topk)
    topk = min(topk, index.document_count)
    feature_max_abs_errors: list[float] = []
    feature_mean_abs_errors: list[float] = []
    score_max_abs_errors: list[float] = []
    score_mean_abs_errors: list[float] = []
    topk_overlaps: list[float] = []
    topk_exact_matches = 0

    for query in queries:
        reference_features = reference.compute_query_features(query)
        artifact_features = compute_query_latent_proxy_features(
            query,
            artifact,
            device=args.device,
        )
        feature_delta = np.abs(reference_features - artifact_features)
        feature_max_abs_errors.append(float(np.max(feature_delta)))
        feature_mean_abs_errors.append(float(np.mean(feature_delta)))

        reference_scores = reference.similarity_scores(query)
        artifact_scores = latent_proxy_similarity_scores(
            query,
            artifact,
            device=args.device,
        )
        score_delta = np.abs(reference_scores.values - artifact_scores.values)
        score_max_abs_errors.append(float(np.max(score_delta)))
        score_mean_abs_errors.append(float(np.mean(score_delta)))

        reference_topk = tuple(hit.doc_id for hit in reference_scores.topk(topk))
        artifact_topk = tuple(hit.doc_id for hit in artifact_scores.topk(topk))
        if reference_topk == artifact_topk:
            topk_exact_matches += 1
        topk_overlaps.append(
            _overlap_at_k(reference_topk, artifact_topk, k=topk)
        )

    summary = {
        "dataset_id": str(task["dataset_id"]),
        "model_name": str(task["model_name"]),
        "family": str(task["family"]),
        "slice_name": str(task["slice_name"]),
        "query_count": len(queries),
        "document_count": index.document_count,
        "vector_dim": index.vector_dim,
        "latent_dim": int(artifact.proxy_vectors.shape[1]),
        "query_divisor": float(args.query_divisor),
        "artifact_root": str(args.artifact_root),
        "topk": topk,
        "max_feature_abs_error": max(feature_max_abs_errors, default=0.0),
        "mean_feature_abs_error": (
            float(sum(feature_mean_abs_errors) / len(feature_mean_abs_errors))
            if feature_mean_abs_errors
            else 0.0
        ),
        "max_score_abs_error": max(score_max_abs_errors, default=0.0),
        "mean_score_abs_error": (
            float(sum(score_mean_abs_errors) / len(score_mean_abs_errors))
            if score_mean_abs_errors
            else 0.0
        ),
        "mean_topk_overlap": (
            float(sum(topk_overlaps) / len(topk_overlaps))
            if topk_overlaps
            else 1.0
        ),
        "exact_topk_match_rate": (
            float(topk_exact_matches) / float(len(queries)) if queries else 1.0
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote {args.output}")
    print(f"max_score_abs_error={summary['max_score_abs_error']}")


if __name__ == "__main__":
    main()
