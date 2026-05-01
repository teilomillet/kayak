from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_pq import TachiomResidualPqConfig  # noqa: E402
from kayak_bridge.tachiom_streaming_index import (  # noqa: E402
    StreamingTachiomBuildConfig,
    build_streaming_tachiom_index,
)
from kayak_bridge.tachiom_types import TachiomTacConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a streaming TAC/PQ artifact from a sharded encoded snapshot. "
            "This is the paper-scale path that avoids task JSON and a full "
            "in-memory token matrix."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-pq", action="store_true")
    parser.add_argument("--tac-centroid-count", type=int, default=32768)
    parser.add_argument("--tac-micro-token-threshold", type=int, default=128)
    parser.add_argument("--tac-small-token-threshold", type=int, default=256)
    parser.add_argument("--tac-active-token-floor", type=int, default=4)
    parser.add_argument("--tac-min-vectors-per-centroid", type=int, default=39)
    parser.add_argument("--tac-kmeans-iterations", type=int, default=10)
    parser.add_argument("--tac-centroids-per-query-vector", type=int, default=120)
    parser.add_argument("--tac-candidate-k", type=int, default=1000)
    parser.add_argument("--tac-candidate-pruning-alpha", type=float, default=0.35)
    parser.add_argument("--centroid-samples-per-centroid", type=int, default=4)
    parser.add_argument("--kmeans-max-centroids-per-token", type=int, default=256)
    parser.add_argument("--kmeans-max-samples-per-token", type=int, default=4096)
    parser.add_argument("--assignment-vector-chunk-size", type=int, default=2048)
    parser.add_argument("--assignment-centroid-chunk-size", type=int, default=4096)
    parser.add_argument("--posting-partition-count", type=int, default=128)
    parser.add_argument("--pq-subspace-count", type=int, default=32)
    parser.add_argument("--pq-codebook-size", type=int, default=256)
    parser.add_argument("--pq-kmeans-iterations", type=int, default=10)
    parser.add_argument("--pq-training-sample-count", type=int, default=32768)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tac_config = TachiomTacConfig(
        centroid_count=args.tac_centroid_count,
        micro_token_threshold=args.tac_micro_token_threshold,
        small_token_threshold=args.tac_small_token_threshold,
        active_token_floor=args.tac_active_token_floor,
        min_vectors_per_centroid=args.tac_min_vectors_per_centroid,
        kmeans_iterations=args.tac_kmeans_iterations,
        centroids_per_query_vector=args.tac_centroids_per_query_vector,
        candidate_k=args.tac_candidate_k,
        candidate_pruning_alpha=(
            None
            if args.tac_candidate_pruning_alpha <= 0.0
            else args.tac_candidate_pruning_alpha
        ),
    )
    pq_config = None
    if not args.disable_pq:
        pq_config = TachiomResidualPqConfig(
            subspace_count=args.pq_subspace_count,
            codebook_size=args.pq_codebook_size,
            kmeans_iterations=args.pq_kmeans_iterations,
            training_sample_count=args.pq_training_sample_count,
        )
    build_config = StreamingTachiomBuildConfig(
        tac=tac_config,
        pq=pq_config,
        centroid_samples_per_centroid=args.centroid_samples_per_centroid,
        kmeans_max_centroids_per_token=args.kmeans_max_centroids_per_token,
        kmeans_max_samples_per_token=args.kmeans_max_samples_per_token,
        assignment_vector_chunk_size=args.assignment_vector_chunk_size,
        assignment_centroid_chunk_size=args.assignment_centroid_chunk_size,
        posting_partition_count=args.posting_partition_count,
    )
    summary = build_streaming_tachiom_index(
        snapshot_root=args.snapshot,
        output_root=args.output,
        config=build_config,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_json_ready(), indent=2, sort_keys=True))
    print(f"wrote {args.output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
