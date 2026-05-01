from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_hnsw import TachiomHnswConfig  # noqa: E402
from kayak_bridge.tachiom_pq import TachiomResidualPqConfig  # noqa: E402
from kayak_bridge.tachiom_snapshot_benchmark import (  # noqa: E402
    benchmark_snapshot_with_tachiom,
)
from kayak_bridge.tachiom_types import TachiomTacConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Tachiom directly from a sharded encoded snapshot. This "
            "path avoids task JSON and Python vector-list materialization for "
            "the selected documents."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--engine",
        choices=(
            "tachiom_tac",
            "tachiom_tac_hnsw",
            "tachiom_tac_pq",
            "tachiom_tac_hnsw_pq",
        ),
        default="tachiom_tac_hnsw_pq",
    )
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument(
        "--max-vector-count",
        type=int,
        default=2_000_000,
        help=(
            "Safety cap for the current in-memory TAC reference path. Full "
            "paper-scale snapshots require a streaming builder."
        ),
    )
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--tac-centroid-count", type=int, default=2048)
    parser.add_argument("--tac-centroids-per-query-vector", type=int, default=16)
    parser.add_argument("--tac-candidate-k", type=int, default=128)
    parser.add_argument("--tac-micro-token-threshold", type=int, default=4)
    parser.add_argument("--tac-small-token-threshold", type=int, default=16)
    parser.add_argument("--tac-active-token-floor", type=int, default=1)
    parser.add_argument("--tac-min-vectors-per-centroid", type=int, default=4)
    parser.add_argument("--tac-kmeans-iterations", type=int, default=6)
    parser.add_argument("--tac-candidate-pruning-alpha", type=float, default=0.0)
    parser.add_argument("--hnsw-max-neighbors", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-level-probability", type=float, default=0.0625)
    parser.add_argument("--hnsw-seed", type=int, default=7)
    parser.add_argument("--pq-subspace-count", type=int, default=32)
    parser.add_argument("--pq-codebook-size", type=int, default=256)
    parser.add_argument("--pq-kmeans-iterations", type=int, default=6)
    parser.add_argument("--pq-training-sample-count", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/tachiom_snapshot/summary.json"),
    )
    parser.add_argument("--emit-quiet-mean", action="store_true")
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
    hnsw_config = TachiomHnswConfig(
        max_neighbors=args.hnsw_max_neighbors,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
        level_probability=args.hnsw_level_probability,
        seed=args.hnsw_seed,
    )
    pq_config = TachiomResidualPqConfig(
        subspace_count=args.pq_subspace_count,
        codebook_size=args.pq_codebook_size,
        kmeans_iterations=args.pq_kmeans_iterations,
        training_sample_count=(
            None
            if args.pq_training_sample_count <= 0
            else args.pq_training_sample_count
        ),
    )
    summary = benchmark_snapshot_with_tachiom(
        args.snapshot,
        engine=args.engine,
        tac_config=tac_config,
        hnsw_config=hnsw_config if "hnsw" in args.engine else None,
        pq_config=pq_config if "pq" in args.engine else None,
        document_limit=args.document_limit,
        query_limit=args.query_limit,
        max_vector_count=args.max_vector_count,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        run_exact=not args.skip_exact,
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "summary": summary.to_json_ready(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        print("== " + args.engine + " / tachiom_snapshot ==")
        print("Mean:", summary.query_batch_mean_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
