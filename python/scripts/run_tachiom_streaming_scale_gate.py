from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.colbert_encoder import DEFAULT_MODEL_NAME  # noqa: E402
from kayak_bridge.msmarco_colbert_snapshot import (  # noqa: E402
    DEFAULT_DATASET_ID,
    build_msmarco_colbert_snapshot,
)
from kayak_bridge.msmarco_passage_task import MsmarcoPassagePaths  # noqa: E402
from kayak_bridge.tachiom_hnsw import TachiomHnswConfig  # noqa: E402
from kayak_bridge.tachiom_pq import TachiomResidualPqConfig  # noqa: E402
from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    benchmark_streaming_tachiom_index,
)
from kayak_bridge.tachiom_streaming_hnsw import (  # noqa: E402
    build_streaming_tachiom_hnsw_graph,
)
from kayak_bridge.tachiom_streaming_index import (  # noqa: E402
    StreamingTachiomBuildConfig,
    build_streaming_tachiom_index,
)
from kayak_bridge.tachiom_types import TachiomTacConfig  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a judged-positive MS MARCO streaming Tachiom scale gate: "
            "snapshot, TAC/PQ index, optional HNSW graph, and benchmarks."
        )
    )
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--document-limit", type=int, required=True)
    parser.add_argument("--query-limit", type=int, required=True)
    parser.add_argument("--document-batch-size", type=int, default=16)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--shard-max-vectors", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-hnsw", action="store_true")
    parser.add_argument("--skip-mojo", action="store_true")
    parser.add_argument("--include-address-mojo", action="store_true")
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--max-query-batch-size",
        type=int,
        help=(
            "Cap same-shape benchmark query batches. Use this with the USL sweep "
            "result when a smaller batch is measured to scale better."
        ),
    )
    parser.add_argument("--max-exact-vector-count", type=int)
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
    parser.add_argument("--assignment-vector-chunk-size", type=int, default=1024)
    parser.add_argument("--assignment-centroid-chunk-size", type=int, default=2048)
    parser.add_argument("--posting-partition-count", type=int, default=128)
    parser.add_argument("--pq-subspace-count", type=int, default=32)
    parser.add_argument("--pq-codebook-size", type=int, default=256)
    parser.add_argument("--pq-kmeans-iterations", type=int, default=10)
    parser.add_argument("--pq-training-sample-count", type=int, default=32768)
    parser.add_argument("--hnsw-max-neighbors", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-level-probability", type=float, default=0.0625)
    parser.add_argument("--hnsw-seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"scale gate output is not empty; use --overwrite: {output_root}"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot_root = output_root / "snapshot"
    index_root = output_root / f"streaming_tachiom_index_c{args.tac_centroid_count}"
    summary_path = output_root / "scale_gate_summary.json"
    stage_timings: dict[str, float] = {}

    snapshot_manifest = _timed_stage(
        "snapshot",
        stage_timings,
        lambda: build_msmarco_colbert_snapshot(
            paths=MsmarcoPassagePaths(
                collection=args.collection,
                queries=args.queries,
                qrels=args.qrels,
            ),
            output=snapshot_root,
            document_limit=args.document_limit,
            query_limit=args.query_limit,
            dataset_id=args.dataset_id,
            model_name=args.model_name,
            document_batch_size=args.document_batch_size,
            query_batch_size=args.query_batch_size,
            shard_max_vectors=args.shard_max_vectors,
            include_query_positives=True,
        ),
    )

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
    index_summary = _timed_stage(
        "streaming_index",
        stage_timings,
        lambda: build_streaming_tachiom_index(
            snapshot_root=snapshot_root,
            output_root=index_root,
            config=StreamingTachiomBuildConfig(
                tac=tac_config,
                pq=TachiomResidualPqConfig(
                    subspace_count=args.pq_subspace_count,
                    codebook_size=args.pq_codebook_size,
                    kmeans_iterations=args.pq_kmeans_iterations,
                    training_sample_count=args.pq_training_sample_count,
                ),
                centroid_samples_per_centroid=args.centroid_samples_per_centroid,
                kmeans_max_centroids_per_token=args.kmeans_max_centroids_per_token,
                kmeans_max_samples_per_token=args.kmeans_max_samples_per_token,
                assignment_vector_chunk_size=args.assignment_vector_chunk_size,
                assignment_centroid_chunk_size=args.assignment_centroid_chunk_size,
                posting_partition_count=args.posting_partition_count,
            ),
        ),
    )
    exact_centroid_summary = _timed_stage(
        "benchmark_streaming_tac_pq",
        stage_timings,
        lambda: benchmark_streaming_tachiom_index(
            snapshot_root=snapshot_root,
            index_root=index_root,
            query_limit=args.query_limit,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            run_exact=True,
            max_exact_vector_count=args.max_exact_vector_count,
            engine="streaming_tac_pq",
            max_query_batch_size=args.max_query_batch_size,
        ),
    )
    mojo_summary = None
    if not args.skip_mojo:
        mojo_summary = _timed_stage(
            "benchmark_streaming_tac_pq_mojo",
            stage_timings,
            lambda: benchmark_streaming_tachiom_index(
                snapshot_root=snapshot_root,
                index_root=index_root,
                query_limit=args.query_limit,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                run_exact=True,
                max_exact_vector_count=args.max_exact_vector_count,
                engine="streaming_tac_pq_mojo",
                max_query_batch_size=args.max_query_batch_size,
            ),
        )

    hnsw_summary = None
    hnsw_benchmark = None
    hnsw_mojo_benchmark = None
    hnsw_mojo_address_benchmark = None
    if not args.skip_hnsw:
        hnsw_summary = _timed_stage(
            "hnsw_graph",
            stage_timings,
            lambda: build_streaming_tachiom_hnsw_graph(
                index_root=index_root,
                config=TachiomHnswConfig(
                    max_neighbors=args.hnsw_max_neighbors,
                    ef_construction=args.hnsw_ef_construction,
                    ef_search=args.hnsw_ef_search,
                    level_probability=args.hnsw_level_probability,
                    seed=args.hnsw_seed,
                ),
            ),
        )
        hnsw_benchmark = _timed_stage(
            "benchmark_streaming_tac_hnsw_pq",
            stage_timings,
            lambda: benchmark_streaming_tachiom_index(
                snapshot_root=snapshot_root,
                index_root=index_root,
                query_limit=args.query_limit,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                run_exact=True,
                max_exact_vector_count=args.max_exact_vector_count,
                engine="streaming_tac_hnsw_pq",
                max_query_batch_size=args.max_query_batch_size,
            ),
        )
        if not args.skip_mojo:
            hnsw_mojo_benchmark = _timed_stage(
                "benchmark_streaming_tac_hnsw_pq_mojo",
                stage_timings,
                lambda: benchmark_streaming_tachiom_index(
                    snapshot_root=snapshot_root,
                    index_root=index_root,
                    query_limit=args.query_limit,
                    warmup_iterations=args.warmup_iterations,
                    measurement_iterations=args.measurement_iterations,
                    run_exact=True,
                    max_exact_vector_count=args.max_exact_vector_count,
                    engine="streaming_tac_hnsw_pq_mojo",
                    max_query_batch_size=args.max_query_batch_size,
                ),
            )
            if args.include_address_mojo:
                hnsw_mojo_address_benchmark = _timed_stage(
                    "benchmark_streaming_tac_hnsw_pq_mojo_address",
                    stage_timings,
                    lambda: benchmark_streaming_tachiom_index(
                        snapshot_root=snapshot_root,
                        index_root=index_root,
                        query_limit=args.query_limit,
                        warmup_iterations=args.warmup_iterations,
                        measurement_iterations=args.measurement_iterations,
                        run_exact=True,
                        max_exact_vector_count=args.max_exact_vector_count,
                        engine="streaming_tac_hnsw_pq_mojo_address",
                        max_query_batch_size=args.max_query_batch_size,
                    ),
                )

    summary = {
        "stage_timings_seconds": stage_timings,
        "snapshot": snapshot_manifest.to_json_ready(),
        "index": index_summary.to_json_ready(),
        "exact_centroid_benchmark": exact_centroid_summary.to_json_ready(),
        "mojo_exact_centroid_benchmark": (
            None if mojo_summary is None else mojo_summary.to_json_ready()
        ),
        "hnsw_graph": None if hnsw_summary is None else hnsw_summary.to_json_ready(),
        "hnsw_benchmark": (
            None if hnsw_benchmark is None else hnsw_benchmark.to_json_ready()
        ),
        "hnsw_mojo_benchmark": (
            None
            if hnsw_mojo_benchmark is None
            else hnsw_mojo_benchmark.to_json_ready()
        ),
        "hnsw_mojo_address_benchmark": (
            None
            if hnsw_mojo_address_benchmark is None
            else hnsw_mojo_address_benchmark.to_json_ready()
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_compact_summary(summary), indent=2, sort_keys=True))
    print(f"wrote {summary_path}")
    return 0


def _compact_summary(summary: dict[str, object]) -> dict[str, object]:
    snapshot = summary["snapshot"]
    index = summary["index"]
    exact = summary["exact_centroid_benchmark"]
    mojo = summary["mojo_exact_centroid_benchmark"]
    hnsw = summary["hnsw_benchmark"]
    hnsw_mojo = summary["hnsw_mojo_benchmark"]
    hnsw_mojo_address = summary["hnsw_mojo_address_benchmark"]
    row: dict[str, object] = {
        "document_count": snapshot["document_count"],
        "document_vector_count": snapshot["document_vector_count"],
        "query_count": snapshot["query_manifest"]["query_count"],
        "query_vector_count": snapshot["query_manifest"]["query_vector_count"],
        "required_positive_doc_count": snapshot["source"][
            "required_positive_doc_count"
        ],
        "centroid_count": index["centroid_count"],
        "index_bytes": index["index_payload_bytes"],
        "stage_timings_seconds": summary["stage_timings_seconds"],
        "exact_centroid": _benchmark_compact(exact),
    }
    if mojo is not None:
        row["mojo_exact_centroid"] = _benchmark_compact(mojo)
    if hnsw is not None:
        row["hnsw"] = _benchmark_compact(hnsw)
    if hnsw_mojo is not None:
        row["hnsw_mojo"] = _benchmark_compact(hnsw_mojo)
    if hnsw_mojo_address is not None:
        row["hnsw_mojo_address"] = _benchmark_compact(hnsw_mojo_address)
    return row


def _benchmark_compact(row: dict[str, object]) -> dict[str, object]:
    return {
        "primary_metric": row["primary_metric"],
        "primary_value": row["primary_value"],
        "exact_primary_value": row["exact_primary_value"],
        "candidate_recall_at_k_vs_exact": row["candidate_recall_at_k_vs_exact"],
        "final_recall_at_k_vs_exact": row["final_recall_at_k_vs_exact"],
        "query_qps": row["query_qps"],
        "max_query_batch_size": row["max_query_batch_size"],
        "index_bytes": row["index_bytes"],
    }


def _timed_stage(
    name: str,
    timings: dict[str, float],
    callback: Callable[[], object],
) -> object:
    print(f"[scale-gate] start {name}", flush=True)
    started_at = time.perf_counter()
    result = callback()
    elapsed = time.perf_counter() - started_at
    timings[name] = elapsed
    print(f"[scale-gate] done {name} seconds={elapsed:.3f}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
