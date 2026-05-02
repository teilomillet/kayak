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

from kayak_bridge.tachiom_hnsw import TachiomHnswConfig  # noqa: E402
from kayak_bridge.tachiom_pq import TachiomResidualPqConfig  # noqa: E402
from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    StreamingTachiomExactReference,
    benchmark_streaming_tachiom_index,
    build_streaming_tachiom_exact_reference,
)
from kayak_bridge.tachiom_streaming_hnsw import (  # noqa: E402
    build_streaming_tachiom_hnsw_graph,
)
from kayak_bridge.tachiom_streaming_index import (  # noqa: E402
    StreamingTachiomBuildConfig,
    build_streaming_tachiom_index,
)
from kayak_bridge.tachiom_types import TachiomTacConfig  # noqa: E402


HNSW_ENGINES = {
    "streaming_tac_hnsw_pq",
    "streaming_tac_hnsw_pq_mojo",
    "streaming_tac_hnsw_pq_mojo_address",
}
ENGINE_CHOICES = (
    "streaming_tac_pq",
    "streaming_tac_pq_mojo",
    "streaming_tac_hnsw_pq",
    "streaming_tac_hnsw_pq_mojo",
    "streaming_tac_hnsw_pq_mojo_address",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and benchmark a streaming Tachiom centroid-count ladder over "
            "one existing encoded snapshot."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--centroid-counts", required=True)
    parser.add_argument(
        "--engines",
        default="streaming_tac_hnsw_pq_mojo",
        help=f"Comma-separated benchmark engines. Choices: {', '.join(ENGINE_CHOICES)}.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--max-query-batch-size", type=int)
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--max-exact-vector-count", type=int)
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    centroid_counts = _parse_positive_ints(args.centroid_counts, "centroid counts")
    engines = _parse_engines(args.engines)
    if args.max_query_batch_size is not None and args.max_query_batch_size <= 0:
        raise ValueError("max_query_batch_size must be positive when provided")
    output_root = args.output_root.resolve()
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output or output_root / "centroid_ladder_summary.json"

    rows: list[dict[str, object]] = []
    exact_reference: StreamingTachiomExactReference | None = None
    for centroid_count in centroid_counts:
        try:
            row, exact_reference = _run_centroid_row(
                args=args,
                output_root=output_root,
                centroid_count=centroid_count,
                engines=engines,
                exact_reference=exact_reference,
            )
        except Exception as exc:  # noqa: BLE001 - the row is the failure evidence.
            row = {
                "centroid_count": centroid_count,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            rows.append(row)
            _write_payload(args=args, rows=rows, output_path=output_path)
            if not args.continue_on_error:
                break
            continue
        rows.append(row)
        _write_payload(args=args, rows=rows, output_path=output_path)

    payload = _write_payload(args=args, rows=rows, output_path=output_path)
    if args.emit_quiet_mean:
        _emit_quiet_means(payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_centroid_row(
    *,
    args: argparse.Namespace,
    output_root: Path,
    centroid_count: int,
    engines: Sequence[str],
    exact_reference: StreamingTachiomExactReference | None,
) -> tuple[dict[str, object], StreamingTachiomExactReference | None]:
    index_root = output_root / f"streaming_tachiom_index_c{centroid_count}"
    timings: dict[str, float] = {}
    index_manifest = index_root / "manifest.json"
    if index_manifest.exists() and args.reuse_existing:
        index_summary = json.loads(index_manifest.read_text(encoding="utf-8"))
    elif index_root.exists():
        raise FileExistsError(f"index output already exists: {index_root}")
    else:
        index_summary = _timed_stage(
            f"index_c{centroid_count}",
            timings,
            lambda: build_streaming_tachiom_index(
                snapshot_root=args.snapshot,
                output_root=index_root,
                config=_build_config(args=args, centroid_count=centroid_count),
            ).to_json_ready(),
        )

    if any(engine in HNSW_ENGINES for engine in engines):
        graph_manifest = index_root / "hnsw_graph" / "manifest.json"
        if graph_manifest.exists() and args.reuse_existing:
            hnsw_summary = json.loads(graph_manifest.read_text(encoding="utf-8"))
        else:
            hnsw_summary = _timed_stage(
                f"hnsw_c{centroid_count}",
                timings,
                lambda: build_streaming_tachiom_hnsw_graph(
                    index_root=index_root,
                    config=TachiomHnswConfig(
                        max_neighbors=args.hnsw_max_neighbors,
                        ef_construction=args.hnsw_ef_construction,
                        ef_search=args.hnsw_ef_search,
                        level_probability=args.hnsw_level_probability,
                        seed=args.hnsw_seed,
                    ),
                ).to_json_ready(),
            )
    else:
        hnsw_summary = None

    if args.run_exact and exact_reference is None:
        exact_reference = _timed_stage(
            "exact_reference",
            timings,
            lambda: build_streaming_tachiom_exact_reference(
                snapshot_root=args.snapshot,
                index_root=index_root,
                query_limit=args.query_limit,
                max_exact_vector_count=args.max_exact_vector_count,
            ),
        )

    benchmarks = {}
    for engine in engines:
        benchmarks[engine] = _timed_stage(
            f"benchmark_{engine}_c{centroid_count}",
            timings,
            lambda engine=engine: benchmark_streaming_tachiom_index(
                snapshot_root=args.snapshot,
                index_root=index_root,
                query_limit=args.query_limit,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                run_exact=args.run_exact,
                max_exact_vector_count=args.max_exact_vector_count,
                engine=engine,
                max_query_batch_size=args.max_query_batch_size,
                exact_reference=exact_reference,
            ).to_json_ready(),
        )

    return (
        {
            "centroid_count": centroid_count,
            "status": "completed",
            "stage_timings_seconds": timings,
            "index": index_summary,
            "hnsw_graph": hnsw_summary,
            "benchmarks": benchmarks,
        },
        exact_reference,
    )


def _build_config(
    *,
    args: argparse.Namespace,
    centroid_count: int,
) -> StreamingTachiomBuildConfig:
    return StreamingTachiomBuildConfig(
        tac=TachiomTacConfig(
            centroid_count=centroid_count,
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
        ),
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
    )


def _write_payload(
    *,
    args: argparse.Namespace,
    rows: Sequence[dict[str, object]],
    output_path: Path,
) -> dict[str, object]:
    payload = {
        "snapshot": str(args.snapshot),
        "output_root": str(args.output_root.resolve()),
        "centroid_counts": _parse_positive_ints(
            args.centroid_counts,
            "centroid counts",
        ),
        "engines": _parse_engines(args.engines),
        "run_exact": args.run_exact,
        "max_query_batch_size": args.max_query_batch_size,
        "rows": list(rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _emit_quiet_means(payload: dict[str, object]) -> None:
    for row in payload["rows"]:
        if row["status"] != "completed":
            continue
        centroid_count = row["centroid_count"]
        for engine, summary in row["benchmarks"].items():
            print(f"== c{centroid_count}_{engine} ==")
            print("Mean:", summary["query_batch_mean_seconds"])
    completed = [row for row in payload["rows"] if row["status"] == "completed"]
    print("quiet_context " f"completed_rows={len(completed)}")


def _parse_positive_ints(value: str, label: str) -> list[int]:
    sizes: list[int] = []
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError(f"{label} must be positive")
        if parsed not in sizes:
            sizes.append(parsed)
    if not sizes:
        raise ValueError(f"at least one {label} value is required")
    return sizes


def _parse_engines(value: str) -> list[str]:
    engines: list[str] = []
    for raw in value.split(","):
        engine = raw.strip()
        if not engine:
            continue
        if engine not in ENGINE_CHOICES:
            raise ValueError(f"unsupported engine: {engine}")
        if engine not in engines:
            engines.append(engine)
    if not engines:
        raise ValueError("at least one engine is required")
    return engines


def _timed_stage(
    name: str,
    timings: dict[str, float],
    callback: Callable[[], object],
) -> object:
    print(f"[centroid-ladder] start {name}", flush=True)
    started_at = time.perf_counter()
    result = callback()
    elapsed = time.perf_counter() - started_at
    timings[name] = elapsed
    print(f"[centroid-ladder] done {name} seconds={elapsed:.3f}", flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
