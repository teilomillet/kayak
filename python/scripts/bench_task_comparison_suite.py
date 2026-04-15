from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.benchmark_variance import (
    freeze_benchmark_summary_mean,
    summarize_benchmark_variance,
)
from kayak_bridge.comparison_scorecard import build_comparison_scorecard
from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.kayak_task_benchmark import benchmark_task_with_kayak_exact
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb
from kayak_bridge.lancedb_index_controls import (
    LanceDbIndexBuildControls,
    LanceDbIndexedQueryControls,
)
from kayak_bridge.lancedb_storage_comparison import (
    benchmark_task_with_lancedb_storage_compare,
)
from kayak_bridge.scale_comparison import benchmark_same_task_scale_sweep
from kayak_bridge.storage_scale_comparison import (
    benchmark_storage_engine_scale_sweep,
)
from kayak_bridge.task_comparison_bundle import build_task_comparison_bundle
from kayak_bridge.task_storage_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def _default_target_document_counts(task: dict[str, object]) -> list[int]:
    base_document_count = len(task["documents"])
    return [
        base_document_count,
        base_document_count * 2,
        base_document_count * 4,
        base_document_count * 8,
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a generic Kayak-versus-LanceDB comparison suite on one encoded task JSON."
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
    parser.add_argument("--rebuild-count", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--index-num-partitions", type=int)
    parser.add_argument("--index-num-sub-vectors", type=int)
    parser.add_argument("--index-target-partition-size", type=int)
    parser.add_argument("--indexed-nprobes", type=int)
    parser.add_argument("--indexed-refine-factor", type=int)
    parser.add_argument(
        "--target-document-count",
        type=int,
        action="append",
        dest="target_document_counts",
        help="Target document count for scale sweeps. Repeat to add more.",
    )
    parser.add_argument(
        "--skip-scale-sweep",
        action="store_true",
        help="Skip the Kayak-versus-LanceDB scale sweep.",
    )
    parser.add_argument(
        "--skip-storage-compare",
        action="store_true",
        help="Skip the same-storage LanceDB-versus-Kayak search comparison.",
    )
    parser.add_argument(
        "--skip-storage-scale",
        action="store_true",
        help="Skip the native Kayak storage-versus-LanceDB storage scale sweep.",
    )
    parser.add_argument(
        "--kayak-vector-payload-encoding",
        type=str,
        default=VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        choices=(
            VECTOR_PAYLOAD_ENCODING_BINARY_LE,
            VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
        ),
        help="Native Kayak packed-index vector payload encoding for storage sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_count <= 0:
        raise ValueError("rebuild-count must be positive")

    task = load_task_json(str(args.task))
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))
    table_name = artifact_prefix
    index_build_controls = LanceDbIndexBuildControls(
        num_partitions=args.index_num_partitions,
        num_sub_vectors=args.index_num_sub_vectors,
        target_partition_size=args.index_target_partition_size,
    )
    indexed_query_controls = LanceDbIndexedQueryControls(
        nprobes=args.indexed_nprobes,
        refine_factor=args.indexed_refine_factor,
    )
    target_document_counts = args.target_document_counts
    if not target_document_counts:
        target_document_counts = _default_target_document_counts(task)

    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    lancedb_scan_path = output_root / f"{artifact_prefix}_lancedb_scan_benchmark.json"
    lancedb_indexed_variance_path = (
        output_root / f"{artifact_prefix}_lancedb_ivf_pq_variance.json"
    )
    lancedb_indexed_frozen_path = (
        output_root / f"{artifact_prefix}_lancedb_ivf_pq_frozen_benchmark.json"
    )
    scorecard_path = output_root / f"{artifact_prefix}_comparison_scorecard.json"
    scale_sweep_path = output_root / f"{artifact_prefix}_scale_sweep.json"
    storage_compare_path = output_root / f"{artifact_prefix}_storage_compare.json"
    storage_scale_path = output_root / f"{artifact_prefix}_storage_scale.json"
    bundle_path = output_root / f"{artifact_prefix}_comparison_bundle.json"

    kayak_exact = benchmark_task_with_kayak_exact(
        task,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    ).to_json_ready()
    _write_json(kayak_exact_path, kayak_exact)

    lancedb_scan = benchmark_task_with_lancedb(
        task=task,
        database_root=output_root / f"{artifact_prefix}_lancedb_scan",
        table_name=table_name,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        build_index=False,
    ).to_json_ready()
    _write_json(lancedb_scan_path, lancedb_scan)

    indexed_runs = []
    for rebuild_index in range(args.rebuild_count):
        indexed_runs.append(
            benchmark_task_with_lancedb(
                task=task,
                database_root=output_root
                / f"{artifact_prefix}_lancedb_ivf_pq_{rebuild_index:02d}",
                table_name=table_name,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                build_index=True,
                index_build_controls=index_build_controls,
                indexed_query_controls=indexed_query_controls,
            ).to_json_ready()
        )
    indexed_variance = summarize_benchmark_variance(indexed_runs).to_json_ready()
    _write_json(lancedb_indexed_variance_path, indexed_variance)

    indexed_frozen = freeze_benchmark_summary_mean(
        indexed_variance["run_summaries"],
        freeze_policy=f"mean_across_{args.rebuild_count}_rebuilds",
    ).to_json_ready()
    _write_json(lancedb_indexed_frozen_path, indexed_frozen)

    scorecard = build_comparison_scorecard(
        systems=(
            ("kayak_exact", str(kayak_exact_path), kayak_exact),
            ("lancedb_scan", str(lancedb_scan_path), lancedb_scan),
            (
                "lancedb_ivf_pq_frozen",
                str(lancedb_indexed_frozen_path),
                indexed_frozen,
            ),
        ),
        baseline_name="kayak_exact",
    ).to_json_ready()
    _write_json(scorecard_path, scorecard)

    scale_sweep = None
    if not args.skip_scale_sweep:
        scale_sweep = benchmark_same_task_scale_sweep(
            task,
            database_root=output_root / f"{artifact_prefix}_lancedb_scale_sweep",
            table_prefix=table_name,
            target_document_counts=target_document_counts,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        ).to_json_ready()
        _write_json(scale_sweep_path, scale_sweep)

    storage_compare = None
    if not args.skip_storage_compare:
        storage_compare = benchmark_task_with_lancedb_storage_compare(
            task=task,
            database_root=output_root / f"{artifact_prefix}_lancedb_storage_compare",
            table_name=table_name,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        ).to_json_ready()
        _write_json(storage_compare_path, storage_compare)

    storage_scale = None
    if not args.skip_storage_scale:
        storage_scale = benchmark_storage_engine_scale_sweep(
            task,
            database_root=output_root / f"{artifact_prefix}_lancedb_storage_scale",
            table_prefix=table_name,
            target_document_counts=target_document_counts,
            kayak_vector_payload_encoding=args.kayak_vector_payload_encoding,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        ).to_json_ready()
        _write_json(storage_scale_path, storage_scale)

    bundle = build_task_comparison_bundle(
        task_path=str(args.task),
        kayak_exact_path=str(kayak_exact_path),
        lancedb_scan_path=str(lancedb_scan_path),
        lancedb_indexed_variance_path=str(lancedb_indexed_variance_path),
        lancedb_indexed_frozen_path=str(lancedb_indexed_frozen_path),
        scorecard_path=str(scorecard_path),
        scorecard=scorecard,
        indexed_variance=indexed_variance,
        indexed_frozen=indexed_frozen,
        scale_sweep_path=(
            None if scale_sweep is None else str(scale_sweep_path)
        ),
        scale_sweep=scale_sweep,
        storage_compare_path=(
            None if storage_compare is None else str(storage_compare_path)
        ),
        storage_compare=storage_compare,
        storage_scale_path=(
            None if storage_scale is None else str(storage_scale_path)
        ),
        storage_scale=storage_scale,
    ).to_json_ready()
    _write_json(bundle_path, bundle)

    print(f"wrote {kayak_exact_path}")
    print(f"wrote {lancedb_scan_path}")
    print(f"wrote {lancedb_indexed_variance_path}")
    print(f"wrote {lancedb_indexed_frozen_path}")
    print(f"wrote {scorecard_path}")
    if scale_sweep is not None:
        print(f"wrote {scale_sweep_path}")
    if storage_compare is not None:
        print(f"wrote {storage_compare_path}")
    if storage_scale is not None:
        print(f"wrote {storage_scale_path}")
    print(f"wrote {bundle_path}")
    print(f"Mean: {indexed_frozen['mean_search_seconds']}")


if __name__ == "__main__":
    main()
