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
from kayak_bridge.kayak_task_benchmark import benchmark_task_with_kayak_exact
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb
from kayak_bridge.lancedb_index_sweep import (
    LanceDbIndexedSweepSpec,
    build_lancedb_indexed_sweep_summary,
    parse_lancedb_indexed_sweep_spec,
)
from kayak_bridge.lancedb_indexed_candidate_bundle import (
    build_lancedb_indexed_candidate_bundle,
)
from kayak_bridge.lancedb_indexed_matrix import (
    build_lancedb_indexed_matrix_summary,
)
from kayak_bridge.lancedb_indexed_selection import (
    select_lancedb_indexed_candidates_from_sweep_summary,
)
from kayak_bridge.task_json_catalog import (
    build_named_task_json,
    default_task_json_output_path,
    load_task_json,
    write_task_json,
)


DEFAULT_DATASET_KEYS = (
    "bright_stackoverflow_real_subset",
    "lemb_narrativeqa_real_subset",
    "legal_rag_bench_real_subset",
    "r2med_biology_real_subset",
)

DEFAULT_SWEEP_CONFIGS = (
    "default",
    "nprobe64:indexed_nprobes=64",
    "refine1:indexed_refine_factor=1",
    "refine2:indexed_refine_factor=2",
    "refine4:indexed_refine_factor=4",
    "p4_refine2:index_num_partitions=4,indexed_refine_factor=2",
    "p8_refine2:index_num_partitions=8,indexed_refine_factor=2",
    "sv16_refine2:index_num_sub_vectors=16,indexed_refine_factor=2",
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _ensure_task_json(dataset_key: str) -> tuple[Path, dict[str, object]]:
    task_path = default_task_json_output_path(dataset_key)
    if task_path.exists():
        return task_path, load_task_json(task_path)
    task = build_named_task_json(dataset_key)
    write_task_json(task_path, task)
    return task_path, task


def _run_indexed_sweep(
    *,
    task: dict[str, object],
    task_path: Path,
    output_root: Path,
    artifact_prefix: str,
    specs: tuple[LanceDbIndexedSweepSpec, ...],
    rebuild_count: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> tuple[str, dict[str, object]]:
    table_name = artifact_prefix
    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    lancedb_scan_path = output_root / f"{artifact_prefix}_lancedb_scan_benchmark.json"
    summary_path = output_root / f"{artifact_prefix}_indexed_sweep_summary.json"

    kayak_exact = benchmark_task_with_kayak_exact(
        task,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    ).to_json_ready()
    _write_json(kayak_exact_path, kayak_exact)

    lancedb_scan = benchmark_task_with_lancedb(
        task=task,
        database_root=output_root / f"{artifact_prefix}_lancedb_scan",
        table_name=table_name,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        build_index=False,
    ).to_json_ready()
    _write_json(lancedb_scan_path, lancedb_scan)

    config_payloads = []
    for spec in specs:
        runs = []
        for rebuild_index in range(rebuild_count):
            summary = benchmark_task_with_lancedb(
                task=task,
                database_root=output_root
                / f"{artifact_prefix}_{spec.name}_{rebuild_index:02d}",
                table_name=f"{table_name}_{spec.name}",
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                build_index=True,
                index_build_controls=spec.index_build_controls,
                indexed_query_controls=spec.indexed_query_controls,
            )
            runs.append(summary.to_json_ready())

        variance = summarize_benchmark_variance(runs).to_json_ready()
        variance_path = output_root / f"{artifact_prefix}_{spec.name}_variance.json"
        _write_json(variance_path, variance)

        frozen = freeze_benchmark_summary_mean(
            variance["run_summaries"],
            freeze_policy=f"mean_across_{rebuild_count}_rebuilds",
        ).to_json_ready()
        frozen_path = output_root / f"{artifact_prefix}_{spec.name}_frozen.json"
        _write_json(frozen_path, frozen)

        config_payloads.append(
            (spec.name, str(variance_path), variance, str(frozen_path), frozen)
        )

    sweep_summary = build_lancedb_indexed_sweep_summary(
        task_path=str(task_path),
        kayak_exact_path=str(kayak_exact_path),
        kayak_exact=kayak_exact,
        lancedb_scan_path=str(lancedb_scan_path),
        lancedb_scan=lancedb_scan,
        configs=config_payloads,
    ).to_json_ready()
    _write_json(summary_path, sweep_summary)
    return str(summary_path), sweep_summary


def _run_selected_candidates(
    *,
    task: dict[str, object],
    task_path: Path,
    output_root: Path,
    artifact_prefix: str,
    sweep_summary: dict[str, object],
    rebuild_count: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> tuple[str, dict[str, object], str, dict[str, object]]:
    table_name = artifact_prefix
    selection_summary, specs = select_lancedb_indexed_candidates_from_sweep_summary(
        sweep_summary
    )

    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    lancedb_scan_path = output_root / f"{artifact_prefix}_lancedb_scan_benchmark.json"
    scorecard_path = output_root / f"{artifact_prefix}_comparison_scorecard.json"
    bundle_path = output_root / f"{artifact_prefix}_candidate_bundle.json"
    selection_path = output_root / f"{artifact_prefix}_candidate_selection.json"

    kayak_exact = benchmark_task_with_kayak_exact(
        task,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    ).to_json_ready()
    _write_json(kayak_exact_path, kayak_exact)

    lancedb_scan = benchmark_task_with_lancedb(
        task=task,
        database_root=output_root / f"{artifact_prefix}_lancedb_scan",
        table_name=table_name,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        build_index=False,
    ).to_json_ready()
    _write_json(lancedb_scan_path, lancedb_scan)

    systems = [
        ("kayak_exact", str(kayak_exact_path), kayak_exact),
        ("lancedb_scan", str(lancedb_scan_path), lancedb_scan),
    ]
    candidates = []
    for spec in specs:
        system_name = f"lancedb_indexed_{spec.name}"
        runs = []
        for rebuild_index in range(rebuild_count):
            summary = benchmark_task_with_lancedb(
                task=task,
                database_root=output_root
                / f"{artifact_prefix}_{spec.name}_{rebuild_index:02d}",
                table_name=f"{table_name}_{spec.name}",
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                build_index=True,
                index_build_controls=spec.index_build_controls,
                indexed_query_controls=spec.indexed_query_controls,
            )
            runs.append(summary.to_json_ready())

        variance = summarize_benchmark_variance(runs).to_json_ready()
        variance_path = output_root / f"{artifact_prefix}_{spec.name}_variance.json"
        _write_json(variance_path, variance)

        frozen = freeze_benchmark_summary_mean(
            variance["run_summaries"],
            freeze_policy=f"mean_across_{rebuild_count}_rebuilds",
        ).to_json_ready()
        frozen_path = output_root / f"{artifact_prefix}_{spec.name}_frozen.json"
        _write_json(frozen_path, frozen)

        systems.append((system_name, str(frozen_path), frozen))
        candidates.append((system_name, str(frozen_path), frozen))

    scorecard = build_comparison_scorecard(
        systems=systems,
        baseline_name="kayak_exact",
    ).to_json_ready()
    _write_json(scorecard_path, scorecard)

    bundle = build_lancedb_indexed_candidate_bundle(
        task_path=str(task_path),
        kayak_exact_path=str(kayak_exact_path),
        lancedb_scan_path=str(lancedb_scan_path),
        scorecard_path=str(scorecard_path),
        scorecard=scorecard,
        candidates=candidates,
    ).to_json_ready()
    _write_json(bundle_path, bundle)
    _write_json(selection_path, selection_summary.to_json_ready())
    return (
        str(selection_path),
        selection_summary.to_json_ready(),
        str(bundle_path),
        bundle,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LanceDB hard-slice workflow across multiple task JSONs: "
            "indexed sweep, auto-selection, and compact rerun."
        )
    )
    parser.add_argument(
        "--dataset-key",
        action="append",
        dest="dataset_keys",
        help="Named task-json key. Repeat to add more datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".cache" / "kayak" / "lancedb_hard_matrix",
        help="Directory where matrix artifacts should be written.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        default="lancedb_hard_matrix",
        help="Artifact filename prefix for the matrix summary.",
    )
    parser.add_argument("--rebuild-count", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Indexed sweep config in `name:key=value,key=value` form. "
            "Defaults to the repo hard-slice sweep set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_count <= 0:
        raise ValueError("rebuild-count must be positive")

    dataset_keys = tuple(args.dataset_keys or DEFAULT_DATASET_KEYS)
    raw_configs = tuple(args.config or DEFAULT_SWEEP_CONFIGS)
    specs = tuple(parse_lancedb_indexed_sweep_spec(value) for value in raw_configs)
    spec_names = tuple(spec.name for spec in specs)
    if len(set(spec_names)) != len(spec_names):
        raise ValueError("sweep config names must be unique")

    slices = []
    for dataset_key in dataset_keys:
        task_path, task = _ensure_task_json(dataset_key)
        dataset_root = args.output_root / dataset_key
        sweep_root = dataset_root / "index_sweep"
        candidate_root = dataset_root / "candidate_compare_auto"
        artifact_prefix = _slugify(dataset_key)

        sweep_summary_path, sweep_summary = _run_indexed_sweep(
            task=task,
            task_path=task_path,
            output_root=sweep_root,
            artifact_prefix=artifact_prefix,
            specs=specs,
            rebuild_count=args.rebuild_count,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
        )
        selection_path, selection_summary, bundle_path, bundle = (
            _run_selected_candidates(
                task=task,
                task_path=task_path,
                output_root=candidate_root,
                artifact_prefix=artifact_prefix,
                sweep_summary=sweep_summary,
                rebuild_count=args.rebuild_count,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )
        slices.append(
            (
                dataset_key,
                str(task_path),
                task,
                sweep_summary_path,
                sweep_summary,
                selection_path,
                selection_summary,
                bundle_path,
                bundle,
            )
        )

    selection_policy = str(slices[0][6]["selection_policy"])
    for slice_payload in slices[1:]:
        if str(slice_payload[6]["selection_policy"]) != selection_policy:
            raise ValueError("all slices must share the same selection policy")

    matrix_summary = build_lancedb_indexed_matrix_summary(
        matrix_id=args.artifact_prefix,
        selection_policy=selection_policy,
        sweep_config_names=spec_names,
        slices=slices,
    ).to_json_ready()
    matrix_path = args.output_root / f"{args.artifact_prefix}_summary.json"
    _write_json(matrix_path, matrix_summary)

    print(f"wrote {matrix_path}")
    mean_best_quality_latency = sum(
        float(row["best_quality_candidate_mean_search_seconds"])
        for row in matrix_summary["rows"]
    ) / float(matrix_summary["slice_count"])
    print(f"Mean: {mean_best_quality_latency}")


if __name__ == "__main__":
    main()
