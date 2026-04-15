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
from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.kayak_task_benchmark import benchmark_task_with_kayak_exact
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb
from kayak_bridge.lancedb_index_sweep import (
    build_lancedb_indexed_sweep_summary,
    parse_lancedb_indexed_sweep_spec,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep LanceDB indexed build/query controls on one task JSON."
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory where sweep artifacts should be written. Defaults to task parent.",
    )
    parser.add_argument(
        "--artifact-prefix",
        type=str,
        help="Artifact filename prefix. Defaults to the task slice name.",
    )
    parser.add_argument("--rebuild-count", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--config",
        type=str,
        action="append",
        required=True,
        help=(
            "Indexed config in `name:key=value,key=value` form. "
            "Use just `name` for the default indexed setting."
        ),
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
    specs = [parse_lancedb_indexed_sweep_spec(value) for value in args.config]
    spec_names = [spec.name for spec in specs]
    if len(set(spec_names)) != len(spec_names):
        raise ValueError("sweep config names must be unique")

    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    lancedb_scan_path = output_root / f"{artifact_prefix}_lancedb_scan_benchmark.json"
    summary_path = output_root / f"{artifact_prefix}_indexed_sweep_summary.json"

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

    config_payloads = []
    for spec in specs:
        runs = []
        for rebuild_index in range(args.rebuild_count):
            summary = benchmark_task_with_lancedb(
                task=task,
                database_root=output_root
                / f"{artifact_prefix}_{spec.name}_{rebuild_index:02d}",
                table_name=f"{table_name}_{spec.name}",
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
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
            freeze_policy=f"mean_across_{args.rebuild_count}_rebuilds",
        ).to_json_ready()
        frozen_path = output_root / f"{artifact_prefix}_{spec.name}_frozen.json"
        _write_json(frozen_path, frozen)

        config_payloads.append(
            (spec.name, str(variance_path), variance, str(frozen_path), frozen)
        )

    sweep_summary = build_lancedb_indexed_sweep_summary(
        task_path=str(args.task),
        kayak_exact_path=str(kayak_exact_path),
        kayak_exact=kayak_exact,
        lancedb_scan_path=str(lancedb_scan_path),
        lancedb_scan=lancedb_scan,
        configs=config_payloads,
    ).to_json_ready()
    _write_json(summary_path, sweep_summary)

    print(f"wrote {kayak_exact_path}")
    print(f"wrote {lancedb_scan_path}")
    for spec in specs:
        print(f"wrote {output_root / f'{artifact_prefix}_{spec.name}_variance.json'}")
        print(f"wrote {output_root / f'{artifact_prefix}_{spec.name}_frozen.json'}")
    print(f"wrote {summary_path}")
    mean_frozen_latency = sum(
        float(row["mean_search_seconds"]) for row in sweep_summary["rows"]
    ) / float(len(sweep_summary["rows"]))
    print(f"Mean: {mean_frozen_latency}")


if __name__ == "__main__":
    main()
