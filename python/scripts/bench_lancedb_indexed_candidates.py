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
from kayak_bridge.lancedb_index_sweep import parse_lancedb_indexed_sweep_spec
from kayak_bridge.lancedb_indexed_selection import (
    select_lancedb_indexed_candidates_from_sweep_summary,
)
from kayak_bridge.lancedb_indexed_candidate_bundle import (
    build_lancedb_indexed_candidate_bundle,
)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return slug.strip("_")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kayak exact, LanceDB scan, and selected LanceDB indexed "
            "candidates on one encoded task JSON."
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
    parser.add_argument(
        "--sweep-summary",
        type=Path,
        help=(
            "Optional indexed sweep summary JSON. When provided, candidate configs "
            "are selected automatically from the sweep summary."
        ),
    )
    parser.add_argument(
        "--candidate-config",
        type=str,
        action="append",
        help=(
            "Indexed candidate config in `name:key=value,key=value` form. "
            "Use just `name` for the default indexed setting."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_count <= 0:
        raise ValueError("rebuild-count must be positive")
    if args.sweep_summary is None and not args.candidate_config:
        raise ValueError("provide either --candidate-config or --sweep-summary")

    task = load_task_json(str(args.task))
    output_root = args.output_root or args.task.parent
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_prefix = args.artifact_prefix or _slugify(str(task["slice_name"]))
    table_name = artifact_prefix
    selection_summary = None
    if args.sweep_summary is not None:
        selection_summary, specs = select_lancedb_indexed_candidates_from_sweep_summary(
            _load_json(args.sweep_summary)
        )
    else:
        specs = [
            parse_lancedb_indexed_sweep_spec(value) for value in args.candidate_config
        ]
    spec_names = [spec.name for spec in specs]
    if len(set(spec_names)) != len(spec_names):
        raise ValueError("candidate config names must be unique")

    kayak_exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    lancedb_scan_path = output_root / f"{artifact_prefix}_lancedb_scan_benchmark.json"
    scorecard_path = output_root / f"{artifact_prefix}_comparison_scorecard.json"
    bundle_path = output_root / f"{artifact_prefix}_candidate_bundle.json"
    selection_path = output_root / f"{artifact_prefix}_candidate_selection.json"

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

    systems = [
        ("kayak_exact", str(kayak_exact_path), kayak_exact),
        ("lancedb_scan", str(lancedb_scan_path), lancedb_scan),
    ]
    candidates = []

    for spec in specs:
        system_name = f"lancedb_indexed_{spec.name}"
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

        systems.append((system_name, str(frozen_path), frozen))
        candidates.append((system_name, str(frozen_path), frozen))

    scorecard = build_comparison_scorecard(
        systems=systems,
        baseline_name="kayak_exact",
    ).to_json_ready()
    _write_json(scorecard_path, scorecard)

    bundle = build_lancedb_indexed_candidate_bundle(
        task_path=str(args.task),
        kayak_exact_path=str(kayak_exact_path),
        lancedb_scan_path=str(lancedb_scan_path),
        scorecard_path=str(scorecard_path),
        scorecard=scorecard,
        candidates=candidates,
    ).to_json_ready()
    _write_json(bundle_path, bundle)
    if selection_summary is not None:
        _write_json(selection_path, selection_summary.to_json_ready())

    print(f"wrote {kayak_exact_path}")
    print(f"wrote {lancedb_scan_path}")
    for spec in specs:
        print(f"wrote {output_root / f'{artifact_prefix}_{spec.name}_variance.json'}")
        print(f"wrote {output_root / f'{artifact_prefix}_{spec.name}_frozen.json'}")
    print(f"wrote {scorecard_path}")
    print(f"wrote {bundle_path}")
    if selection_summary is not None:
        print(f"wrote {selection_path}")
    mean_candidate_latency = sum(
        float(row["mean_search_seconds"]) for row in bundle["rows"]
    ) / float(len(bundle["rows"]))
    print(f"Mean: {mean_candidate_latency}")


if __name__ == "__main__":
    main()
