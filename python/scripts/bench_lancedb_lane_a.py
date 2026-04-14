from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
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
from kayak_bridge.lancedb_benchmark import benchmark_task_with_lancedb
from kayak_bridge.lancedb_lane_a_bundle import build_lancedb_lane_a_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen LanceDB Lane A comparison artifacts."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".cache" / "kayak",
        help="Directory where comparison artifacts should be written.",
    )
    parser.add_argument(
        "--rebuild-count",
        type=int,
        default=5,
        help="Indexed rebuild count used for the frozen mean policy.",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help="Warmup passes for each benchmark run.",
    )
    parser.add_argument(
        "--measurement-iterations",
        type=int,
        default=3,
        help="Measured passes for each benchmark run.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run_task_builder() -> None:
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "python/scripts/build_browsecomp_plus_task_json.py")],
        check=True,
        cwd=REPO_ROOT,
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _benchmark_scan(
    *,
    task: dict,
    db_root: Path,
    table_name: str,
    output_path: Path,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict:
    summary = benchmark_task_with_lancedb(
        task=task,
        database_root=db_root,
        table_name=table_name,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        build_index=False,
    )
    payload = summary.to_json_ready()
    _write_json(output_path, payload)
    return payload


def _benchmark_indexed_variance(
    *,
    task: dict,
    db_root_stem: Path,
    table_name: str,
    output_path: Path,
    rebuild_count: int,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict:
    runs = []
    for rebuild_index in range(rebuild_count):
        summary = benchmark_task_with_lancedb(
            task=task,
            database_root=Path(f"{db_root_stem}-{rebuild_index:02d}"),
            table_name=table_name,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            build_index=True,
        )
        runs.append(summary.to_json_ready())

    variance = summarize_benchmark_variance(runs).to_json_ready()
    _write_json(output_path, variance)
    return variance


def _freeze_indexed_summary(
    *,
    variance_payload: dict,
    output_path: Path,
    rebuild_count: int,
) -> dict:
    frozen = freeze_benchmark_summary_mean(
        variance_payload["run_summaries"],
        freeze_policy=f"mean_across_{rebuild_count}_rebuilds",
    ).to_json_ready()
    _write_json(output_path, frozen)
    return frozen


def _build_scorecard(
    *,
    output_path: Path,
    kayak_exact_path: Path,
    lancedb_scan_path: Path,
    lancedb_indexed_frozen_path: Path,
) -> dict:
    systems = [
        ("kayak_exact", str(kayak_exact_path), _load_json(kayak_exact_path)),
        ("lancedb_scan", str(lancedb_scan_path), _load_json(lancedb_scan_path)),
        (
            "lancedb_ivf_pq_frozen",
            str(lancedb_indexed_frozen_path),
            _load_json(lancedb_indexed_frozen_path),
        ),
    ]
    scorecard = build_comparison_scorecard(
        systems=systems,
        baseline_name="kayak_exact",
    ).to_json_ready()
    _write_json(output_path, scorecard)
    return scorecard


def main() -> None:
    args = parse_args()
    if args.rebuild_count <= 0:
        raise ValueError("rebuild-count must be positive")

    _run_task_builder()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    evidence_task = load_task_json(
        str(output_root / "browsecomp_plus_real_subset" / "python_task_evidence.json")
    )
    gold_task = load_task_json(
        str(output_root / "browsecomp_plus_real_subset" / "python_task_gold.json")
    )

    evidence_scan_path = output_root / "lancedb_browsecomp_plus_evidence_benchmark.json"
    gold_scan_path = output_root / "lancedb_browsecomp_plus_gold_benchmark.json"
    evidence_scan = _benchmark_scan(
        task=evidence_task,
        db_root=output_root / "lancedb_browsecomp_plus_evidence",
        table_name="browsecomp_plus_evidence",
        output_path=evidence_scan_path,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    gold_scan = _benchmark_scan(
        task=gold_task,
        db_root=output_root / "lancedb_browsecomp_plus_gold",
        table_name="browsecomp_plus_gold",
        output_path=gold_scan_path,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    evidence_variance_path = (
        output_root / "lancedb_browsecomp_plus_evidence_indexed_variance.json"
    )
    gold_variance_path = (
        output_root / "lancedb_browsecomp_plus_gold_indexed_variance.json"
    )
    evidence_variance = _benchmark_indexed_variance(
        task=evidence_task,
        db_root_stem=output_root / "lancedb_browsecomp_plus_evidence_indexed_variance",
        table_name="browsecomp_plus_evidence",
        output_path=evidence_variance_path,
        rebuild_count=args.rebuild_count,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    gold_variance = _benchmark_indexed_variance(
        task=gold_task,
        db_root_stem=output_root / "lancedb_browsecomp_plus_gold_indexed_variance",
        table_name="browsecomp_plus_gold",
        output_path=gold_variance_path,
        rebuild_count=args.rebuild_count,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )

    evidence_frozen_path = (
        output_root / "lancedb_browsecomp_plus_evidence_indexed_frozen_benchmark.json"
    )
    gold_frozen_path = (
        output_root / "lancedb_browsecomp_plus_gold_indexed_frozen_benchmark.json"
    )
    evidence_frozen = _freeze_indexed_summary(
        variance_payload=evidence_variance,
        output_path=evidence_frozen_path,
        rebuild_count=args.rebuild_count,
    )
    gold_frozen = _freeze_indexed_summary(
        variance_payload=gold_variance,
        output_path=gold_frozen_path,
        rebuild_count=args.rebuild_count,
    )

    evidence_scorecard_path = output_root / "lancedb_browsecomp_plus_evidence_scorecard.json"
    gold_scorecard_path = output_root / "lancedb_browsecomp_plus_gold_scorecard.json"
    evidence_scorecard = _build_scorecard(
        output_path=evidence_scorecard_path,
        kayak_exact_path=output_root / "browsecomp_plus_evidence_benchmark.json",
        lancedb_scan_path=evidence_scan_path,
        lancedb_indexed_frozen_path=evidence_frozen_path,
    )
    gold_scorecard = _build_scorecard(
        output_path=gold_scorecard_path,
        kayak_exact_path=output_root / "browsecomp_plus_gold_benchmark.json",
        lancedb_scan_path=gold_scan_path,
        lancedb_indexed_frozen_path=gold_frozen_path,
    )

    bundle = build_lancedb_lane_a_bundle(
        gold_scorecard_path=str(gold_scorecard_path),
        gold_scorecard=gold_scorecard,
        gold_indexed_variance_path=str(gold_variance_path),
        gold_indexed_variance=gold_variance,
        gold_indexed_frozen_path=str(gold_frozen_path),
        gold_indexed_frozen=gold_frozen,
        evidence_scorecard_path=str(evidence_scorecard_path),
        evidence_scorecard=evidence_scorecard,
        evidence_indexed_variance_path=str(evidence_variance_path),
        evidence_indexed_variance=evidence_variance,
        evidence_indexed_frozen_path=str(evidence_frozen_path),
        evidence_indexed_frozen=evidence_frozen,
    ).to_json_ready()
    bundle_path = output_root / "lancedb_lane_a_bundle.json"
    _write_json(bundle_path, bundle)

    print(f"wrote {evidence_scan_path}")
    print(f"wrote {gold_scan_path}")
    print(f"wrote {evidence_variance_path}")
    print(f"wrote {gold_variance_path}")
    print(f"wrote {evidence_frozen_path}")
    print(f"wrote {gold_frozen_path}")
    print(f"wrote {evidence_scorecard_path}")
    print(f"wrote {gold_scorecard_path}")
    print(f"wrote {bundle_path}")
    # Keep a parsable timing line for compatibility with the quiet wrapper.
    print(f"Mean: {gold_frozen['mean_search_seconds']}")


if __name__ == "__main__":
    main()
