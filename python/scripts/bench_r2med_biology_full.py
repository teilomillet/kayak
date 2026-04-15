from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak

from kayak_bridge.r2med_biology_full import (
    DEFAULT_ARTIFACT_ROOT,
    benchmark_r2med_biology_full_exact,
)
from kayak_bridge.trec_run import write_trec_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kayak exact on the full public R2MED/Biology corpus and "
            "export a TREC-style run file."
        )
    )
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-output", type=Path, default=None)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--backend", type=str, default=kayak.MOJO_EXACT_CPU_BACKEND)
    parser.add_argument("--force-rebuild-snapshot", action="store_true")
    parser.add_argument("--force-rebuild-queries", action="store_true")
    return parser.parse_args()


def _default_output_path(artifact_root: Path) -> Path:
    return artifact_root / "r2med_biology_full_summary.json"


def _default_run_output_path(artifact_root: Path) -> Path:
    return artifact_root / "r2med_biology_full.run"


def _scoped_artifact_root(
    base_root: Path,
    *,
    document_limit: int | None,
    query_limit: int | None,
) -> Path:
    document_scope = "docs_full" if document_limit is None else f"docs_{document_limit}"
    query_scope = "queries_full" if query_limit is None else f"queries_{query_limit}"
    return base_root / f"{document_scope}_{query_scope}"


def main() -> None:
    args = parse_args()
    artifact_root = _scoped_artifact_root(
        args.artifact_root,
        document_limit=args.document_limit,
        query_limit=args.query_limit,
    )
    snapshot_root = artifact_root / "store"
    query_cache_path = artifact_root / "queries.json"
    output_path = args.output or _default_output_path(artifact_root)
    run_output_path = args.run_output or _default_run_output_path(artifact_root)

    summary, hits_by_query = benchmark_r2med_biology_full_exact(
        snapshot_root=snapshot_root,
        query_cache_path=query_cache_path,
        document_limit=args.document_limit,
        query_limit=args.query_limit,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
        k=args.k,
        backend=args.backend,
        force_rebuild_snapshot=args.force_rebuild_snapshot,
        force_rebuild_queries=args.force_rebuild_queries,
    )

    query_ids = tuple(f"q{index}" for index in range(summary.query_count))
    if query_cache_path.exists():
        with query_cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        query_ids = tuple(str(row["query_id"]) for row in payload["queries"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_trec_run(
        run_output_path,
        query_ids=query_ids,
        hits_by_query=hits_by_query,
        run_name="kayak_r2med_biology_full",
    )

    print(f"wrote {output_path}")
    print(f"wrote {run_output_path}")
    print(
        "nDCG@10="
        f"{summary.mean_ndcg_at_k:.6f}, "
        f"mean_search_seconds={summary.mean_search_seconds:.6f}"
    )
    print(f"Mean: {summary.mean_search_seconds:.6f}")


if __name__ == "__main__":
    main()
