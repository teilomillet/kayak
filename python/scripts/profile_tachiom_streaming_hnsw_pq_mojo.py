from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.dtypes import VECTOR_DTYPE  # noqa: E402
from kayak_bridge.encoded_snapshot_loader import load_snapshot_queries  # noqa: E402
from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    _same_shape_query_groups,
)
from kayak_bridge.tachiom_streaming_search import (  # noqa: E402
    load_streaming_tachiom_hnsw_pq_mojo_index,
)


FLOAT_FIELDS = (
    "full_search_mean_seconds",
    "candidate_generation_mean_seconds",
    "hnsw_traversal_mean_seconds",
    "candidate_score_accumulation_mean_seconds",
    "candidate_topk_mean_seconds",
    "candidate_pruning_mean_seconds",
    "residual_score_table_mean_seconds",
    "rerank_scoring_mean_seconds",
    "rerank_topk_mean_seconds",
)

COUNT_FIELDS = (
    "selected_centroid_count",
    "posting_visit_count",
    "touched_document_count",
    "seen_document_count",
    "ranked_candidate_count",
    "output_candidate_count",
    "output_final_count",
)

ISOLATED_STAGE_FIELDS = {
    "hnsw_traversal": "hnsw_traversal_mean_seconds_batch_sum",
    "candidate_score_accumulation": (
        "candidate_score_accumulation_mean_seconds_batch_sum"
    ),
    "candidate_topk": "candidate_topk_mean_seconds_batch_sum",
    "candidate_pruning": "candidate_pruning_mean_seconds_batch_sum",
    "residual_score_table": "residual_score_table_mean_seconds_batch_sum",
    "rerank_scoring": "rerank_scoring_mean_seconds_batch_sum",
    "rerank_topk": "rerank_topk_mean_seconds_batch_sum",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile native Mojo HNSW+PQ substeps over a materialized streaming "
            "Tachiom artifact."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument(
        "--max-query-batch-size",
        type=int,
        help="Cap same-shape batches before sending them to the profile binding.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def profile_streaming_hnsw_pq_mojo(
    *,
    snapshot_root: Path,
    index_root: Path,
    graph_root: Path | None,
    query_limit: int | None,
    measurement_iterations: int,
    max_query_batch_size: int | None,
) -> dict[str, Any]:
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if max_query_batch_size is not None and max_query_batch_size <= 0:
        raise ValueError("max_query_batch_size must be positive when provided")

    index = load_streaming_tachiom_hnsw_pq_mojo_index(
        index_root,
        graph_root=graph_root,
    )
    queries = load_snapshot_queries(snapshot_root, query_limit=query_limit)
    profiles = _profile_queries(
        index=index,
        query_matrices=queries.query_matrices,
        final_k=queries.k,
        measurement_iterations=measurement_iterations,
        max_query_batch_size=max_query_batch_size,
    )
    aggregate = aggregate_profiles(profiles)
    return {
        "schema_version": 1,
        "benchmark": "tachiom_streaming_hnsw_pq_mojo_internal_profile",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "snapshot": str(snapshot_root),
        "index": str(index_root),
        "graph": None if graph_root is None else str(graph_root),
        "engine": "streaming_tac_hnsw_pq_mojo",
        "index_kind": index.index_kind,
        "rerank_kind": index.rerank_kind,
        "index_bytes": index.index_bytes,
        "query_count": len(profiles),
        "query_vector_count_total": queries.query_vector_count,
        "final_k": queries.k,
        "measurement_iterations": measurement_iterations,
        "max_query_batch_size": max_query_batch_size,
        "measurement_note": (
            "Substep timings are benchmark-only boundaries. They isolate "
            "native HNSW traversal, candidate score accumulation, candidate "
            "top-k/pruning, residual-PQ table construction, rerank scoring, "
            "and final top-k; isolated means are not expected to sum exactly "
            "to full search."
        ),
        "query_profiles": profiles,
        "aggregate": aggregate,
    }


def _profile_queries(
    *,
    index: Any,
    query_matrices: Sequence[np.ndarray],
    final_k: int,
    measurement_iterations: int,
    max_query_batch_size: int | None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any] | None] = [None] * len(query_matrices)
    for group in _same_shape_query_groups(
        query_matrices,
        max_query_batch_size=max_query_batch_size,
    ):
        batch = np.ascontiguousarray(
            np.stack([query_matrices[index] for index in group]),
            dtype=VECTOR_DTYPE,
        )
        batch_rows = index.profile_search_batch(
            batch,
            final_k=final_k,
            measurement_iterations=measurement_iterations,
        )
        for row_index, profile in zip(group, batch_rows, strict=True):
            rows[row_index] = profile
    return tuple(_require_profile(row) for row in rows)


def aggregate_profiles(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {}
    aggregate: dict[str, Any] = {
        f"{field}_batch_sum": sum(float(profile[field]) for profile in profiles)
        for field in FLOAT_FIELDS
    }
    for field in COUNT_FIELDS:
        aggregate[f"{field}_total"] = sum(int(profile[field]) for profile in profiles)

    first = profiles[0]
    aggregate["query_count"] = len(profiles)
    aggregate["query_vector_count"] = int(first["query_vector_count"])
    aggregate["document_count"] = int(first["document_count"])
    aggregate["document_vector_count"] = int(first["document_vector_count"])
    aggregate["total_document_vector_count"] = int(
        first["total_document_vector_count"]
    )
    aggregate["centroid_count"] = int(first["centroid_count"])
    aggregate["centroids_per_query_vector"] = int(
        first["centroids_per_query_vector"]
    )
    aggregate["candidate_k"] = int(first["candidate_k"])
    aggregate["final_k"] = int(first["final_k"])
    aggregate["ef_search"] = int(first["ef_search"])
    aggregate["candidate_pruning_alpha"] = float(first["candidate_pruning_alpha"])

    full_search = float(aggregate["full_search_mean_seconds_batch_sum"])
    candidate_generation = float(
        aggregate["candidate_generation_mean_seconds_batch_sum"]
    )
    for field in FLOAT_FIELDS:
        aggregate[f"{field}_per_full_search_second"] = ratio(
            float(aggregate[f"{field}_batch_sum"]),
            full_search,
        )
        aggregate[f"{field}_per_candidate_generation_second"] = ratio(
            float(aggregate[f"{field}_batch_sum"]),
            candidate_generation,
        )
    aggregate["dominant_isolated_stage"] = dominant_stage(aggregate)
    return aggregate


def dominant_stage(aggregate: dict[str, Any]) -> str:
    return max(
        ISOLATED_STAGE_FIELDS,
        key=lambda stage: float(aggregate[ISOLATED_STAGE_FIELDS[stage]]),
    )


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _require_profile(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        raise RuntimeError("profile run did not fill every query row")
    return row


def emit_quiet_means(report: dict[str, Any]) -> None:
    aggregate = report["aggregate"]
    for field in FLOAT_FIELDS:
        value = aggregate.get(f"{field}_batch_sum")
        if value is None:
            continue
        print(f"== tachiom_streaming_hnsw_pq_mojo_{field}_batch_sum ==")
        print(f"Mean: {value}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = profile_streaming_hnsw_pq_mojo(
        snapshot_root=args.snapshot,
        index_root=args.index,
        graph_root=args.graph,
        query_limit=args.query_limit,
        measurement_iterations=args.measurement_iterations,
        max_query_batch_size=args.max_query_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.emit_quiet_mean:
        emit_quiet_means(report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
