"""Fused GPU i8 rows for FastPlaid scope comparisons.

This module owns the benchmark-only fused centroid-posting boundary used in
FastPlaid comparison reports. It does not run FastPlaid and does not expose a
public GPU search backend.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

from kayak_bridge.gpu_device_capability import MojoGpuCapability
from kayak_bridge.gpu_i8_fastplaid_fused_probe import (
    fused_query_windows,
    fused_reference_score_windows,
    rank_document_scores_by_query,
    run_fused_centroid_posting_scope_probe,
)
from kayak_bridge.gpu_i8_fastplaid_topk_metrics import (
    STATUS_PARTIAL_GPU_UNAVAILABLE,
)
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex

from bench_fastplaid_speed_track import (  # noqa: E402
    SpeedTrackShape,
    SyntheticInputs,
    mean_recall_at_k,
)


def build_fused_centroid_posting_scope_row(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    reference_positions: Sequence[Sequence[int]],
    capability: MojoGpuCapability,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=args.kayak_plaid_centroid_count,
            centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
            candidate_k=args.candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at
    payload = index.i8_payload_snapshot()
    query_windows = fused_query_windows(
        base_queries=inputs.queries,
        shape=shape,
        window_count=args.gpu_topk_session_iterations,
        seed=args.seed + 300_000,
    )
    reference_score_windows = fused_reference_score_windows(
        index=index,
        payload=payload,
        query_windows=query_windows,
        centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
    )
    base_ranked_positions = rank_document_scores_by_query(
        reference_score_windows[0],
        shape=shape,
    )
    probe = run_fused_centroid_posting_scope_probe(
        capability=capability,
        shape=shape,
        query_windows=query_windows,
        payload=payload,
        reference_score_windows=reference_score_windows,
        centroids_per_query_vector=args.kayak_plaid_centroids_per_query_vector,
        configured_candidate_k=args.candidate_k,
        warmup_iterations=args.warmup_iterations,
    )
    parsed = _parsed_payload(probe)
    return {
        "system_name": "kayak_gpu_i8_fused_centroid_posting_probe",
        "engine": "kayak_gpu_probe",
        "status": (
            probe.get("status")
            if isinstance(probe, dict)
            else STATUS_PARTIAL_GPU_UNAVAILABLE
        ),
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": index.index_kind,
        "rerank_kind": index.rerank_kind,
        "scope": "real_kayak_i8_fused_centroid_posting_topk",
        "shape": _shape_payload(shape, candidate_k=args.candidate_k),
        "cpu_i8_build_seconds": build_seconds,
        "recall_at_k_vs_kayak_exact": mean_recall_at_k(
            candidate_positions_by_query=base_ranked_positions,
            reference_positions_by_query=reference_positions,
            k=shape.top_k,
        ),
        "target_accelerator": capability.target_accelerator,
        "parsed": parsed,
        "error": probe.get("error") if isinstance(probe, dict) else None,
        "measurements": probe.get("measurements") if isinstance(probe, dict) else None,
    }


def build_missing_fused_centroid_posting_scope_row(
    *,
    shape: SpeedTrackShape,
    candidate_k: int,
    capability: MojoGpuCapability,
) -> dict[str, Any]:
    return {
        "system_name": "kayak_gpu_i8_fused_centroid_posting_probe",
        "engine": "kayak_gpu_probe",
        "status": STATUS_PARTIAL_GPU_UNAVAILABLE,
        "backend": (
            f"mojo:{capability.target_accelerator}"
            if capability.target_accelerator is not None
            else None
        ),
        "index_kind": "sampled_centroid_postings_i8_proxy",
        "scope": "real_kayak_i8_fused_centroid_posting_topk",
        "shape": _shape_payload(shape, candidate_k=candidate_k),
        "parsed": {},
        "derived": {},
    }


def _shape_payload(shape: SpeedTrackShape, *, candidate_k: int) -> dict[str, int]:
    return shape.to_json_ready() | {
        "candidate_k": candidate_k,
        "candidate_score_count_total": shape.query_count * candidate_k,
    }


def _parsed_payload(row: dict[str, object] | None) -> dict[str, object]:
    parsed = None if row is None else row.get("parsed")
    return parsed if isinstance(parsed, dict) else {}
