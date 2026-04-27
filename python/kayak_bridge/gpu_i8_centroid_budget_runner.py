"""Runs centroid-budget sweeps for the GPU i8 candidate-window pipeline.

This module owns index construction, exact-reference comparison, and benchmark
row assembly. CLI parsing and report printing stay in the script layer.
"""

from __future__ import annotations

from datetime import UTC, datetime
import time
from typing import Any, Sequence

from bench_fastplaid_speed_track import (  # noqa: E402
    SpeedTrackShape,
    benchmark_kayak_exact,
    build_synthetic_inputs,
    mean_recall_at_k,
)
from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    STATUS_OK,
    AddressServeSweepCase,
)
from kayak_bridge.gpu_i8_centroid_budget_sweep import (  # noqa: E402
    CentroidBudgetSweepControls,
    add_baseline_ratios,
    candidate_window_recall_at_k,
    summary_payload,
)
from kayak_bridge.mojo_exact_cpu import load_module  # noqa: E402
from kayak_bridge.plaid_approx import (  # noqa: E402
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
)
from profile_gpu_i8_candidate_generation_breakdown import (  # noqa: E402
    aggregate_profiles,
    profile_pairs_to_dict,
)
from profile_gpu_i8_real_payload_rerank import (  # noqa: E402
    rank_candidate_positions_by_score,
    time_candidate_generation,
    time_same_candidate_scores,
)


def build_report(
    *,
    cases: Sequence[AddressServeSweepCase],
    controls: CentroidBudgetSweepControls,
    case_selection: dict[str, object],
) -> dict[str, Any]:
    controls.validate()
    rows = [
        run_case(case, case_index=index, controls=controls)
        for index, case in enumerate(cases)
    ]
    return {
        "schema_version": 1,
        "benchmark": "gpu_i8_centroid_budget_sweep",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": report_status(rows),
        "case_selection": case_selection,
        "controls": controls.to_json_ready(),
        "cases": rows,
        "summary": summary_payload(rows),
        "measurement_note": (
            "This sweep varies centroids_per_query_vector while holding "
            "centroid_count and candidate_k fixed. It measures exact-reference "
            "candidate-window recall, final i8 rerank recall, CPU candidate "
            "generation time, CPU same-candidate i8 score time, and the "
            "candidate-generation breakdown. It does not change the public "
            "search API or move candidate generation to GPU."
        ),
    }


def report_status(rows: Sequence[dict[str, Any]]) -> str:
    if all(row.get("status") == STATUS_OK for row in rows):
        return STATUS_OK
    return "error"


def run_case(
    case: AddressServeSweepCase,
    *,
    case_index: int,
    controls: CentroidBudgetSweepControls,
) -> dict[str, Any]:
    shape = case.shape(vector_dim=controls.vector_dim, top_k=controls.top_k)
    inputs = build_synthetic_inputs(
        shape,
        seed=controls.seed + case_index,
        normalize_vectors=False,
    )
    exact_row, exact_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend="mojo_exact_cpu",
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    budget_rows = [
        run_budget(
            shape=shape,
            inputs=inputs,
            exact_positions=exact_positions,
            candidate_k=case.candidate_k,
            centroids_per_query_vector=budget,
            controls=controls,
        )
        for budget in controls.centroid_budgets
    ]
    return {
        "name": case.name,
        "status": STATUS_OK,
        "shape": case.to_json_ready(
            vector_dim=controls.vector_dim,
            top_k=controls.top_k,
        ),
        "seed": controls.seed + case_index,
        "exact_reference": exact_row,
        "baseline_centroids_per_query_vector": (
            controls.baseline_centroids_per_query_vector
        ),
        "budget_rows": add_baseline_ratios(
            budget_rows,
            baseline_budget=controls.baseline_centroids_per_query_vector,
        ),
    }


def run_budget(
    *,
    shape: SpeedTrackShape,
    inputs: Any,
    exact_positions: Sequence[Sequence[int]],
    candidate_k: int,
    centroids_per_query_vector: int,
    controls: CentroidBudgetSweepControls,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=controls.kayak_plaid_centroid_count,
            centroids_per_query_vector=centroids_per_query_vector,
            candidate_k=candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at

    candidate_positions = index.i8_candidate_positions_batch(inputs.queries)
    candidate_window_recall = candidate_window_recall_at_k(
        candidate_positions_by_query=candidate_positions,
        reference_positions_by_query=exact_positions,
        k=shape.top_k,
    )
    scores = index.i8_score_candidate_positions_batch(
        inputs.queries,
        candidate_positions,
    )
    ranked_positions = rank_candidate_positions_by_score(
        candidate_positions,
        scores,
        final_k=shape.top_k,
    )
    final_recall = mean_recall_at_k(
        candidate_positions_by_query=ranked_positions,
        reference_positions_by_query=exact_positions,
        k=shape.top_k,
    )
    candidate_timing = time_candidate_generation(
        index,
        inputs.queries,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    score_timing = time_same_candidate_scores(
        index,
        inputs.queries,
        candidate_positions,
        warmup_iterations=controls.warmup_iterations,
        measurement_iterations=controls.measurement_iterations,
    )
    return {
        "status": STATUS_OK,
        "centroids_per_query_vector": centroids_per_query_vector,
        "cpu_i8_build_seconds": build_seconds,
        "cpu_i8_candidate_generation": candidate_timing.to_json_ready(),
        "cpu_i8_same_candidate_reference": score_timing.to_json_ready()
        | {"candidate_score_count_total": shape.query_count * candidate_k},
        "candidate_window_recall_at_k_vs_kayak_exact": candidate_window_recall,
        "recall_at_k_vs_kayak_exact": final_recall,
        "candidate_generation_breakdown": profile_breakdown(
            index=index,
            queries=inputs.queries,
            centroids_per_query_vector=centroids_per_query_vector,
            candidate_k=candidate_k,
            measurement_iterations=controls.measurement_iterations,
        ),
        "comparison": {
            "cpu_candidate_generation_plus_score_seconds": (
                candidate_timing.mean_seconds + score_timing.mean_seconds
            )
        },
    }


def profile_breakdown(
    *,
    index: KayakPlaidApproxIndex,
    queries: Any,
    centroids_per_query_vector: int,
    candidate_k: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    module = load_module()
    raw_profiles = module.plaid_i8_candidate_generation_profile_prepared_batch_address(
        [
            int(queries.ctypes.data),
            int(queries.shape[0]),
            int(queries.shape[1]),
            centroids_per_query_vector,
            candidate_k,
            measurement_iterations,
            index._prepared_index,
        ]
    )
    profiles = tuple(profile_pairs_to_dict(row) for row in raw_profiles)
    return {
        "query_profiles": profiles,
        "aggregate": aggregate_profiles(profiles),
    }
