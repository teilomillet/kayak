"""Replay centroid-budget policies over measured sweep rows.

This module owns measured-row baseline comparison and summary aggregation. It
does not choose deployable defaults or run benchmark loops.
"""

from __future__ import annotations

from typing import Any, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from kayak_bridge.gpu_i8_centroid_budget_policy import (
    ORACLE_FASTEST_NO_FINAL_RECALL_LOSS,
    ORACLE_FASTEST_NO_WINDOW_RECALL_LOSS,
    ORACLE_POLICIES,
    STATUS_MISSING_BUDGET,
    STATUS_OK,
    CentroidBudgetPolicyChoice,
    choose_policy_budget,
    validate_policy_name,
)


def replay_policy_row(
    *,
    case: AddressServeSweepCase,
    case_row: dict[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    choice = (
        _oracle_choice(case_row, policy_name)
        if policy_name in ORACLE_POLICIES
        else choose_policy_budget(policy_name, case)
    )
    selected = _budget_row(case_row.get("budget_rows", []), choice)
    if selected is None:
        return _missing_budget_row(choice, case_row)
    baseline = _baseline_row(case_row)
    return {
        "status": STATUS_OK,
        **choice.to_json_ready(),
        "cpu_i8_candidate_generation": selected["cpu_i8_candidate_generation"],
        "cpu_i8_same_candidate_reference": (
            selected["cpu_i8_same_candidate_reference"]
        ),
        "candidate_window_recall_at_k_vs_kayak_exact": (
            selected["candidate_window_recall_at_k_vs_kayak_exact"]
        ),
        "recall_at_k_vs_kayak_exact": selected["recall_at_k_vs_kayak_exact"],
        "comparison": _comparison_payload(selected, baseline),
    }


def summarize_policy_rows(
    policy_cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for case_row in policy_cases:
        for row in case_row.get("policy_rows", []):
            by_policy.setdefault(str(row["policy_name"]), []).append(row)
    return {
        "policy_count": len(by_policy),
        "policies": {
            policy: _summarize_one_policy(rows)
            for policy, rows in sorted(by_policy.items())
        },
    }


def _oracle_choice(
    case_row: dict[str, Any],
    policy_name: str,
) -> CentroidBudgetPolicyChoice:
    validate_policy_name(policy_name)
    baseline = _baseline_row(case_row)
    if policy_name == ORACLE_FASTEST_NO_FINAL_RECALL_LOSS:
        key = "recall_at_k_vs_kayak_exact"
        rationale = "Fastest swept row with no final recall loss versus baseline."
    elif policy_name == ORACLE_FASTEST_NO_WINDOW_RECALL_LOSS:
        key = "candidate_window_recall_at_k_vs_kayak_exact"
        rationale = (
            "Fastest swept row with no candidate-window recall loss versus "
            "baseline."
        )
    else:
        raise ValueError(f"unknown oracle policy: {policy_name}")
    baseline_recall = float(baseline[key])
    candidates = [
        row
        for row in case_row.get("budget_rows", [])
        if row.get("status") == STATUS_OK and float(row[key]) >= baseline_recall
    ]
    selected = min(
        candidates,
        key=lambda row: float(
            row["comparison"]["cpu_candidate_generation_plus_score_seconds"]
        ),
    )
    return CentroidBudgetPolicyChoice(
        policy_name=policy_name,
        centroids_per_query_vector=int(selected["centroids_per_query_vector"]),
        policy_kind="oracle_calibration",
        rationale=rationale,
    )


def _budget_row(
    rows: Sequence[dict[str, Any]],
    choice: CentroidBudgetPolicyChoice,
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if int(row["centroids_per_query_vector"])
            == choice.centroids_per_query_vector
        ),
        None,
    )


def _baseline_row(case_row: dict[str, Any]) -> dict[str, Any]:
    baseline_budget = int(case_row["baseline_centroids_per_query_vector"])
    row = _budget_row(
        case_row.get("budget_rows", []),
        CentroidBudgetPolicyChoice(
            policy_name="baseline",
            policy_kind="static",
            centroids_per_query_vector=baseline_budget,
            rationale="baseline",
        ),
    )
    if row is None:
        raise ValueError("baseline budget row missing from sweep case")
    return row


def _comparison_payload(
    selected: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | bool | None]:
    selected_candidate = float(
        selected["cpu_i8_candidate_generation"]["mean_seconds"]
    )
    baseline_candidate = float(
        baseline["cpu_i8_candidate_generation"]["mean_seconds"]
    )
    selected_plus_score = float(
        selected["comparison"]["cpu_candidate_generation_plus_score_seconds"]
    )
    baseline_plus_score = float(
        baseline["comparison"]["cpu_candidate_generation_plus_score_seconds"]
    )
    final_delta = (
        float(selected["recall_at_k_vs_kayak_exact"])
        - float(baseline["recall_at_k_vs_kayak_exact"])
    )
    window_delta = (
        float(selected["candidate_window_recall_at_k_vs_kayak_exact"])
        - float(baseline["candidate_window_recall_at_k_vs_kayak_exact"])
    )
    latency_ratio = _ratio(selected_plus_score, baseline_plus_score)
    return {
        "cpu_candidate_generation_plus_score_seconds": selected_plus_score,
        "candidate_generation_seconds_vs_baseline_budget": _ratio(
            selected_candidate,
            baseline_candidate,
        ),
        "candidate_plus_score_seconds_vs_baseline_budget": latency_ratio,
        "final_recall_delta_vs_baseline_budget": final_delta,
        "candidate_window_recall_delta_vs_baseline_budget": window_delta,
        "no_final_recall_loss_vs_baseline_budget": final_delta >= 0.0,
        "no_candidate_window_recall_loss_vs_baseline_budget": (
            window_delta >= 0.0
        ),
        "latency_no_slower_than_baseline_budget": (
            latency_ratio is not None and latency_ratio <= 1.0
        ),
    }


def _summarize_one_policy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == STATUS_OK]
    final_recalls = [
        float(row["recall_at_k_vs_kayak_exact"])
        for row in ok_rows
    ]
    window_recalls = [
        float(row["candidate_window_recall_at_k_vs_kayak_exact"])
        for row in ok_rows
    ]
    candidate_plus_score = [
        float(row["comparison"]["cpu_candidate_generation_plus_score_seconds"])
        for row in ok_rows
    ]
    latency_ratios = [
        float(row["comparison"]["candidate_plus_score_seconds_vs_baseline_budget"])
        for row in ok_rows
        if row["comparison"]["candidate_plus_score_seconds_vs_baseline_budget"]
        is not None
    ]
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "mean_recall_at_k_vs_kayak_exact": _mean(final_recalls),
        "min_recall_at_k_vs_kayak_exact": (
            min(final_recalls) if final_recalls else None
        ),
        "mean_candidate_window_recall_at_k_vs_kayak_exact": (
            _mean(window_recalls)
        ),
        "mean_candidate_plus_score_seconds": _mean(candidate_plus_score),
        "mean_candidate_plus_score_seconds_vs_baseline_budget": (
            _mean(latency_ratios)
        ),
        "no_final_recall_loss_case_count": _count_true(
            ok_rows,
            "no_final_recall_loss_vs_baseline_budget",
        ),
        "no_candidate_window_recall_loss_case_count": _count_true(
            ok_rows,
            "no_candidate_window_recall_loss_vs_baseline_budget",
        ),
        "latency_no_slower_case_count": _count_true(
            ok_rows,
            "latency_no_slower_than_baseline_budget",
        ),
    }


def _missing_budget_row(
    choice: CentroidBudgetPolicyChoice,
    case_row: dict[str, Any],
) -> dict[str, Any]:
    available = [
        int(row["centroids_per_query_vector"])
        for row in case_row.get("budget_rows", [])
    ]
    return {
        "status": STATUS_MISSING_BUDGET,
        **choice.to_json_ready(),
        "available_centroid_budgets": available,
    }


def _count_true(rows: Sequence[dict[str, Any]], key: str) -> int:
    return sum(
        1
        for row in rows
        if bool(row.get("comparison", {}).get(key))
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator
