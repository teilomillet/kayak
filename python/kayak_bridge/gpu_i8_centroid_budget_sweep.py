"""Contracts for the GPU i8 centroid-budget profiling sweep.

This module owns budget parsing, candidate-window recall semantics, and summary
selection. It does not build indexes or run benchmark loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase
from profile_gpu_i8_real_payload_rerank import STATUS_OK, ratio


DEFAULT_CENTROID_BUDGETS = (4, 8, 16, 24, 32)


@dataclass(frozen=True, slots=True)
class CentroidBudgetSweepControls:
    vector_dim: int = 128
    top_k: int = 10
    seed: int = 7
    warmup_iterations: int = 1
    measurement_iterations: int = 3
    kayak_plaid_centroid_count: int = 128
    centroid_budgets: tuple[int, ...] = DEFAULT_CENTROID_BUDGETS
    baseline_centroids_per_query_vector: int = 32

    def validate(self) -> None:
        if self.vector_dim != 128:
            raise ValueError("GPU i8 centroid-budget sweep requires vector_dim=128")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.measurement_iterations <= 0:
            raise ValueError("measurement_iterations must be positive")
        if self.kayak_plaid_centroid_count <= 0:
            raise ValueError("kayak_plaid_centroid_count must be positive")
        if not self.centroid_budgets:
            raise ValueError("centroid_budgets must not be empty")
        if any(budget <= 0 for budget in self.centroid_budgets):
            raise ValueError("centroid budgets must be positive")
        if self.baseline_centroids_per_query_vector not in self.centroid_budgets:
            raise ValueError("baseline budget must be included in centroid_budgets")

    def to_json_ready(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "kayak_plaid_centroid_count": self.kayak_plaid_centroid_count,
            "centroid_budgets": list(self.centroid_budgets),
            "baseline_centroids_per_query_vector": (
                self.baseline_centroids_per_query_vector
            ),
            "kayak_plaid_payload": "i8",
            "warmup_iterations": self.warmup_iterations,
            "measurement_iterations": self.measurement_iterations,
        }


def parse_centroid_budgets(value: str) -> tuple[int, ...]:
    budgets: list[int] = []
    for raw_budget in value.split(","):
        text = raw_budget.strip()
        if not text:
            continue
        try:
            budget = int(text)
        except ValueError as exc:
            raise ValueError("centroid budgets must be integers") from exc
        if budget <= 0:
            raise ValueError("centroid budgets must be positive")
        budgets.append(budget)
    if not budgets:
        raise ValueError("at least one centroid budget is required")
    return tuple(sorted(set(budgets)))


def non_full_candidate_cases(
    cases: Sequence[AddressServeSweepCase],
    *,
    include_full_window: bool,
) -> tuple[AddressServeSweepCase, ...]:
    if include_full_window:
        return tuple(cases)
    return tuple(case for case in cases if case.candidate_k < case.document_count)


def candidate_window_recall_at_k(
    *,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_positions_by_query: Sequence[Sequence[int]],
    k: int,
) -> float:
    if len(candidate_positions_by_query) != len(reference_positions_by_query):
        raise ValueError("candidate and reference query counts must match")
    if k <= 0:
        raise ValueError("k must be positive")
    if not candidate_positions_by_query:
        raise ValueError("at least one query is required")

    recalls: list[float] = []
    for candidate_positions, reference_positions in zip(
        candidate_positions_by_query,
        reference_positions_by_query,
        strict=True,
    ):
        reference = set(reference_positions[:k])
        denominator = min(k, len(reference))
        if denominator == 0:
            recalls.append(0.0)
            continue
        candidates = set(candidate_positions)
        recalls.append(len(reference & candidates) / float(denominator))
    return float(sum(recalls) / len(recalls))


def add_baseline_ratios(
    budget_rows: Sequence[dict[str, Any]],
    *,
    baseline_budget: int,
) -> list[dict[str, Any]]:
    baseline = next(
        (
            row
            for row in budget_rows
            if row["centroids_per_query_vector"] == baseline_budget
        ),
        None,
    )
    if baseline is None:
        return [dict(row) for row in budget_rows]

    baseline_candidate = float(
        baseline["cpu_i8_candidate_generation"]["mean_seconds"]
    )
    baseline_candidate_plus_score = float(
        baseline["comparison"]["cpu_candidate_generation_plus_score_seconds"]
    )
    enriched: list[dict[str, Any]] = []
    for row in budget_rows:
        current = dict(row)
        comparison = dict(current["comparison"])
        comparison["candidate_generation_seconds_vs_baseline_budget"] = ratio(
            float(current["cpu_i8_candidate_generation"]["mean_seconds"]),
            baseline_candidate,
        )
        comparison["candidate_plus_score_seconds_vs_baseline_budget"] = ratio(
            float(comparison["cpu_candidate_generation_plus_score_seconds"]),
            baseline_candidate_plus_score,
        )
        comparison["final_recall_delta_vs_baseline_budget"] = (
            float(current["recall_at_k_vs_kayak_exact"])
            - float(baseline["recall_at_k_vs_kayak_exact"])
        )
        comparison["candidate_window_recall_delta_vs_baseline_budget"] = (
            float(current["candidate_window_recall_at_k_vs_kayak_exact"])
            - float(baseline["candidate_window_recall_at_k_vs_kayak_exact"])
        )
        current["comparison"] = comparison
        enriched.append(current)
    return enriched


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == STATUS_OK]
    per_case = {
        str(row["name"]): summarize_case(row)
        for row in ok_rows
    }
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "case_summaries": per_case,
    }


def summarize_case(row: dict[str, Any]) -> dict[str, Any]:
    budgets = [
        budget_row
        for budget_row in row.get("budget_rows", [])
        if budget_row.get("status") == STATUS_OK
    ]
    if not budgets:
        return {}
    baseline_budget = int(row["baseline_centroids_per_query_vector"])
    baseline = next(
        (
            budget
            for budget in budgets
            if int(budget["centroids_per_query_vector"]) == baseline_budget
        ),
        budgets[-1],
    )
    baseline_final_recall = float(baseline["recall_at_k_vs_kayak_exact"])
    baseline_window_recall = float(
        baseline["candidate_window_recall_at_k_vs_kayak_exact"]
    )
    no_final_loss = [
        budget
        for budget in budgets
        if float(budget["recall_at_k_vs_kayak_exact"]) >= baseline_final_recall
    ]
    no_window_loss = [
        budget
        for budget in budgets
        if float(budget["candidate_window_recall_at_k_vs_kayak_exact"])
        >= baseline_window_recall
    ]
    return {
        "baseline_centroids_per_query_vector": baseline_budget,
        "baseline_final_recall_at_k": baseline_final_recall,
        "baseline_candidate_window_recall_at_k": baseline_window_recall,
        "fastest_no_final_recall_loss": _budget_choice(no_final_loss),
        "fastest_no_candidate_window_recall_loss": _budget_choice(no_window_loss),
    }


def _budget_choice(budgets: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not budgets:
        return None
    best = min(
        budgets,
        key=lambda row: float(
            row["comparison"]["cpu_candidate_generation_plus_score_seconds"]
        ),
    )
    return {
        "centroids_per_query_vector": int(best["centroids_per_query_vector"]),
        "candidate_generation_mean_seconds": float(
            best["cpu_i8_candidate_generation"]["mean_seconds"]
        ),
        "candidate_plus_score_mean_seconds": float(
            best["comparison"]["cpu_candidate_generation_plus_score_seconds"]
        ),
        "candidate_window_recall_at_k_vs_kayak_exact": float(
            best["candidate_window_recall_at_k_vs_kayak_exact"]
        ),
        "recall_at_k_vs_kayak_exact": float(best["recall_at_k_vs_kayak_exact"]),
    }
