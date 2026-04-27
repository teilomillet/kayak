"""Runs centroid-budget policy replay on top of measured sweep rows.

This module owns report assembly for policy replay. It reuses the measured
budget sweep and does not introduce a new production search policy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import (
    STATUS_OK,
    AddressServeSweepCase,
)
from kayak_bridge.gpu_i8_centroid_budget_policy_replay import (
    replay_policy_row,
    summarize_policy_rows,
)
from kayak_bridge.gpu_i8_centroid_budget_runner import (
    build_report as build_sweep_report,
)
from kayak_bridge.gpu_i8_centroid_budget_sweep import (
    CentroidBudgetSweepControls,
)


def build_report(
    *,
    cases: Sequence[AddressServeSweepCase],
    controls: CentroidBudgetSweepControls,
    policy_names: Sequence[str],
    case_selection: dict[str, object],
) -> dict[str, Any]:
    sweep_report = build_sweep_report(
        cases=cases,
        controls=controls,
        case_selection=case_selection,
    )
    policy_cases = [
        replay_case_policies(
            case=case,
            sweep_case=sweep_case,
            policy_names=policy_names,
        )
        for case, sweep_case in zip(
            cases,
            sweep_report["cases"],
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "benchmark": "gpu_i8_centroid_budget_policy_replay",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": report_status(policy_cases),
        "case_selection": case_selection,
        "controls": controls.to_json_ready() | {
            "policy_names": list(policy_names),
        },
        "policy_cases": policy_cases,
        "summary": summarize_policy_rows(policy_cases),
        "source_sweep": {
            "benchmark": sweep_report["benchmark"],
            "status": sweep_report["status"],
            "cases": sweep_report["cases"],
            "summary": sweep_report["summary"],
        },
        "measurement_note": (
            "This benchmark replays named centroid-budget policies over the "
            "same measured sweep rows. Static and shape-rule policies use only "
            "explicit vector-count shape fields. Oracle policies are labeled "
            "calibration ceilings because they inspect recall after the sweep."
        ),
    }


def replay_case_policies(
    *,
    case: AddressServeSweepCase,
    sweep_case: dict[str, Any],
    policy_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "name": case.name,
        "status": sweep_case["status"],
        "shape": sweep_case["shape"],
        "seed": sweep_case["seed"],
        "baseline_centroids_per_query_vector": (
            sweep_case["baseline_centroids_per_query_vector"]
        ),
        "policy_rows": [
            replay_policy_row(
                case=case,
                case_row=sweep_case,
                policy_name=policy_name,
            )
            for policy_name in policy_names
        ],
    }


def report_status(policy_cases: Sequence[dict[str, Any]]) -> str:
    for row in policy_cases:
        if row.get("status") != STATUS_OK:
            return "error"
        if any(
            policy_row.get("status") != STATUS_OK
            for policy_row in row.get("policy_rows", [])
        ):
            return "error"
    return STATUS_OK
