"""Centroid-budget policy choices for the GPU i8 pipeline.

This module owns benchmark-only policy selection from explicit shape fields. It
does not inspect measured recall, build indexes, or change public search
defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase


STATUS_OK = "ok"
STATUS_MISSING_BUDGET = "missing_budget"

SHAPE_RULE_V0_POLICY = "shape_rule_v0"
ORACLE_FASTEST_NO_FINAL_RECALL_LOSS = "oracle_fastest_no_final_recall_loss"
ORACLE_FASTEST_NO_WINDOW_RECALL_LOSS = (
    "oracle_fastest_no_candidate_window_recall_loss"
)
ORACLE_POLICIES = (
    ORACLE_FASTEST_NO_FINAL_RECALL_LOSS,
    ORACLE_FASTEST_NO_WINDOW_RECALL_LOSS,
)
DEFAULT_POLICY_NAMES = (
    "static4",
    "static8",
    "static16",
    "static24",
    "static32",
    SHAPE_RULE_V0_POLICY,
    ORACLE_FASTEST_NO_FINAL_RECALL_LOSS,
)


@dataclass(frozen=True, slots=True)
class CentroidBudgetPolicyChoice:
    policy_name: str
    centroids_per_query_vector: int
    policy_kind: str
    rationale: str

    def to_json_ready(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "policy_kind": self.policy_kind,
            "centroids_per_query_vector": self.centroids_per_query_vector,
            "rationale": self.rationale,
        }


def parse_policy_names(value: str) -> tuple[str, ...]:
    policies: list[str] = []
    for raw_policy in value.split(","):
        policy = raw_policy.strip()
        if not policy:
            continue
        validate_policy_name(policy)
        if policy not in policies:
            policies.append(policy)
    if not policies:
        raise ValueError("at least one centroid-budget policy is required")
    return tuple(policies)


def validate_policy_name(policy_name: str) -> None:
    if policy_name.startswith("static"):
        _static_budget(policy_name)
        return
    if policy_name == SHAPE_RULE_V0_POLICY:
        return
    if policy_name in ORACLE_POLICIES:
        return
    raise ValueError(f"unknown centroid-budget policy: {policy_name}")


def choose_policy_budget(
    policy_name: str,
    case: AddressServeSweepCase,
) -> CentroidBudgetPolicyChoice:
    validate_policy_name(policy_name)
    if policy_name.startswith("static"):
        budget = _static_budget(policy_name)
        return CentroidBudgetPolicyChoice(
            policy_name=policy_name,
            centroids_per_query_vector=budget,
            policy_kind="static",
            rationale="Fixed centroids_per_query_vector for policy replay.",
        )
    if policy_name == SHAPE_RULE_V0_POLICY:
        return _shape_rule_v0(case)
    raise ValueError(f"{policy_name} is an oracle policy, not a shape policy")


def _shape_rule_v0(case: AddressServeSweepCase) -> CentroidBudgetPolicyChoice:
    if case.query_vector_count >= 32:
        return _shape_choice(4, "query_vector_count >= 32")
    if case.document_vector_count >= 64:
        return _shape_choice(16, "document_vector_count >= 64")
    if case.query_count >= 4:
        return _shape_choice(8, "query_count >= 4")
    if case.document_count >= 512:
        return _shape_choice(24, "document_count >= 512")
    if case.query_vector_count >= 16:
        return _shape_choice(16, "query_vector_count >= 16")
    return _shape_choice(4, "default non-full candidate window")


def _shape_choice(
    budget: int,
    rationale: str,
) -> CentroidBudgetPolicyChoice:
    return CentroidBudgetPolicyChoice(
        policy_name=SHAPE_RULE_V0_POLICY,
        centroids_per_query_vector=budget,
        policy_kind="shape_rule_v0",
        rationale=(
            "Benchmark-only rule derived from prior centroid-budget sweep: "
            + rationale
            + "."
        ),
    )


def _static_budget(policy_name: str) -> int:
    raw_budget = policy_name.removeprefix("static")
    if not raw_budget:
        raise ValueError("static policy must include a positive budget")
    try:
        budget = int(raw_budget)
    except ValueError as exc:
        raise ValueError("static policy budget must be an integer") from exc
    if budget <= 0:
        raise ValueError("static policy budget must be positive")
    return budget
