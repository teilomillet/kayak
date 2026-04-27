"""Candidate-window policy choices for GPU i8 benchmark comparisons.

This module owns benchmark-only candidate_k rewrites from explicit shape
fields. It does not change public search defaults or infer quality from
measurements at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from kayak_bridge.gpu_i8_address_serve_sweep import AddressServeSweepCase


INPUT_CANDIDATE_K_POLICY = "input"
DOC_VECTORS64_125PCT_POLICY = "doc_vectors64_125pct_v0"


@dataclass(frozen=True, slots=True)
class CandidateWindowPolicyChoice:
    policy_name: str
    input_candidate_k: int
    candidate_k: int
    policy_kind: str
    rationale: str

    def to_json_ready(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "policy_kind": self.policy_kind,
            "input_candidate_k": self.input_candidate_k,
            "candidate_k": self.candidate_k,
            "rationale": self.rationale,
        }


def validate_candidate_window_policy(policy_name: str) -> None:
    if policy_name in {
        INPUT_CANDIDATE_K_POLICY,
        DOC_VECTORS64_125PCT_POLICY,
    }:
        return
    raise ValueError(f"unknown candidate-window policy: {policy_name}")


def choose_candidate_window(
    policy_name: str,
    case: AddressServeSweepCase,
) -> CandidateWindowPolicyChoice:
    validate_candidate_window_policy(policy_name)
    if policy_name == INPUT_CANDIDATE_K_POLICY:
        return CandidateWindowPolicyChoice(
            policy_name=policy_name,
            input_candidate_k=case.candidate_k,
            candidate_k=case.candidate_k,
            policy_kind="input",
            rationale="Use the explicit candidate_k from the benchmark case.",
        )
    return _doc_vectors64_125pct(case)


def _doc_vectors64_125pct(
    case: AddressServeSweepCase,
) -> CandidateWindowPolicyChoice:
    if (
        case.document_vector_count >= 64
        and case.candidate_k < case.document_count
    ):
        widened = min(case.document_count, _ceil_ratio(case.candidate_k, 5, 4))
        return CandidateWindowPolicyChoice(
            policy_name=DOC_VECTORS64_125PCT_POLICY,
            input_candidate_k=case.candidate_k,
            candidate_k=widened,
            policy_kind=DOC_VECTORS64_125PCT_POLICY,
            rationale=(
                "Benchmark-only rule derived from the doc_vectors64 "
                "candidate_k diagnostic: widen non-full document-vector-heavy "
                "windows to ceil(1.25 * input_candidate_k)."
            ),
        )
    return CandidateWindowPolicyChoice(
        policy_name=DOC_VECTORS64_125PCT_POLICY,
        input_candidate_k=case.candidate_k,
        candidate_k=case.candidate_k,
        policy_kind=DOC_VECTORS64_125PCT_POLICY,
        rationale=(
            "Benchmark-only rule made no change because the case is not a "
            "non-full document_vector_count >= 64 window."
        ),
    )


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator
