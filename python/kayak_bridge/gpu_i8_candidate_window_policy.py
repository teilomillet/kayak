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
COVERAGE_SAFETY_V0_POLICY = "coverage_safety_v0"


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
        COVERAGE_SAFETY_V0_POLICY,
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
    if policy_name == COVERAGE_SAFETY_V0_POLICY:
        return _coverage_safety_v0(case)
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


def _coverage_safety_v0(
    case: AddressServeSweepCase,
) -> CandidateWindowPolicyChoice:
    if case.candidate_k >= case.document_count:
        return _coverage_choice(
            case,
            candidate_k=case.candidate_k,
            rationale="the input window already covers every document",
        )
    if case.document_count >= 1024:
        return _coverage_choice(
            case,
            candidate_k=case.document_count,
            rationale=(
                "fixed-seed documents1024 diagnostic showed k=768 could "
                "still trail FastPlaid CUDA, while full-window k=1024 "
                "recovered recall"
            ),
        )
    if case.document_vector_count >= 96:
        return _coverage_choice(
            case,
            candidate_k=min(case.document_count, _ceil_ratio(case.candidate_k, 7, 4)),
            rationale=(
                "fixed-seed doc_vectors96 diagnostic required k=448 from "
                "input k=256 to beat FastPlaid CPU"
            ),
        )
    if case.document_vector_count >= 64 or case.document_count >= 512:
        return _coverage_choice(
            case,
            candidate_k=min(case.document_count, _ceil_ratio(case.candidate_k, 5, 4)),
            rationale=(
                "fixed-seed doc_vectors64 and doc_vectors48 diagnostics "
                "showed k=320 provides headroom from input k=256"
            ),
        )
    return _coverage_choice(
        case,
        candidate_k=case.candidate_k,
        rationale="no measured coverage risk rule matched this shape",
    )


def _coverage_choice(
    case: AddressServeSweepCase,
    *,
    candidate_k: int,
    rationale: str,
) -> CandidateWindowPolicyChoice:
    return CandidateWindowPolicyChoice(
        policy_name=COVERAGE_SAFETY_V0_POLICY,
        input_candidate_k=case.candidate_k,
        candidate_k=candidate_k,
        policy_kind=COVERAGE_SAFETY_V0_POLICY,
        rationale="Benchmark-only coverage rule: " + rationale + ".",
    )


def _ceil_ratio(value: int, numerator: int, denominator: int) -> int:
    return (value * numerator + denominator - 1) // denominator
