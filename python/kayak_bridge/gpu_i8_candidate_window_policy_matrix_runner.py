"""Run construction helpers for GPU i8 candidate-window policy matrices.

This module owns case/policy selection and single-policy CLI argument
construction. It does not execute benchmarks or aggregate result metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from kayak_bridge.gpu_i8_address_serve_sweep import (
    CASE_SETS,
    STATUS_OK,
    AddressServeSweepCase,
)
from kayak_bridge.gpu_i8_candidate_window_policy import (
    COVERAGE_SAFETY_V0_POLICY,
    DOC_VECTORS64_125PCT_POLICY,
    INPUT_CANDIDATE_K_POLICY,
)


DEFAULT_CANDIDATE_WINDOW_POLICIES = (
    INPUT_CANDIDATE_K_POLICY,
    DOC_VECTORS64_125PCT_POLICY,
    COVERAGE_SAFETY_V0_POLICY,
)
DEFAULT_CASE_SOURCE = "candidate_window_generalization"


def case_sources(
    args: Any,
) -> tuple[tuple[str, tuple[AddressServeSweepCase, ...]], ...]:
    if args.case:
        return (("explicit", tuple(args.case)),)
    names = tuple(args.case_set) if args.case_set else (DEFAULT_CASE_SOURCE,)
    return tuple((name, CASE_SETS[name]) for name in names)


def candidate_window_policies(args: Any) -> tuple[str, ...]:
    policies = (
        tuple(args.candidate_window_policy)
        if args.candidate_window_policy
        else DEFAULT_CANDIDATE_WINDOW_POLICIES
    )
    unique: list[str] = []
    for policy_name in policies:
        if policy_name not in unique:
            unique.append(policy_name)
    return tuple(unique)


def single_policy_argv(
    args: Any,
    *,
    case_source: str,
    cases: Sequence[AddressServeSweepCase],
    candidate_window_policy: str,
    output: Path,
) -> list[str]:
    report_root = args.report_root / safe_name(case_source) / candidate_window_policy
    argv = [
        "--vector-dim",
        str(args.vector_dim),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--measurement-iterations",
        str(args.measurement_iterations),
        "--gpu-topk-session-iterations",
        str(args.gpu_topk_session_iterations),
        "--kayak-plaid-centroid-count",
        str(args.kayak_plaid_centroid_count),
        "--kayak-i8-candidate-order",
        args.kayak_i8_candidate_order,
        "--policy-name",
        args.policy_name,
        "--candidate-window-policy",
        candidate_window_policy,
        "--fastplaid-devices",
        ",".join(args.fastplaid_devices),
        "--report-root",
        str(report_root),
        "--output",
        str(output),
    ]
    _add_cases(argv, case_source=case_source, cases=cases)
    _add_optional_flags(argv, args)
    return argv


def serialize_case(case: AddressServeSweepCase) -> str:
    return (
        f"{case.name}:"
        f"documents={case.document_count},"
        f"document_vectors={case.document_vector_count},"
        f"queries={case.query_count},"
        f"query_vectors={case.query_vector_count},"
        f"candidate_k={case.candidate_k}"
    )


def policy_summary_path(
    root: Path,
    *,
    case_source: str,
    candidate_window_policy: str,
) -> Path:
    return (
        root
        / safe_name(case_source)
        / candidate_window_policy
        / "policy_summary.json"
    )


def controls_payload(args: Any) -> dict[str, object]:
    return {
        "vector_dim": args.vector_dim,
        "top_k": args.top_k,
        "seed": args.seed,
        "fixed_case_seed": args.fixed_case_seed,
        "policy_name": args.policy_name,
        "baseline_candidate_window_policy": (
            args.baseline_candidate_window_policy
        ),
        "fastplaid_devices": list(args.fastplaid_devices),
        "warmup_iterations": args.warmup_iterations,
        "measurement_iterations": args.measurement_iterations,
        "gpu_topk_session_iterations": args.gpu_topk_session_iterations,
        "gpu_hybrid_shortlist_k": args.gpu_hybrid_shortlist_k,
        "kayak_plaid_centroid_count": args.kayak_plaid_centroid_count,
        "kayak_i8_candidate_order": args.kayak_i8_candidate_order,
        "kayak_i8_positive_centroids_only": (
            args.kayak_i8_positive_centroids_only
        ),
        "include_full_window": args.include_full_window,
    }


def report_status(reports: Sequence[dict[str, Any]]) -> str:
    if reports and all(report.get("status") == STATUS_OK for report in reports):
        return STATUS_OK
    return "error"


def safe_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in value
    )


def _add_cases(
    argv: list[str],
    *,
    case_source: str,
    cases: Sequence[AddressServeSweepCase],
) -> None:
    if case_source == "explicit":
        for case in cases:
            argv.extend(["--case", serialize_case(case)])
        return
    argv.extend(["--case-set", case_source])


def _add_optional_flags(argv: list[str], args: Any) -> None:
    if args.include_full_window:
        argv.append("--include-full-window")
    if args.fixed_case_seed:
        argv.append("--fixed-case-seed")
    if args.gpu_hybrid_shortlist_k is not None:
        argv.extend(["--gpu-hybrid-shortlist-k", str(args.gpu_hybrid_shortlist_k)])
    if args.kayak_i8_positive_centroids_only:
        argv.append("--kayak-i8-positive-centroids-only")
    if args.require_fastplaid:
        argv.append("--require-fastplaid")
    else:
        argv.append("--no-require-fastplaid")
    if args.allow_missing_gpu:
        argv.append("--allow-missing-gpu")
    if args.overwrite_index_root:
        argv.append("--overwrite-index-root")
