from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

from bench_fastplaid_speed_track import build_synthetic_inputs  # noqa: E402
from kayak_bridge.gpu_i8_address_serve_sweep import (  # noqa: E402
    CASE_SETS,
    STATUS_OK,
    AddressServeSweepCase,
    AddressServeSweepControls,
    case_set_names,
    parse_sweep_case,
)
from kayak_bridge.gpu_i8_centroid_budget_policy import (  # noqa: E402
    choose_policy_budget,
)
from kayak_bridge.mojo_exact_cpu import load_module  # noqa: E402
from kayak_bridge.plaid_approx import (  # noqa: E402
    KayakPlaidApproxConfig,
    KayakPlaidApproxIndex,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile CPU i8 PLAID candidate-generation substeps for the GPU "
            "candidate-window rerank pipeline."
        )
    )
    parser.add_argument(
        "--case-set",
        choices=case_set_names(),
        default="wide_topk",
    )
    parser.add_argument(
        "--case",
        action="append",
        type=parse_sweep_case,
        default=None,
        help="Explicit name:key=value,key=value case. Overrides --case-set.",
    )
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument(
        "--kayak-plaid-centroids-per-query-vector",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--centroid-budget-policy",
        default=None,
        help=(
            "Optional benchmark-only policy name. When set, each case uses "
            "the policy-selected centroids_per_query_vector instead of the "
            "static --kayak-plaid-centroids-per-query-vector value."
        ),
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            ".cache/kayak/gpu_i8_candidate_generation_breakdown/summary.json"
        ),
    )
    return parser.parse_args(argv)


def selected_cases(args: argparse.Namespace) -> tuple[AddressServeSweepCase, ...]:
    if args.case:
        return tuple(args.case)
    return CASE_SETS[args.case_set]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    controls = AddressServeSweepControls(
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        seed=args.seed,
        measurement_iterations=args.measurement_iterations,
        kayak_plaid_centroid_count=args.kayak_plaid_centroid_count,
        kayak_plaid_centroids_per_query_vector=(
            args.kayak_plaid_centroids_per_query_vector
        ),
    )
    controls.validate()
    cases = selected_cases(args)
    rows = [
        run_case(
            case,
            case_index=index,
            controls=controls,
            centroids_per_query_vector=centroids_per_query_vector_for_case(
                args, case
            ),
        )
        for index, case in enumerate(cases)
    ]
    return {
        "schema_version": 1,
        "benchmark": "gpu_i8_candidate_generation_breakdown",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": report_status(rows),
        "case_selection": {
            "source": "explicit" if args.case else args.case_set,
            "case_count": len(cases),
        },
        "controls": controls.to_json_ready(),
        "centroid_budget_policy": args.centroid_budget_policy,
        "cases": rows,
        "summary": summary_payload(rows),
        "measurement_note": (
            "This benchmark profiles CPU-side i8 candidate-generation "
            "substeps that feed the GPU rerank/top-k primitive. The substep "
            "timings are benchmark-only boundaries and are not expected to "
            "sum exactly to the full candidate-generation timing because "
            "intermediate inputs are precomputed for isolation."
        ),
    }


def report_status(rows: Sequence[dict[str, Any]]) -> str:
    if all(row.get("status") == STATUS_OK for row in rows):
        return STATUS_OK
    return "error"


def centroids_per_query_vector_for_case(
    args: argparse.Namespace, case: AddressServeSweepCase
) -> int:
    if args.centroid_budget_policy is None:
        return int(args.kayak_plaid_centroids_per_query_vector)
    return choose_policy_budget(
        str(args.centroid_budget_policy), case
    ).centroids_per_query_vector


def run_case(
    case: AddressServeSweepCase,
    *,
    case_index: int,
    controls: AddressServeSweepControls,
    centroids_per_query_vector: int,
) -> dict[str, Any]:
    shape = case.shape(vector_dim=controls.vector_dim, top_k=controls.top_k)
    inputs = build_synthetic_inputs(
        shape,
        seed=controls.seed + case_index,
        normalize_vectors=False,
    )
    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=KayakPlaidApproxConfig(
            centroid_count=controls.kayak_plaid_centroid_count,
            centroids_per_query_vector=centroids_per_query_vector,
            candidate_k=case.candidate_k,
            payload="i8",
        ),
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at
    module = load_module()
    raw_profiles = (
        module.plaid_i8_candidate_generation_profile_prepared_batch_address(
            [
                int(inputs.queries.ctypes.data),
                int(inputs.queries.shape[0]),
                int(inputs.queries.shape[1]),
                centroids_per_query_vector,
                case.candidate_k,
                controls.measurement_iterations,
                index._prepared_index,
            ]
        )
    )
    profiles = tuple(profile_pairs_to_dict(row) for row in raw_profiles)
    return {
        "name": case.name,
        "status": STATUS_OK,
        "shape": case.to_json_ready(
            vector_dim=controls.vector_dim,
            top_k=controls.top_k,
        ),
        "seed": controls.seed + case_index,
        "cpu_i8_build_seconds": build_seconds,
        "query_profiles": profiles,
        "aggregate": aggregate_profiles(profiles),
    }


def profile_pairs_to_dict(row: Sequence[Sequence[Any]]) -> dict[str, Any]:
    return {str(key): value for key, value in row}


def aggregate_profiles(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {}
    float_fields = (
        "full_candidate_mean_seconds",
        "workspace_full_candidate_mean_seconds",
        "workspace_candidate_position_agreement",
        "unordered_candidate_mean_seconds",
        "unordered_candidate_set_agreement",
        "centroid_scoring_mean_seconds",
        "centroid_selection_mean_seconds",
        "unordered_centroid_selection_mean_seconds",
        "centroid_selection_set_agreement",
        "posting_accumulation_mean_seconds",
        "final_topk_mean_seconds",
        "unordered_final_topk_mean_seconds",
    )
    int_fields = (
        "selected_centroid_count",
        "posting_visit_count",
        "touched_document_count",
        "output_candidate_count",
    )
    aggregate: dict[str, Any] = {
        f"{field}_batch_sum": sum(float(profile[field]) for profile in profiles)
        for field in float_fields
    }
    for field in int_fields:
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
    full = aggregate["full_candidate_mean_seconds_batch_sum"]
    workspace_full = aggregate[
        "workspace_full_candidate_mean_seconds_batch_sum"
    ]
    aggregate[
        "workspace_full_candidate_mean_seconds_per_full_candidate_second"
    ] = ratio(workspace_full, full)
    aggregate["unordered_candidate_mean_seconds_per_full_candidate_second"] = (
        ratio(
            aggregate["unordered_candidate_mean_seconds_batch_sum"],
            full,
        )
    )
    aggregate[
        "unordered_centroid_selection_mean_seconds_per_centroid_selection_second"
    ] = ratio(
        aggregate["unordered_centroid_selection_mean_seconds_batch_sum"],
        aggregate["centroid_selection_mean_seconds_batch_sum"],
    )
    for field in (
        "centroid_scoring_mean_seconds",
        "centroid_selection_mean_seconds",
        "unordered_centroid_selection_mean_seconds",
        "posting_accumulation_mean_seconds",
        "final_topk_mean_seconds",
        "unordered_final_topk_mean_seconds",
    ):
        aggregate[f"{field}_per_full_candidate_second"] = ratio(
            aggregate[f"{field}_batch_sum"],
            full,
        )
    aggregate[
        "unordered_final_topk_mean_seconds_per_final_topk_second"
    ] = ratio(
        aggregate["unordered_final_topk_mean_seconds_batch_sum"],
        aggregate["final_topk_mean_seconds_batch_sum"],
    )
    return aggregate


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def summary_payload(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == STATUS_OK]
    full_values = [
        float(row["aggregate"]["full_candidate_mean_seconds_batch_sum"])
        for row in ok_rows
        if row.get("aggregate")
    ]
    posting_values = [
        float(row["aggregate"]["posting_accumulation_mean_seconds_batch_sum"])
        for row in ok_rows
        if row.get("aggregate")
    ]
    workspace_ratios = [
        float(
            row["aggregate"][
                "workspace_full_candidate_mean_seconds_per_full_candidate_second"
            ]
        )
        for row in ok_rows
        if row.get("aggregate")
        and row["aggregate"][
            "workspace_full_candidate_mean_seconds_per_full_candidate_second"
        ]
        is not None
    ]
    workspace_agreements = [
        float(
            row["aggregate"][
                "workspace_candidate_position_agreement_batch_sum"
            ]
        )
        / float(row["aggregate"]["query_count"])
        for row in ok_rows
        if row.get("aggregate")
        and int(row["aggregate"]["query_count"]) > 0
    ]
    unordered_candidate_ratios = [
        float(
            row["aggregate"][
                "unordered_candidate_mean_seconds_per_full_candidate_second"
            ]
        )
        for row in ok_rows
        if row.get("aggregate")
        and row["aggregate"][
            "unordered_candidate_mean_seconds_per_full_candidate_second"
        ]
        is not None
    ]
    unordered_candidate_agreements = [
        float(row["aggregate"]["unordered_candidate_set_agreement_batch_sum"])
        / float(row["aggregate"]["query_count"])
        for row in ok_rows
        if row.get("aggregate")
        and int(row["aggregate"]["query_count"]) > 0
    ]
    centroid_selection_ratios = [
        float(
            row["aggregate"][
                "unordered_centroid_selection_mean_seconds_per_centroid_selection_second"
            ]
        )
        for row in ok_rows
        if row.get("aggregate")
        and row["aggregate"][
            "unordered_centroid_selection_mean_seconds_per_centroid_selection_second"
        ]
        is not None
    ]
    centroid_selection_agreements = [
        float(row["aggregate"]["centroid_selection_set_agreement_batch_sum"])
        / float(row["aggregate"]["query_count"])
        for row in ok_rows
        if row.get("aggregate")
        and int(row["aggregate"]["query_count"]) > 0
    ]
    unordered_topk_ratios = [
        float(
            row["aggregate"][
                "unordered_final_topk_mean_seconds_per_final_topk_second"
            ]
        )
        for row in ok_rows
        if row.get("aggregate")
        and row["aggregate"][
            "unordered_final_topk_mean_seconds_per_final_topk_second"
        ]
        is not None
    ]
    return {
        "case_count": len(rows),
        "ok_case_count": len(ok_rows),
        "best_full_candidate_mean_seconds_batch_sum": (
            min(full_values) if full_values else None
        ),
        "worst_full_candidate_mean_seconds_batch_sum": (
            max(full_values) if full_values else None
        ),
        "best_posting_accumulation_mean_seconds_batch_sum": (
            min(posting_values) if posting_values else None
        ),
        "worst_posting_accumulation_mean_seconds_batch_sum": (
            max(posting_values) if posting_values else None
        ),
        "best_workspace_full_candidate_seconds_per_full_candidate_second": (
            min(workspace_ratios) if workspace_ratios else None
        ),
        "worst_workspace_full_candidate_seconds_per_full_candidate_second": (
            max(workspace_ratios) if workspace_ratios else None
        ),
        "min_workspace_candidate_position_agreement": (
            min(workspace_agreements) if workspace_agreements else None
        ),
        "best_unordered_candidate_seconds_per_full_candidate_second": (
            min(unordered_candidate_ratios)
            if unordered_candidate_ratios
            else None
        ),
        "worst_unordered_candidate_seconds_per_full_candidate_second": (
            max(unordered_candidate_ratios)
            if unordered_candidate_ratios
            else None
        ),
        "min_unordered_candidate_set_agreement": (
            min(unordered_candidate_agreements)
            if unordered_candidate_agreements
            else None
        ),
        "best_unordered_centroid_selection_seconds_per_centroid_selection_second": (
            min(centroid_selection_ratios)
            if centroid_selection_ratios
            else None
        ),
        "worst_unordered_centroid_selection_seconds_per_centroid_selection_second": (
            max(centroid_selection_ratios)
            if centroid_selection_ratios
            else None
        ),
        "min_centroid_selection_set_agreement": (
            min(centroid_selection_agreements)
            if centroid_selection_agreements
            else None
        ),
        "best_unordered_final_topk_seconds_per_final_topk_second": (
            min(unordered_topk_ratios) if unordered_topk_ratios else None
        ),
        "worst_unordered_final_topk_seconds_per_final_topk_second": (
            max(unordered_topk_ratios) if unordered_topk_ratios else None
        ),
    }


def emit_quiet_means(report: dict[str, Any]) -> None:
    for row in report["cases"]:
        aggregate = row.get("aggregate") or {}
        for field in (
            "full_candidate_mean_seconds_batch_sum",
            "workspace_full_candidate_mean_seconds_batch_sum",
            "unordered_candidate_mean_seconds_batch_sum",
            "centroid_scoring_mean_seconds_batch_sum",
            "centroid_selection_mean_seconds_batch_sum",
            "unordered_centroid_selection_mean_seconds_batch_sum",
            "posting_accumulation_mean_seconds_batch_sum",
            "final_topk_mean_seconds_batch_sum",
            "unordered_final_topk_mean_seconds_batch_sum",
        ):
            value = aggregate.get(field)
            if value is None:
                continue
            print(
                f"== gpu_i8_candidate_generation_breakdown_{row['name']}_{field} =="
            )
            print(f"Mean: {value}")


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        emit_quiet_means(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
