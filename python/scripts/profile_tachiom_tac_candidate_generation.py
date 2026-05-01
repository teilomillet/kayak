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

from bench_tachiom_tac_probe import (  # noqa: E402
    TACHIOM_PAPER_URL,
    TokenStructuredInputs,
    build_token_structured_inputs,
)
from bench_tachiom_tac_sweep import (  # noqa: E402
    NamedShape,
    NamedTacConfig,
    shape_presets,
    tac_config_presets,
)
from kayak_bridge.tachiom_probe import (  # noqa: E402
    TachiomTacIndex,
    TachiomTacMojoIndex,
)


STATUS_OK = "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile Mojo Tachiom TAC candidate-generation and exact rerank "
            "substeps on token-structured shapes."
        )
    )
    parser.add_argument(
        "--shape-set",
        choices=("smoke", "moderate", "large", "moderate_large"),
        default="moderate",
    )
    parser.add_argument(
        "--config-set",
        choices=("smoke", "frontier"),
        default="frontier",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        default=None,
        help="Profile only the named TAC config. Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--zipf-skew", type=float, default=1.1)
    parser.add_argument("--document-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.04)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/tachiom_tac_profile/summary.json"),
    )
    return parser.parse_args()


def selected_tac_configs(args: argparse.Namespace) -> tuple[NamedTacConfig, ...]:
    configs = tac_config_presets(args.config_set)
    if args.config_name is None:
        return configs
    requested = set(args.config_name)
    selected = tuple(config for config in configs if config.name in requested)
    missing = requested - {config.name for config in selected}
    if missing:
        raise ValueError("unknown config-name(s): " + ", ".join(sorted(missing)))
    return selected


def run_shape(
    *,
    named_shape: NamedShape,
    inputs: TokenStructuredInputs,
    tac_configs: Sequence[NamedTacConfig],
    measurement_iterations: int,
) -> dict[str, Any]:
    rows = [
        run_config(
            named_shape=named_shape,
            inputs=inputs,
            named_config=named_config,
            measurement_iterations=measurement_iterations,
        )
        for named_config in tac_configs
    ]
    return {
        "shape_name": named_shape.name,
        "shape": named_shape.shape.to_json_ready(),
        "configs": rows,
        "dominant_stage_counts": _dominant_stage_counts(rows),
    }


def run_config(
    *,
    named_shape: NamedShape,
    inputs: TokenStructuredInputs,
    named_config: NamedTacConfig,
    measurement_iterations: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    python_index = TachiomTacIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        token_ids=inputs.token_ids,
        config=named_config.config,
        final_k=named_shape.shape.top_k,
    )
    python_build_seconds = time.perf_counter() - started_at
    started_at = time.perf_counter()
    mojo_index = TachiomTacMojoIndex.from_tac_index(python_index)
    mojo_prepare_seconds = time.perf_counter() - started_at
    profiles = mojo_index.profile_candidate_generation_batch(
        inputs.queries,
        final_k=named_shape.shape.top_k,
        measurement_iterations=measurement_iterations,
    )
    aggregate = aggregate_profiles(profiles)
    return {
        "config_name": named_config.name,
        "status": STATUS_OK,
        "python_tac_build_seconds": python_build_seconds,
        "mojo_prepare_seconds": mojo_prepare_seconds,
        "build_seconds": python_build_seconds + mojo_prepare_seconds,
        "index_bytes": python_index.index_bytes,
        "index_bytes_kind": "exact_vectors_plus_tac_sidecar",
        "tac_sidecar_index_bytes": python_index.sidecar_index_bytes,
        "exact_rerank_vector_bytes": python_index.exact_rerank_vector_bytes,
        "centroid_count": python_index.centroid_count,
        "centroids_per_query_vector": named_config.config.centroids_per_query_vector,
        "candidate_k": named_config.config.candidate_k,
        "posting_count": python_index.posting_count,
        "allocation": python_index.allocation_summary.to_json_ready(),
        "query_profiles": profiles,
        "aggregate": aggregate,
    }


def aggregate_profiles(profiles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        return {}
    float_fields = (
        "full_candidate_mean_seconds",
        "full_search_mean_seconds",
        "centroid_scoring_mean_seconds",
        "centroid_selection_mean_seconds",
        "posting_accumulation_mean_seconds",
        "final_topk_mean_seconds",
        "exact_rerank_mean_seconds",
    )
    int_fields = (
        "selected_centroid_count",
        "posting_visit_count",
        "touched_document_count",
        "seen_document_count",
        "output_candidate_count",
        "output_final_count",
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
    aggregate["final_k"] = int(first["final_k"])

    full_candidate = float(aggregate["full_candidate_mean_seconds_batch_sum"])
    full_search = float(aggregate["full_search_mean_seconds_batch_sum"])
    for field in (
        "centroid_scoring_mean_seconds",
        "centroid_selection_mean_seconds",
        "posting_accumulation_mean_seconds",
        "final_topk_mean_seconds",
    ):
        aggregate[f"{field}_per_full_candidate_second"] = ratio(
            float(aggregate[f"{field}_batch_sum"]),
            full_candidate,
        )
        aggregate[f"{field}_per_full_search_second"] = ratio(
            float(aggregate[f"{field}_batch_sum"]),
            full_search,
        )
    aggregate["exact_rerank_mean_seconds_per_full_search_second"] = ratio(
        float(aggregate["exact_rerank_mean_seconds_batch_sum"]),
        full_search,
    )
    aggregate["full_candidate_mean_seconds_per_full_search_second"] = ratio(
        full_candidate,
        full_search,
    )
    aggregate["dominant_isolated_stage"] = dominant_stage(aggregate)
    return aggregate


def dominant_stage(aggregate: dict[str, Any]) -> str:
    stage_fields = {
        "centroid_scoring": "centroid_scoring_mean_seconds_batch_sum",
        "centroid_selection": "centroid_selection_mean_seconds_batch_sum",
        "posting_accumulation": "posting_accumulation_mean_seconds_batch_sum",
        "final_topk": "final_topk_mean_seconds_batch_sum",
        "exact_rerank": "exact_rerank_mean_seconds_batch_sum",
    }
    return max(stage_fields, key=lambda stage: float(aggregate[stage_fields[stage]]))


def _dominant_stage_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        stage = str((row.get("aggregate") or {}).get("dominant_isolated_stage"))
        if stage:
            counts[stage] = counts.get(stage, 0) + 1
    return counts


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def emit_quiet_means(report: dict[str, Any]) -> None:
    fields = (
        "full_candidate_mean_seconds_batch_sum",
        "full_search_mean_seconds_batch_sum",
        "centroid_scoring_mean_seconds_batch_sum",
        "centroid_selection_mean_seconds_batch_sum",
        "posting_accumulation_mean_seconds_batch_sum",
        "final_topk_mean_seconds_batch_sum",
        "exact_rerank_mean_seconds_batch_sum",
    )
    for shape in report["shapes"]:
        for row in shape["configs"]:
            aggregate = row.get("aggregate") or {}
            for field in fields:
                value = aggregate.get(field)
                if value is None:
                    continue
                label = (
                    "tachiom_tac_profile_"
                    + str(shape["shape_name"])
                    + "_"
                    + str(row["config_name"])
                    + "_"
                    + field
                )
                print(f"== {label} ==")
                print(f"Mean: {value}")


def main() -> None:
    args = parse_args()
    if args.measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    tac_configs = selected_tac_configs(args)
    shape_reports: list[dict[str, Any]] = []
    for named_shape in shape_presets(args.shape_set):
        inputs = build_token_structured_inputs(
            named_shape.shape,
            seed=args.seed,
            zipf_skew=args.zipf_skew,
            document_noise=args.document_noise,
            query_noise=args.query_noise,
        )
        shape_reports.append(
            run_shape(
                named_shape=named_shape,
                inputs=inputs,
                tac_configs=tac_configs,
                measurement_iterations=args.measurement_iterations,
            )
        )

    report = {
        "schema_version": 1,
        "benchmark": "tachiom_tac_candidate_generation_profile",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "paper": {
            "name": "Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing",
            "url": TACHIOM_PAPER_URL,
        },
        "shape_set": args.shape_set,
        "config_set": args.config_set,
        "measurement_iterations": args.measurement_iterations,
        "status": STATUS_OK,
        "measurement_note": (
            "Substep timings are benchmark-only boundaries and are not expected "
            "to sum exactly to full search because intermediate values are "
            "precomputed for isolated measurements."
        ),
        "shapes": shape_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        emit_quiet_means(report)


if __name__ == "__main__":
    main()
