from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, median
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.tachiom_streaming_benchmark import (  # noqa: E402
    benchmark_streaming_tachiom_index,
    build_streaming_tachiom_exact_reference,
)


ENGINE_CHOICES = (
    "streaming_tac_hnsw_pq",
    "streaming_tac_hnsw_pq_mojo",
    "streaming_tac_hnsw_pq_mojo_address",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep query-side Tachiom policy knobs over an existing streaming "
            "TAC/HNSW/PQ artifact."
        )
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--engine", choices=ENGINE_CHOICES, default=ENGINE_CHOICES[1])
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--centroids-per-query-vector-values", required=True)
    parser.add_argument("--hnsw-ef-search-values", required=True)
    parser.add_argument(
        "--candidate-pruning-alphas",
        required=True,
        help=(
            "Comma-separated pruning settings. Use 'artifact' for the stored "
            "index value, 'disabled' for no pruning, and positive floats in "
            "(0, 1) for query-time overrides."
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=3)
    parser.add_argument("--sweep-repeats", type=int, default=1)
    parser.add_argument("--max-query-batch-size", type=int)
    parser.add_argument("--run-exact", action="store_true")
    parser.add_argument("--max-exact-vector-count", type=int)
    parser.add_argument("--min-primary-value", type=float)
    parser.add_argument("--min-final-recall-at-k-vs-exact", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sweep_repeats <= 0:
        raise ValueError("sweep_repeats must be positive")
    if args.max_query_batch_size is not None and args.max_query_batch_size <= 0:
        raise ValueError("max_query_batch_size must be positive when provided")
    kc_values = _parse_positive_ints(
        args.centroids_per_query_vector_values,
        "centroids-per-query-vector values",
    )
    ef_values = _parse_positive_ints(
        args.hnsw_ef_search_values,
        "hnsw ef_search values",
    )
    pruning_settings = _parse_pruning_settings(args.candidate_pruning_alphas)
    settings = [
        {
            "label": (
                f"kc{kc}_ef{ef}_{pruning_setting['label']}"
            ),
            "centroids_per_query_vector": kc,
            "hnsw_ef_search": ef,
            "candidate_pruning_alpha": pruning_setting[
                "candidate_pruning_alpha"
            ],
            "disable_candidate_pruning": pruning_setting[
                "disable_candidate_pruning"
            ],
        }
        for kc, ef, pruning_setting in itertools.product(
            kc_values,
            ef_values,
            pruning_settings,
        )
    ]
    exact_reference = None
    if args.run_exact:
        exact_reference = build_streaming_tachiom_exact_reference(
            snapshot_root=args.snapshot,
            index_root=args.index,
            query_limit=args.query_limit,
            max_exact_vector_count=args.max_exact_vector_count,
        )

    measurement_rows = []
    for repeat_index in range(args.sweep_repeats):
        for setting in settings:
            summary = benchmark_streaming_tachiom_index(
                snapshot_root=args.snapshot,
                index_root=args.index,
                query_limit=args.query_limit,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                run_exact=args.run_exact,
                max_exact_vector_count=args.max_exact_vector_count,
                engine=args.engine,
                graph_root=args.graph,
                max_query_batch_size=args.max_query_batch_size,
                centroids_per_query_vector=int(
                    setting["centroids_per_query_vector"]
                ),
                hnsw_ef_search=int(setting["hnsw_ef_search"]),
                candidate_pruning_alpha=setting["candidate_pruning_alpha"],
                disable_candidate_pruning=bool(
                    setting["disable_candidate_pruning"]
                ),
                exact_reference=exact_reference,
            )
            measurement_rows.append(
                {
                    "repeat_index": repeat_index,
                    "setting": setting,
                    "summary": summary,
                }
            )

    aggregate_rows = _aggregate_rows(settings=settings, measurement_rows=measurement_rows)
    best_speed_row = max(aggregate_rows, key=lambda row: float(row["query_qps_median"]))
    eligible_rows = [
        row
        for row in aggregate_rows
        if _row_is_eligible(
            row,
            min_primary_value=args.min_primary_value,
            min_final_recall_at_k_vs_exact=args.min_final_recall_at_k_vs_exact,
        )
    ]
    best_eligible_row = (
        None
        if not eligible_rows
        else max(eligible_rows, key=lambda row: float(row["query_qps_median"]))
    )
    payload = {
        "engine": args.engine,
        "snapshot": str(args.snapshot),
        "index": str(args.index),
        "graph": None if args.graph is None else str(args.graph),
        "centroids_per_query_vector_values": kc_values,
        "hnsw_ef_search_values": ef_values,
        "pruning_settings": pruning_settings,
        "sweep_repeats": args.sweep_repeats,
        "max_query_batch_size": args.max_query_batch_size,
        "exact_reference": (
            None if exact_reference is None else exact_reference.to_json_ready()
        ),
        "recommendation": {
            "best_speed": _recommendation_row(best_speed_row),
            "best_eligible": (
                None
                if best_eligible_row is None
                else _recommendation_row(best_eligible_row)
            ),
            "decision_rule": (
                "Prefer the fastest row that meets explicit judged-metric and "
                "exact-overlap floors."
            ),
            "reason": (
                "Centroid count alone changed the candidate-window size and "
                "quality. This sweep tests whether query-side k_c and ef_search "
                "recover quality without rebuilding the artifact."
            ),
        },
        "rows": aggregate_rows,
        "measurements": [
            {
                "repeat_index": int(row["repeat_index"]),
                "setting": row["setting"],
                "summary": row["summary"].to_json_ready(),
            }
            for row in measurement_rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.emit_quiet_mean:
        _emit_quiet_means(aggregate_rows=aggregate_rows, best_speed_row=best_speed_row)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_positive_ints(value: str, label: str) -> list[int]:
    sizes: list[int] = []
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped:
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError(f"{label} must be positive")
        if parsed not in sizes:
            sizes.append(parsed)
    if not sizes:
        raise ValueError(f"at least one {label} value is required")
    return sizes


def _parse_pruning_settings(value: str) -> list[dict[str, object]]:
    settings: list[dict[str, object]] = []
    seen_labels: set[str] = set()
    for raw in value.split(","):
        stripped = raw.strip().lower()
        if not stripped:
            continue
        if stripped in {"artifact", "stored"}:
            setting = {
                "label": "artifact",
                "candidate_pruning_alpha": None,
                "disable_candidate_pruning": False,
            }
        elif stripped in {"disabled", "disable", "none", "off"}:
            setting = {
                "label": "disabled",
                "candidate_pruning_alpha": None,
                "disable_candidate_pruning": True,
            }
        else:
            alpha = float(stripped)
            if not (0.0 < alpha < 1.0):
                raise ValueError("candidate pruning alpha values must be in (0, 1)")
            setting = {
                "label": _format_alpha_label(alpha),
                "candidate_pruning_alpha": alpha,
                "disable_candidate_pruning": False,
            }
        label = str(setting["label"])
        if label not in seen_labels:
            settings.append(setting)
            seen_labels.add(label)
    if not settings:
        raise ValueError("at least one pruning setting is required")
    return settings


def _format_alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace(".", "p")


def _aggregate_rows(
    *,
    settings: Sequence[dict[str, object]],
    measurement_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    aggregate_rows: list[dict[str, object]] = []
    for setting in settings:
        summaries = [
            row["summary"]
            for row in measurement_rows
            if row["setting"]["label"] == setting["label"]
        ]
        if not summaries:
            raise RuntimeError(f"missing measurements for setting {setting['label']}")
        first = summaries[0]
        durations = [row.query_batch_mean_seconds for row in summaries]
        median_seconds = float(median(durations))
        query_count = first.query_count
        aggregate_rows.append(
            {
                "setting_label": setting["label"],
                "centroids_per_query_vector": first.centroids_per_query_vector,
                "hnsw_ef_search": first.hnsw_ef_search,
                "candidate_pruning_alpha": first.candidate_pruning_alpha,
                "candidate_pruning_alpha_source": (
                    first.candidate_pruning_alpha_source
                ),
                "query_batch_min_seconds": float(min(durations)),
                "query_batch_median_seconds": median_seconds,
                "query_batch_mean_seconds": float(mean(durations)),
                "query_batch_max_seconds": float(max(durations)),
                "query_qps_median": float(query_count) / median_seconds,
                "repeat_count": len(summaries),
                "primary_value": first.primary_value,
                "mean_reciprocal_rank": first.mean_reciprocal_rank,
                "mean_ndcg_at_k": first.mean_ndcg_at_k,
                "mean_recall_at_k": first.mean_recall_at_k,
                "success_rate_at_k": first.success_rate_at_k,
                "candidate_recall_at_k_vs_exact": first.candidate_recall_at_k_vs_exact,
                "final_recall_at_k_vs_exact": first.final_recall_at_k_vs_exact,
                "candidate_window_min_count": first.candidate_window_min_count,
                "candidate_window_mean_count": first.candidate_window_mean_count,
                "candidate_window_max_count": first.candidate_window_max_count,
                "query_count": query_count,
                "document_count": first.document_count,
                "document_vector_count_total": first.document_vector_count_total,
                "query_vector_count_total": first.query_vector_count_total,
                "query_vector_count_mean": first.query_vector_count_mean,
                "centroid_count": first.centroid_count,
                "posting_count": first.posting_count,
                "index_bytes": first.index_bytes,
            }
        )
    return aggregate_rows


def _row_is_eligible(
    row: dict[str, object],
    *,
    min_primary_value: float | None,
    min_final_recall_at_k_vs_exact: float | None,
) -> bool:
    if (
        min_primary_value is not None
        and float(row["primary_value"]) < min_primary_value
    ):
        return False
    if min_final_recall_at_k_vs_exact is not None:
        final_recall = row["final_recall_at_k_vs_exact"]
        if final_recall is None:
            return False
        if float(final_recall) < min_final_recall_at_k_vs_exact:
            return False
    return True


def _recommendation_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "setting_label": row["setting_label"],
        "centroids_per_query_vector": row["centroids_per_query_vector"],
        "hnsw_ef_search": row["hnsw_ef_search"],
        "candidate_pruning_alpha": row["candidate_pruning_alpha"],
        "query_qps_median": row["query_qps_median"],
        "query_batch_median_seconds": row["query_batch_median_seconds"],
        "primary_value": row["primary_value"],
        "final_recall_at_k_vs_exact": row["final_recall_at_k_vs_exact"],
        "candidate_window_mean_count": row["candidate_window_mean_count"],
    }


def _emit_quiet_means(
    *,
    aggregate_rows: Sequence[dict[str, object]],
    best_speed_row: dict[str, object],
) -> None:
    for row in aggregate_rows:
        print(f"== {row['setting_label']} ==")
        print("Mean:", row["query_batch_median_seconds"])
    print("== best_observed ==")
    print("Mean:", best_speed_row["query_batch_median_seconds"])
    print(
        "quiet_context "
        f"best_setting={best_speed_row['setting_label']} "
        f"best_qps={float(best_speed_row['query_qps_median']):.6f} "
        f"best_primary={float(best_speed_row['primary_value']):.6f} "
        f"best_final_recall={best_speed_row['final_recall_at_k_vs_exact']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
