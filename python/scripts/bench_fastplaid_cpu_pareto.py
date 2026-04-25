from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import kayak
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig

from bench_fastplaid_speed_track import (
    FASTPLAID_BLOG_URL,
    FASTPLAID_REPO_URL,
    SpeedTrackShape,
    SyntheticInputs,
    benchmark_fastplaid,
    benchmark_kayak_exact,
    benchmark_kayak_plaid,
    build_synthetic_inputs,
    parse_engine_list,
    write_report,
)


@dataclass(frozen=True, slots=True)
class ShapePreset:
    name: str
    shape: SpeedTrackShape


@dataclass(frozen=True, slots=True)
class KayakPlaidConfigPreset:
    name: str
    config: KayakPlaidApproxConfig


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def shape_presets(name: str) -> tuple[ShapePreset, ...]:
    if name == "smoke":
        return (
            ShapePreset("small_64d_16dv_4q_8qv", _shape(64, 16, 4, 8)),
            ShapePreset("token_48d_96dv_2q_32qv", _shape(48, 96, 2, 32)),
        )
    if name == "scorecard":
        return (
            ShapePreset("small_128d_16dv_4q_8qv", _shape(128, 16, 4, 8)),
            ShapePreset("medium_256d_32dv_8q_16qv", _shape(256, 32, 8, 16)),
            ShapePreset("token_128d_300dv_4q_50qv", _shape(128, 300, 4, 50)),
        )
    if name == "long_token":
        return (
            ShapePreset("token_128d_300dv_4q_50qv", _shape(128, 300, 4, 50)),
        )
    raise argparse.ArgumentTypeError(
        "shape set must be one of: smoke, scorecard, long_token"
    )


def _shape(
    document_count: int,
    document_vector_count: int,
    query_count: int,
    query_vector_count: int,
) -> SpeedTrackShape:
    return SpeedTrackShape(
        document_count=document_count,
        document_vector_count=document_vector_count,
        query_count=query_count,
        query_vector_count=query_vector_count,
        vector_dim=128,
        top_k=10,
        update_document_count=0,
    )


def kayak_plaid_config_presets(name: str) -> tuple[KayakPlaidConfigPreset, ...]:
    if name == "smoke":
        return (
            KayakPlaidConfigPreset("light", _plaid_config(64, 8, 32)),
            KayakPlaidConfigPreset("balanced", _plaid_config(128, 16, 80)),
            KayakPlaidConfigPreset("recall", _plaid_config(128, 32, 160)),
        )
    if name == "scorecard":
        return (
            KayakPlaidConfigPreset("light", _plaid_config(64, 8, 32)),
            KayakPlaidConfigPreset("balanced", _plaid_config(128, 16, 80)),
            KayakPlaidConfigPreset("recall", _plaid_config(128, 32, 160)),
            KayakPlaidConfigPreset("wide_recall", _plaid_config(256, 32, 192)),
        )
    if name == "long_token":
        return (
            KayakPlaidConfigPreset("candidate_96", _plaid_config(128, 32, 96)),
            KayakPlaidConfigPreset("candidate_128", _plaid_config(128, 32, 128)),
            KayakPlaidConfigPreset("candidate_160", _plaid_config(128, 32, 160)),
            KayakPlaidConfigPreset("candidate_192", _plaid_config(128, 32, 192)),
        )
    raise argparse.ArgumentTypeError(
        "Kayak PLAID config set must be one of: smoke, scorecard, long_token"
    )


def _plaid_config(
    centroid_count: int,
    centroids_per_query_vector: int,
    candidate_k: int,
) -> KayakPlaidApproxConfig:
    return KayakPlaidApproxConfig(
        centroid_count=centroid_count,
        centroids_per_query_vector=centroids_per_query_vector,
        candidate_k=candidate_k,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a CPU-only FastPlaid-vs-Kayak Pareto sweep over explicit "
            "multi-vector shapes and Kayak approximation budgets."
        )
    )
    parser.add_argument(
        "--engines",
        type=parse_engine_list,
        default=parse_engine_list("kayak_exact,kayak_plaid,fastplaid"),
    )
    parser.add_argument(
        "--shape-set",
        choices=("smoke", "scorecard", "long_token"),
        default="smoke",
    )
    parser.add_argument(
        "--kayak-plaid-config-set",
        choices=("smoke", "scorecard", "long_token"),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--measurement-iterations", type=int, default=1)
    parser.add_argument(
        "--kayak-backend",
        choices=(kayak.MOJO_EXACT_CPU_BACKEND, kayak.NUMPY_REFERENCE_BACKEND),
        default=kayak.MOJO_EXACT_CPU_BACKEND,
    )
    parser.add_argument("--fastplaid-device", default="cpu")
    parser.add_argument(
        "--fastplaid-low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fastplaid-kmeans-niters", type=int, default=4)
    parser.add_argument("--fastplaid-max-points-per-centroid", type=int, default=256)
    parser.add_argument("--fastplaid-nbits", type=int, default=4)
    parser.add_argument("--fastplaid-batch-size", type=int, default=25_000)
    parser.add_argument("--fastplaid-update-buffer-size", type=int, default=100)
    parser.add_argument(
        "--require-fastplaid",
        action="store_true",
        help="Fail instead of writing skipped FastPlaid rows when unavailable.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(".cache/kayak/fastplaid_cpu_pareto/indexes"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/fastplaid_cpu_pareto/summary.json"),
    )
    return parser.parse_args()


def pareto_front(
    rows: Sequence[dict[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[dict[str, Any]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    front: list[dict[str, Any]] = []
    for row in ok_rows:
        if not any(
            _dominates(candidate, row, maximize=maximize, minimize=minimize)
            for candidate in ok_rows
            if candidate is not row
        ):
            front.append(row)
    return sorted(
        front,
        key=lambda row: (
            -float(row.get("recall_at_k_vs_kayak_exact", 0.0)),
            -float(row.get("query_qps", 0.0)),
            int(row.get("index_bytes", 0)),
            str(row.get("system_name", "")),
        ),
    )


def _dominates(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> bool:
    strictly_better = False
    for key in maximize:
        lhs_value = lhs.get(key)
        rhs_value = rhs.get(key)
        if lhs_value is None or rhs_value is None:
            return False
        if float(lhs_value) < float(rhs_value):
            return False
        if float(lhs_value) > float(rhs_value):
            strictly_better = True
    for key in minimize:
        lhs_value = lhs.get(key)
        rhs_value = rhs.get(key)
        if lhs_value is None or rhs_value is None:
            return False
        if float(lhs_value) > float(rhs_value):
            return False
        if float(lhs_value) < float(rhs_value):
            strictly_better = True
    return strictly_better


def benchmark_shape(
    *,
    shape_preset: ShapePreset,
    config_presets: Sequence[KayakPlaidConfigPreset],
    args: argparse.Namespace,
) -> dict[str, Any]:
    shape = shape_preset.shape
    shape.validate()
    inputs = build_synthetic_inputs(
        shape,
        seed=args.seed,
        normalize_vectors=args.normalize_vectors,
    )
    systems: list[dict[str, Any]] = []
    kayak_exact, reference_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend=args.kayak_backend,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    systems.append(_annotate_row(kayak_exact, shape_name=shape_preset.name))

    if "kayak_plaid" in args.engines:
        for config_preset in config_presets:
            row = benchmark_kayak_plaid(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=config_preset.config,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
            row["system_name"] = "kayak_plaid_mojo_" + config_preset.name
            row["config_name"] = config_preset.name
            systems.append(_annotate_row(row, shape_name=shape_preset.name))

    if "fastplaid" in args.engines:
        row = benchmark_fastplaid(
            shape=shape,
            inputs=inputs,
            reference_positions_by_query=reference_positions,
            index_root=args.index_root / shape_preset.name,
            overwrite_index_root=True,
            device=args.fastplaid_device,
            low_memory=args.fastplaid_low_memory,
            kmeans_niters=args.fastplaid_kmeans_niters,
            max_points_per_centroid=args.fastplaid_max_points_per_centroid,
            nbits=args.fastplaid_nbits,
            batch_size=args.fastplaid_batch_size,
            use_triton_kmeans=None,
            update_buffer_size=args.fastplaid_update_buffer_size,
            seed=args.seed,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            require_fastplaid=args.require_fastplaid,
        )
        row["config_name"] = "fastplaid"
        systems.append(_annotate_row(row, shape_name=shape_preset.name))

    _add_ratios(systems)
    approximate_rows = [
        row
        for row in systems
        if row.get("status") == "ok"
        and row.get("recall_at_k_vs_kayak_exact") is not None
        and not _is_exact_reference_row(row)
    ]
    shape_front = pareto_front(
        approximate_rows,
        maximize=("recall_at_k_vs_kayak_exact", "query_qps"),
        minimize=("index_bytes",),
    )
    return {
        "shape_name": shape_preset.name,
        "shape": shape.to_json_ready(),
        "input_bytes": {
            "documents": int(inputs.documents.nbytes),
            "queries": int(inputs.queries.nbytes),
        },
        "systems": systems,
        "query_recall_bytes_pareto_front": [
            _front_row(row) for row in shape_front
        ],
        "fastplaid_dominated_by_kayak": _fastplaid_dominated_by_kayak(systems),
    }


def _annotate_row(row: dict[str, Any], *, shape_name: str) -> dict[str, Any]:
    annotated = dict(row)
    annotated["shape_name"] = shape_name
    return annotated


def _add_ratios(rows: list[dict[str, Any]]) -> None:
    exact = next(
        (
            row
            for row in rows
            if row.get("status") == "ok" and _is_exact_reference_row(row)
        ),
        None,
    )
    fastplaid = next(
        (
            row
            for row in rows
            if row.get("status") == "ok" and row.get("engine") == "fastplaid"
        ),
        None,
    )
    for row in rows:
        if row.get("status") != "ok":
            continue
        if exact is not None and row is not exact:
            row["query_qps_ratio_vs_exact"] = _ratio(row["query_qps"], exact["query_qps"])
            row["query_batch_seconds_ratio_vs_exact"] = _ratio(
                row["query_batch_mean_seconds"],
                exact["query_batch_mean_seconds"],
            )
            row["index_bytes_ratio_vs_exact"] = _ratio(
                row.get("index_bytes"),
                exact.get("index_bytes"),
            )
        if fastplaid is not None and row is not fastplaid:
            row["query_qps_ratio_vs_fastplaid"] = _ratio(
                row["query_qps"], fastplaid["query_qps"]
            )
            row["query_batch_seconds_ratio_vs_fastplaid"] = _ratio(
                row["query_batch_mean_seconds"],
                fastplaid["query_batch_mean_seconds"],
            )


def _ratio(value: object, baseline: object) -> float | None:
    if value is None or baseline is None:
        return None
    denominator = float(baseline)
    if denominator == 0.0:
        return None
    return float(value) / denominator


def _front_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "shape_name",
        "system_name",
        "engine",
        "config_name",
        "query_batch_mean_seconds",
        "query_qps",
        "recall_at_k_vs_kayak_exact",
        "index_bytes",
        "build_seconds",
        "candidate_k",
        "centroid_count",
        "centroids_per_query_vector",
    )
    return {key: row[key] for key in keys if key in row}


def _fastplaid_dominated_by_kayak(rows: Sequence[dict[str, Any]]) -> bool | None:
    fastplaid = next(
        (
            row
            for row in rows
            if row.get("status") == "ok" and row.get("engine") == "fastplaid"
        ),
        None,
    )
    if fastplaid is None:
        return None
    return any(
        row.get("engine") == "kayak"
        and row.get("status") == "ok"
        and not _is_exact_reference_row(row)
        and _dominates(
            row,
            fastplaid,
            maximize=("recall_at_k_vs_kayak_exact", "query_qps"),
            minimize=("index_bytes",),
        )
        for row in rows
    )


def build_report(
    *,
    shape_results: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    all_approx_rows = [
        row
        for result in shape_results
        for row in result["systems"]
        if row.get("status") == "ok"
        and row.get("recall_at_k_vs_kayak_exact") is not None
        and not _is_exact_reference_row(row)
    ]
    return {
        "schema_version": 1,
        "benchmark": "fastplaid_cpu_pareto",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_urls": {
            "fastplaid_blog": FASTPLAID_BLOG_URL,
            "fastplaid_repo": FASTPLAID_REPO_URL,
        },
        "controls": {
            "engines": list(args.engines),
            "shape_set": args.shape_set,
            "kayak_plaid_config_set": args.kayak_plaid_config_set,
            "seed": args.seed,
            "normalize_vectors": args.normalize_vectors,
            "warmup_iterations": args.warmup_iterations,
            "measurement_iterations": args.measurement_iterations,
            "kayak_backend": args.kayak_backend,
            "fastplaid_device": args.fastplaid_device,
            "fastplaid_low_memory": args.fastplaid_low_memory,
            "fastplaid_kmeans_niters": args.fastplaid_kmeans_niters,
            "fastplaid_max_points_per_centroid": (
                args.fastplaid_max_points_per_centroid
            ),
            "fastplaid_nbits": args.fastplaid_nbits,
            "fastplaid_batch_size": args.fastplaid_batch_size,
        },
        "pareto_definition": {
            "maximize": ["recall_at_k_vs_kayak_exact", "query_qps"],
            "minimize": ["index_bytes"],
            "scope": "per explicit vector-count shape",
        },
        "shape_results": list(shape_results),
        "all_shape_front_rows": [
            row
            for result in shape_results
            for row in result["query_recall_bytes_pareto_front"]
        ],
        "all_approximate_rows": [_front_row(row) for row in all_approx_rows],
    }


def _is_exact_reference_row(row: dict[str, Any]) -> bool:
    return row.get("engine") == "kayak" and str(row.get("index_kind", "")).startswith(
        "exact_"
    )


def main() -> None:
    args = parse_args()
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    _require_positive("measurement_iterations", args.measurement_iterations)

    shape_results = [
        benchmark_shape(
            shape_preset=shape_preset,
            config_presets=kayak_plaid_config_presets(
                args.kayak_plaid_config_set
            ),
            args=args,
        )
        for shape_preset in shape_presets(args.shape_set)
    ]
    report = build_report(shape_results=shape_results, args=args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    first_exact = shape_results[0]["systems"][0]
    print(f"Mean: {first_exact['query_batch_mean_seconds']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
