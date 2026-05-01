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

from kayak_bridge.plaid_approx import KayakPlaidApproxConfig
from kayak_bridge.tachiom_probe import TachiomResidualPqConfig, TachiomTacConfig

from bench_tachiom_tac_probe import (
    TACHIOM_PAPER_URL,
    TokenStructuredInputs,
    TokenStructuredShape,
    benchmark_exact,
    benchmark_kayak_plaid,
    benchmark_tachiom_tac,
    benchmark_tachiom_tac_i8_mojo,
    benchmark_tachiom_tac_mojo,
    benchmark_tachiom_tac_pq,
    benchmark_tachiom_tac_pq_mojo,
    build_pairwise_rows,
    build_token_structured_inputs,
    emit_quiet_mean_sections,
)


@dataclass(frozen=True, slots=True)
class NamedShape:
    name: str
    shape: TokenStructuredShape


@dataclass(frozen=True, slots=True)
class NamedTacConfig:
    name: str
    config: TachiomTacConfig


@dataclass(frozen=True, slots=True)
class NamedPlaidConfig:
    name: str
    config: KayakPlaidApproxConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Tachiom TAC first-gate budget sweep against Kayak's current "
            "i8 PLAID-style lane on explicit token-structured shapes."
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
        "--engines",
        type=_parse_engines,
        default=_parse_engines("tachiom_tac,kayak_plaid"),
        help="Comma-separated engines to compare after exact reference.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--zipf-skew", type=float, default=1.1)
    parser.add_argument("--document-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.04)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=2)
    parser.add_argument("--tac-pq-subspace-count", type=int, default=32)
    parser.add_argument("--tac-pq-codebook-size", type=int, default=256)
    parser.add_argument("--tac-pq-kmeans-iterations", type=int, default=6)
    parser.add_argument("--tac-pq-training-sample-count", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/tachiom_tac_sweep/summary.json"),
    )
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args()


def _parse_engines(value: str) -> tuple[str, ...]:
    engines = tuple(part.strip() for part in value.split(",") if part.strip())
    supported = {
        "tachiom_tac",
        "tachiom_tac_mojo",
        "tachiom_tac_i8_mojo",
        "tachiom_tac_pq",
        "tachiom_tac_pq_mojo",
        "kayak_plaid",
    }
    unknown = tuple(engine for engine in engines if engine not in supported)
    if unknown:
        raise argparse.ArgumentTypeError(
            "unsupported engine(s): "
            + ", ".join(unknown)
            + "; supported engines: "
            + ", ".join(sorted(supported))
        )
    if not engines:
        raise argparse.ArgumentTypeError("at least one comparison engine is required")
    return engines


def shape_presets(name: str) -> tuple[NamedShape, ...]:
    if name == "smoke":
        return (
            NamedShape(
                "smoke_96d_48dv_4q_12qv",
                TokenStructuredShape(96, 48, 4, 12, 128, 10, 256),
            ),
        )
    if name == "moderate":
        return (
            NamedShape(
                "moderate_512d_96dv_8q_24qv",
                TokenStructuredShape(512, 96, 8, 24, 128, 10, 1024),
            ),
        )
    if name == "large":
        return (
            NamedShape(
                "large_2048d_128dv_8q_32qv",
                TokenStructuredShape(2048, 128, 8, 32, 128, 10, 4096),
            ),
        )
    if name == "moderate_large":
        return shape_presets("moderate") + shape_presets("large")
    raise ValueError(f"unsupported shape set: {name}")


def tac_config_presets(name: str) -> tuple[NamedTacConfig, ...]:
    if name == "smoke":
        return (
            NamedTacConfig(
                "tac_768_cq12_k48",
                TachiomTacConfig(
                    centroid_count=768,
                    centroids_per_query_vector=12,
                    candidate_k=48,
                    kmeans_iterations=6,
                ),
            ),
        )
    if name == "frontier":
        return (
            NamedTacConfig(
                "tac_2k_cq12_k96",
                TachiomTacConfig(
                    centroid_count=2048,
                    centroids_per_query_vector=12,
                    candidate_k=96,
                    kmeans_iterations=6,
                ),
            ),
            NamedTacConfig(
                "tac_2k_cq16_k128",
                TachiomTacConfig(
                    centroid_count=2048,
                    centroids_per_query_vector=16,
                    candidate_k=128,
                    kmeans_iterations=6,
                ),
            ),
            NamedTacConfig(
                "tac_4k_cq24_k256",
                TachiomTacConfig(
                    centroid_count=4096,
                    centroids_per_query_vector=24,
                    candidate_k=256,
                    kmeans_iterations=6,
                ),
            ),
            NamedTacConfig(
                "tac_8k_cq32_k512",
                TachiomTacConfig(
                    centroid_count=8192,
                    centroids_per_query_vector=32,
                    candidate_k=512,
                    kmeans_iterations=6,
                ),
            ),
        )
    raise ValueError(f"unsupported config set: {name}")


def plaid_config_presets(name: str) -> tuple[NamedPlaidConfig, ...]:
    if name == "smoke":
        return (
            NamedPlaidConfig(
                "i8_128_cq32_k48",
                KayakPlaidApproxConfig(
                    centroid_count=128,
                    centroids_per_query_vector=32,
                    candidate_k=48,
                    payload="i8",
                ),
            ),
        )
    if name == "frontier":
        return (
            NamedPlaidConfig(
                "i8_256_cq48_k128",
                KayakPlaidApproxConfig(
                    centroid_count=256,
                    centroids_per_query_vector=48,
                    candidate_k=128,
                    payload="i8",
                ),
            ),
            NamedPlaidConfig(
                "i8_512_cq64_k256",
                KayakPlaidApproxConfig(
                    centroid_count=512,
                    centroids_per_query_vector=64,
                    candidate_k=256,
                    payload="i8",
                ),
            ),
            NamedPlaidConfig(
                "i8_1024_cq96_k512",
                KayakPlaidApproxConfig(
                    centroid_count=1024,
                    centroids_per_query_vector=96,
                    candidate_k=512,
                    payload="i8",
                ),
            ),
            NamedPlaidConfig(
                "i8_2048_cq128_k1024",
                KayakPlaidApproxConfig(
                    centroid_count=2048,
                    centroids_per_query_vector=128,
                    candidate_k=1024,
                    payload="i8",
                ),
            ),
        )
    raise ValueError(f"unsupported config set: {name}")


def run_shape(
    *,
    named_shape: NamedShape,
    inputs: TokenStructuredInputs,
    tac_configs: Sequence[NamedTacConfig],
    plaid_configs: Sequence[NamedPlaidConfig],
    pq_config: TachiomResidualPqConfig,
    engines: Sequence[str],
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    exact_row, reference_positions = benchmark_exact(
        shape=named_shape.shape,
        inputs=inputs,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    )
    exact_row["config_name"] = "exact"
    systems: list[dict[str, Any]] = [exact_row]

    if "tachiom_tac" in engines:
        for named_config in tac_configs:
            row = benchmark_tachiom_tac(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    if "tachiom_tac_mojo" in engines:
        for named_config in tac_configs:
            row = benchmark_tachiom_tac_mojo(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    if "tachiom_tac_i8_mojo" in engines:
        for named_config in tac_configs:
            row = benchmark_tachiom_tac_i8_mojo(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    if "tachiom_tac_pq" in engines:
        for named_config in tac_configs:
            row = benchmark_tachiom_tac_pq(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                pq_config=pq_config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    if "tachiom_tac_pq_mojo" in engines:
        for named_config in tac_configs:
            row = benchmark_tachiom_tac_pq_mojo(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                pq_config=pq_config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    if "kayak_plaid" in engines:
        for named_config in plaid_configs:
            row = benchmark_kayak_plaid(
                shape=named_shape.shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=named_config.config,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
            )
            row["config_name"] = named_config.name
            systems.append(row)

    return {
        "shape_name": named_shape.name,
        "shape": named_shape.shape.to_json_ready(),
        "systems": systems,
        "pairwise_vs_exact": build_pairwise_rows(systems),
        "pareto_rows": _pareto_rows(systems),
        "best_full_recall_by_qps": _best_full_recall_by_qps(systems),
    }


def _pareto_rows(systems: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ok_rows = [
        row for row in systems
        if row.get("status") == "ok" and row.get("engine") != "exact"
    ]
    pareto: list[dict[str, Any]] = []
    for row in ok_rows:
        dominated = False
        for other in ok_rows:
            if other is row:
                continue
            if _dominates(other, row):
                dominated = True
                break
        if not dominated:
            pareto.append(_compact_row(row))
    return pareto


def _dominates(lhs: dict[str, Any], rhs: dict[str, Any]) -> bool:
    lhs_recall = float(lhs.get("final_recall_at_k_vs_exact") or 0.0)
    rhs_recall = float(rhs.get("final_recall_at_k_vs_exact") or 0.0)
    lhs_qps = float(lhs.get("query_qps") or 0.0)
    rhs_qps = float(rhs.get("query_qps") or 0.0)
    lhs_bytes = float(lhs.get("index_bytes") or float("inf"))
    rhs_bytes = float(rhs.get("index_bytes") or float("inf"))
    return (
        lhs_recall >= rhs_recall
        and lhs_qps >= rhs_qps
        and lhs_bytes <= rhs_bytes
        and (lhs_recall > rhs_recall or lhs_qps > rhs_qps or lhs_bytes < rhs_bytes)
    )


def _best_full_recall_by_qps(systems: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in systems
        if row.get("status") == "ok"
        and row.get("engine") != "exact"
        and float(row.get("final_recall_at_k_vs_exact") or 0.0) >= 1.0
    ]
    if not candidates:
        return None
    return _compact_row(max(candidates, key=lambda row: float(row["query_qps"])))


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "engine": row.get("engine"),
        "system_name": row.get("system_name"),
        "config_name": row.get("config_name"),
        "candidate_recall_at_k_vs_exact": row.get("candidate_recall_at_k_vs_exact"),
        "final_recall_at_k_vs_exact": row.get("final_recall_at_k_vs_exact"),
        "query_qps": row.get("query_qps"),
        "query_batch_mean_seconds": row.get("query_batch_mean_seconds"),
        "index_bytes": row.get("index_bytes"),
        "build_seconds": row.get("build_seconds"),
        "centroid_count": row.get("centroid_count"),
        "centroids_per_query_vector": row.get("centroids_per_query_vector"),
        "candidate_k": row.get("candidate_k"),
        "payload": row.get("payload"),
        "pq_subspace_count": row.get("pq_subspace_count"),
        "pq_effective_codebook_size": row.get("pq_effective_codebook_size"),
    }


def main() -> None:
    args = parse_args()
    tac_configs = tac_config_presets(args.config_set)
    plaid_configs = plaid_config_presets(args.config_set)
    pq_config = TachiomResidualPqConfig(
        subspace_count=args.tac_pq_subspace_count,
        codebook_size=args.tac_pq_codebook_size,
        kmeans_iterations=args.tac_pq_kmeans_iterations,
        training_sample_count=(
            None
            if args.tac_pq_training_sample_count <= 0
            else args.tac_pq_training_sample_count
        ),
    )
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
                plaid_configs=plaid_configs,
                pq_config=pq_config,
                engines=args.engines,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper": {
            "name": "Efficient Multivector Retrieval with Token-Aware Clustering and Hierarchical Indexing",
            "url": TACHIOM_PAPER_URL,
        },
        "epistemic_status": {
            "claim": (
                "This sweep compares Kayak's first-gate TAC exact-centroid "
                "probe against the existing i8 PLAID-style lane. It is not a "
                "full Tachiom HNSW/PQ implementation."
            ),
            "promotion_rule": (
                "Promote TAC into Mojo/HNSW only if the full-recall frontier "
                "beats or complements existing Kayak i8 points on explicit "
                "vector-count shapes."
            ),
        },
        "shape_set": args.shape_set,
        "config_set": args.config_set,
        "engines": list(args.engines),
        "warmup_iterations": args.warmup_iterations,
        "measurement_iterations": args.measurement_iterations,
        "shapes": shape_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        for shape_report in shape_reports:
            emit_quiet_mean_sections(
                shape_report["systems"],
                label_prefix=str(shape_report["shape_name"]),
            )


if __name__ == "__main__":
    main()
