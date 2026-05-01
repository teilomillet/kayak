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

from kayak_bridge.tachiom_probe import TachiomResidualPqConfig  # noqa: E402

from bench_tachiom_tac_probe import (  # noqa: E402
    TACHIOM_PAPER_URL,
    benchmark_exact,
    benchmark_tachiom_tac_pq_mojo,
    build_pairwise_rows,
    build_token_structured_inputs,
    emit_quiet_mean_sections,
)
from bench_tachiom_tac_sweep import (  # noqa: E402
    NamedShape,
    NamedTacConfig,
    shape_presets,
    tac_config_presets,
)


@dataclass(frozen=True, slots=True)
class NamedPqConfig:
    name: str
    config: TachiomResidualPqConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep residual-PQ refine shapes for the Tachiom TAC candidate "
            "window. This isolates PQ quality/byte/speed tradeoffs after TAC "
            "candidate generation has full recall."
        )
    )
    parser.add_argument(
        "--shape-set",
        choices=("smoke", "moderate", "large", "moderate_large"),
        default="smoke",
    )
    parser.add_argument(
        "--config-set",
        choices=("smoke", "frontier"),
        default="smoke",
    )
    parser.add_argument(
        "--tac-config-name",
        default=None,
        help="Optional TAC config name from the selected config set.",
    )
    parser.add_argument(
        "--pq-subspace-counts",
        type=_parse_positive_ints,
        default=(16, 32, 64, 128),
    )
    parser.add_argument(
        "--pq-codebook-sizes",
        type=_parse_positive_ints,
        default=(256,),
    )
    parser.add_argument("--pq-kmeans-iterations", type=int, default=6)
    parser.add_argument(
        "--pq-training-sample-count",
        type=int,
        default=0,
        help="Optional residual-PQ training sample count; 0 uses all tokens.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--zipf-skew", type=float, default=1.1)
    parser.add_argument("--document-noise", type=float, default=0.08)
    parser.add_argument("--query-noise", type=float, default=0.04)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/tachiom_tac_pq_sweep/summary.json"),
    )
    parser.add_argument("--emit-quiet-mean", action="store_true")
    return parser.parse_args()


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one value is required")
    invalid = tuple(item for item in values if item <= 0)
    if invalid:
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def pq_config_grid(
    *,
    subspace_counts: Sequence[int],
    codebook_sizes: Sequence[int],
    kmeans_iterations: int,
    training_sample_count: int | None,
) -> tuple[NamedPqConfig, ...]:
    rows: list[NamedPqConfig] = []
    for subspace_count in subspace_counts:
        for codebook_size in codebook_sizes:
            rows.append(
                NamedPqConfig(
                    f"pq_m{subspace_count}_k{codebook_size}",
                    TachiomResidualPqConfig(
                        subspace_count=subspace_count,
                        codebook_size=codebook_size,
                        kmeans_iterations=kmeans_iterations,
                        training_sample_count=training_sample_count,
                    ),
                )
            )
    return tuple(rows)


def choose_tac_config(
    configs: Sequence[NamedTacConfig],
    *,
    name: str | None,
) -> NamedTacConfig:
    if name is None:
        return configs[0]
    for config in configs:
        if config.name == name:
            return config
    raise ValueError(f"unknown TAC config name for selected set: {name}")


def run_shape(
    *,
    named_shape: NamedShape,
    named_tac_config: NamedTacConfig,
    pq_configs: Sequence[NamedPqConfig],
    warmup_iterations: int,
    measurement_iterations: int,
    seed: int,
    zipf_skew: float,
    document_noise: float,
    query_noise: float,
) -> dict[str, Any]:
    inputs = build_token_structured_inputs(
        named_shape.shape,
        seed=seed,
        zipf_skew=zipf_skew,
        document_noise=document_noise,
        query_noise=query_noise,
    )
    exact_row, reference_positions = benchmark_exact(
        shape=named_shape.shape,
        inputs=inputs,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    )
    exact_row["config_name"] = "exact"
    systems: list[dict[str, Any]] = [exact_row]

    for pq_config in pq_configs:
        row = benchmark_tachiom_tac_pq_mojo(
            shape=named_shape.shape,
            inputs=inputs,
            reference_positions_by_query=reference_positions,
            config=named_tac_config.config,
            pq_config=pq_config.config,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
        row["config_name"] = f"{named_tac_config.name}_{pq_config.name}"
        systems.append(row)

    return {
        "shape_name": named_shape.name,
        "shape": named_shape.shape.to_json_ready(),
        "tac_config_name": named_tac_config.name,
        "systems": systems,
        "pairwise_vs_exact": build_pairwise_rows(systems),
        "best_by_recall_then_qps": _best_by_recall_then_qps(systems),
    }


def _best_by_recall_then_qps(systems: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in systems
        if row.get("status") == "ok" and row.get("engine") != "exact"
    ]
    if not candidates:
        return None
    row = max(
        candidates,
        key=lambda item: (
            float(item.get("final_recall_at_k_vs_exact") or 0.0),
            float(item.get("query_qps") or 0.0),
        ),
    )
    return {
        "config_name": row.get("config_name"),
        "final_recall_at_k_vs_exact": row.get("final_recall_at_k_vs_exact"),
        "candidate_recall_at_k_vs_exact": row.get("candidate_recall_at_k_vs_exact"),
        "query_qps": row.get("query_qps"),
        "index_bytes": row.get("index_bytes"),
        "pq_subspace_count": row.get("pq_subspace_count"),
        "pq_effective_codebook_size": row.get("pq_effective_codebook_size"),
        "pq_training_sample_count": row.get("pq_training_sample_count"),
    }


def main() -> None:
    args = parse_args()
    selected_tac_config = choose_tac_config(
        tac_config_presets(args.config_set),
        name=args.tac_config_name,
    )
    pq_configs = pq_config_grid(
        subspace_counts=args.pq_subspace_counts,
        codebook_sizes=args.pq_codebook_sizes,
        kmeans_iterations=args.pq_kmeans_iterations,
        training_sample_count=(
            None
            if args.pq_training_sample_count <= 0
            else args.pq_training_sample_count
        ),
    )
    shape_reports = [
        run_shape(
            named_shape=named_shape,
            named_tac_config=selected_tac_config,
            pq_configs=pq_configs,
            warmup_iterations=args.warmup_iterations,
            measurement_iterations=args.measurement_iterations,
            seed=args.seed,
            zipf_skew=args.zipf_skew,
            document_noise=args.document_noise,
            query_noise=args.query_noise,
        )
        for named_shape in shape_presets(args.shape_set)
    ]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "paper": {
            "name": (
                "Efficient Multivector Retrieval with Token-Aware Clustering "
                "and Hierarchical Indexing"
            ),
            "url": TACHIOM_PAPER_URL,
        },
        "epistemic_status": {
            "claim": (
                "Residual-PQ is a paper-shaped refine payload. This sweep "
                "tests subspace/codebook tradeoffs after TAC candidate recall, "
                "not HNSW candidate generation."
            ),
        },
        "shape_set": args.shape_set,
        "config_set": args.config_set,
        "tac_config_name": selected_tac_config.name,
        "pq_configs": [config.name for config in pq_configs],
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
