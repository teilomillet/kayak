from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak

from kayak_bridge.r2med_biology_late_interaction_quality import (
    DEFAULT_VARIANT_CACHE_ROOT,
    benchmark_r2med_late_interaction_policies,
)
from kayak_bridge.r2med_biology_full import DEFAULT_SNAPSHOT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark pure late-interaction query variants and weighted score "
            "fusion policies on the full R2MED/Biology snapshot."
        )
    )
    parser.add_argument("--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument(
        "--variant-cache-root",
        type=Path,
        default=DEFAULT_VARIANT_CACHE_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / ".cache"
            / "kayak"
            / "r2med_biology_late_interaction_policies_summary.json"
        ),
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--backend", type=str, default=kayak.MOJO_EXACT_CPU_BACKEND)
    parser.add_argument("--force-rebuild-query-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = benchmark_r2med_late_interaction_policies(
        snapshot_root=args.snapshot_root,
        variant_cache_root=args.variant_cache_root,
        k=args.k,
        backend=args.backend,
        force_rebuild_query_variants=args.force_rebuild_query_variants,
    )
    rows = sorted(summary.rows, key=lambda row: row.mean_ndcg_at_k, reverse=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    best = rows[0]
    print(f"wrote {args.output}")
    print(
        "best policy="
        f"{best.policy_name}, "
        f"nDCG@10={best.mean_ndcg_at_k:.6f}, "
        f"MRR@10={best.mean_reciprocal_rank:.6f}"
    )
    print(f"Mean: {best.mean_ndcg_at_k:.6f}")


if __name__ == "__main__":
    main()
