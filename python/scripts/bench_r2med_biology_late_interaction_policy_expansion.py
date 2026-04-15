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

from kayak_bridge.r2med_biology_full import DEFAULT_SNAPSHOT_ROOT
from kayak_bridge.r2med_biology_late_interaction_quality import (
    DEFAULT_VARIANT_CACHE_ROOT,
)
from kayak_bridge.r2med_biology_late_interaction_policy_expansion import (
    benchmark_r2med_policy_expansions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search pure late-interaction seed-policy expansions on the full "
            "R2MED/Biology snapshot."
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
            / "r2med_biology_late_interaction_policy_expansion_summary.json"
        ),
    )
    parser.add_argument("--shortlist-k", type=int, default=30)
    parser.add_argument("--match-k", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--backend", type=str, default=kayak.MOJO_EXACT_CPU_BACKEND)
    parser.add_argument("--force-rebuild-query-variants", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = benchmark_r2med_policy_expansions(
        snapshot_root=args.snapshot_root,
        variant_cache_root=args.variant_cache_root,
        shortlist_k=args.shortlist_k,
        match_k=args.match_k,
        k=args.k,
        backend=args.backend,
        force_rebuild_query_variants=args.force_rebuild_query_variants,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_json_ready(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    best_maxsim = summary.maxsim_expansion_rows[0]
    best_rescored = summary.rescored_expansion_rows[0]
    print(f"wrote {args.output}")
    print(
        "best maxsim policy="
        f"{best_maxsim.policy_name}, "
        f"nDCG@10={best_maxsim.mean_ndcg_at_k:.6f}"
    )
    print(
        "best rescored policy="
        f"{best_rescored.policy_name}, "
        f"nDCG@10={best_rescored.mean_ndcg_at_k:.6f}, "
        f"MRR@10={best_rescored.mean_reciprocal_rank:.6f}, "
        f"variant={summary.rescored_variant_name}"
    )
    print(f"Mean: {best_rescored.mean_ndcg_at_k:.6f}")


if __name__ == "__main__":
    main()
