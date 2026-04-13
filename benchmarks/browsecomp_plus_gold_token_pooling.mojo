from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    TokenPoolingSummary,
    build_token_pooling_summary,
    standard_token_pooling_factors,
    supported_token_pooling_policies,
    token_pooling_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = output_root / "browsecomp_plus_gold_token_pooling"
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[TokenPoolingSummary]()
    var max_pool_factor = cache.stored_task.task.nominal_document_vector_count

    for pooling_policy in supported_token_pooling_policies():
        for pool_factor in standard_token_pooling_factors(max_pool_factor):
            var summary = build_token_pooling_summary(
                backend,
                cache.stored_task,
                cache.stored_index,
                benchmark_root
                / (pooling_policy + "_pool_" + String(pool_factor)),
                pool_factor,
                pooling_policy,
            )
            print(
                "policy=",
                summary.pooling_policy,
                " pool_factor=",
                summary.pool_factor,
                " ref_recall=",
                summary.mean_reference_recall_at_k,
                " ndcg=",
                summary.mean_ndcg_at_k,
                " search_s=",
                summary.mean_search_seconds,
                " bytes/vector=",
                summary.artifact_bytes_per_vector,
            )
            print(
                "Mean: ",
                summary.mean_search_seconds,
                " policy=",
                summary.pooling_policy,
                " pool_factor=",
                summary.pool_factor,
            )
            summaries.append(summary.copy())

    (output_root / "browsecomp_plus_gold_token_pooling.json").write_text(
        token_pooling_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "browsecomp_plus_gold_token_pooling.json"))
