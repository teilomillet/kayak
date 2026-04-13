from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    TokenPoolingStageAwareSummary,
    build_token_pooling_stage_aware_summary,
    standard_candidate_window_sizes,
    standard_token_pooling_factors,
    supported_token_pooling_policies,
    token_pooling_stage_aware_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = output_root / "browsecomp_plus_gold_token_pooling_stage_aware"
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[TokenPoolingStageAwareSummary]()
    var max_pool_factor = cache.stored_task.task.nominal_document_vector_count
    var candidate_window_sizes = standard_candidate_window_sizes(
        cache.stored_task.task.k,
        len(cache.stored_task.task.documents),
    )

    for pooling_policy in supported_token_pooling_policies():
        for pool_factor in standard_token_pooling_factors(max_pool_factor):
            for candidate_k in candidate_window_sizes:
                var summary = build_token_pooling_stage_aware_summary(
                    backend,
                    cache.stored_task,
                    cache.stored_index,
                    benchmark_root
                    / (
                        pooling_policy
                        + "_pool_"
                        + String(pool_factor)
                        + "_candidate_"
                        + String(candidate_k)
                    ),
                    pool_factor,
                    pooling_policy,
                    candidate_k,
                )
                print(
                    "policy=",
                    summary.pooling_policy,
                    " pool_factor=",
                    summary.pool_factor,
                    " candidate_k=",
                    summary.candidate_k,
                    " stage1_recall=",
                    summary.mean_stage1_candidate_recall_at_final_k,
                    " restored_recall=",
                    summary.mean_restored_reference_recall_at_final_k,
                    " restored_ndcg=",
                    summary.mean_restored_ndcg_at_k,
                    " total_s=",
                    summary.mean_restored_search_seconds,
                )
                print(
                    "Mean: ",
                    summary.mean_restored_search_seconds,
                    " policy=",
                    summary.pooling_policy,
                    " pool_factor=",
                    summary.pool_factor,
                    " candidate_k=",
                    summary.candidate_k,
                )
                summaries.append(summary.copy())

    (output_root / "browsecomp_plus_gold_token_pooling_stage_aware.json").write_text(
        token_pooling_stage_aware_summaries_json(summaries)
    )
    print(
        "wrote ",
        String(output_root / "browsecomp_plus_gold_token_pooling_stage_aware.json"),
    )
