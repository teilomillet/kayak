from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
    TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
    TrainingFreeSequenceCompressionSummary,
    build_training_free_sequence_compression_summary,
    standard_training_free_sequence_compression_budget_sizes,
    supported_token_pooling_policies,
    training_free_sequence_compression_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = (
        output_root / "browsecomp_plus_gold_training_free_sequence_compression"
    )
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[TrainingFreeSequenceCompressionSummary]()
    var max_budget = cache.stored_task.task.nominal_document_vector_count

    var full_summary = build_training_free_sequence_compression_summary(
        backend,
        cache.stored_task,
        cache.stored_index,
        benchmark_root / TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
        TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT,
        max_budget,
    )
    print(
        "method=",
        full_summary.method_kind,
        " ref_recall=",
        full_summary.mean_reference_recall_at_k,
        " ndcg=",
        full_summary.mean_ndcg_at_k,
        " search_s=",
        full_summary.mean_search_seconds,
        " bytes/vector=",
        full_summary.artifact_bytes_per_vector,
    )
    summaries.append(full_summary.copy())

    for document_vector_budget in standard_training_free_sequence_compression_budget_sizes(
        max_budget
    ):
        var pruning_summary = build_training_free_sequence_compression_summary(
            backend,
            cache.stored_task,
            cache.stored_index,
            benchmark_root / ("prefix_pruning_" + String(document_vector_budget)),
            TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING,
            document_vector_budget,
        )
        print(
            "method=",
            pruning_summary.method_kind,
            " doc_budget=",
            pruning_summary.requested_document_vector_budget,
            " ref_recall=",
            pruning_summary.mean_reference_recall_at_k,
            " ndcg=",
            pruning_summary.mean_ndcg_at_k,
            " search_s=",
            pruning_summary.mean_search_seconds,
            " bytes/vector=",
            pruning_summary.artifact_bytes_per_vector,
        )
        summaries.append(pruning_summary.copy())

        for pooling_policy in supported_token_pooling_policies():
            var pooling_summary = build_training_free_sequence_compression_summary(
                backend,
                cache.stored_task,
                cache.stored_index,
                benchmark_root
                / (
                    pooling_policy
                    + "_token_pooling_"
                    + String(document_vector_budget)
                ),
                TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING,
                document_vector_budget,
                pooling_policy,
            )
            print(
                "method=",
                pooling_summary.method_kind,
                " policy=",
                pooling_summary.transform_policy,
                " doc_budget=",
                pooling_summary.requested_document_vector_budget,
                " ref_recall=",
                pooling_summary.mean_reference_recall_at_k,
                " ndcg=",
                pooling_summary.mean_ndcg_at_k,
                " search_s=",
                pooling_summary.mean_search_seconds,
                " bytes/vector=",
                pooling_summary.artifact_bytes_per_vector,
            )
            summaries.append(pooling_summary.copy())

    (
        output_root / "browsecomp_plus_gold_training_free_sequence_compression.json"
    ).write_text(training_free_sequence_compression_summaries_json(summaries))
    print(
        "wrote ",
        String(
            output_root
            / "browsecomp_plus_gold_training_free_sequence_compression.json"
        ),
    )
