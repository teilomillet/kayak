from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
    MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
    MultiVectorIndexCompressionSummary,
    build_multi_vector_index_compression_summary,
    multi_vector_index_compression_summaries_json,
    standard_multi_vector_index_compression_budget_sizes,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = (
        output_root / "browsecomp_plus_gold_multi_vector_index_compression"
    )
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[MultiVectorIndexCompressionSummary]()
    var max_budget = cache.stored_task.task.nominal_document_vector_count

    var full_summary = build_multi_vector_index_compression_summary(
        backend,
        cache.stored_task,
        cache.stored_index,
        benchmark_root / MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
        max_budget,
    )
    print(
        "method=",
        full_summary.method_kind,
        " boundary=",
        full_summary.execution_boundary,
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

    for document_vector_budget in standard_multi_vector_index_compression_budget_sizes(
        max_budget
    ):
        var pooling_summary = build_multi_vector_index_compression_summary(
            backend,
            cache.stored_task,
            cache.stored_index,
            benchmark_root
            / (
                MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING
                + "_"
                + String(document_vector_budget)
            ),
            MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
            document_vector_budget,
        )
        print(
            "method=",
            pooling_summary.method_kind,
            " boundary=",
            pooling_summary.execution_boundary,
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
        output_root / "browsecomp_plus_gold_multi_vector_index_compression.json"
    ).write_text(multi_vector_index_compression_summaries_json(summaries))
    print(
        "wrote ",
        String(
            output_root / "browsecomp_plus_gold_multi_vector_index_compression.json"
        ),
    )
