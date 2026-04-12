from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    VectorPruningSummary,
    build_vector_pruning_summary,
    standard_vector_pruning_budget_sizes,
    vector_pruning_summaries_json,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var benchmark_root = output_root / "browsecomp_plus_gold_vector_pruning"
    makedirs(benchmark_root, exist_ok=True)
    var summaries = List[VectorPruningSummary]()
    var max_budget = cache.stored_task.task.nominal_document_vector_count

    for document_vector_budget in standard_vector_pruning_budget_sizes(max_budget):
        var summary = build_vector_pruning_summary(
            backend,
            cache.stored_task,
            cache.stored_index,
            benchmark_root / ("doc_budget_" + String(document_vector_budget)),
            document_vector_budget,
        )
        print(
            "doc_budget=",
            summary.document_vector_budget,
            " ref_recall=",
            summary.mean_reference_recall_at_k,
            " ndcg=",
            summary.mean_ndcg_at_k,
            " search_s=",
            summary.mean_search_seconds,
            " bytes/vector=",
            summary.artifact_bytes_per_vector,
        )
        summaries.append(summary.copy())
    (output_root / "browsecomp_plus_gold_vector_pruning.json").write_text(
        vector_pruning_summaries_json(summaries)
    )
    print("wrote ", String(output_root / "browsecomp_plus_gold_vector_pruning.json"))
