import std.benchmark as benchmark

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import ExactCpuBackend, ExactScoringConfig
from kayak.benchmarks import (
    BackendPolicyBenchmarkSummary,
    backend_policy_benchmark_summaries_json,
    build_backend_policy_benchmark_summary,
)
from kayak.search import search_exact
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def default_backend() -> ExactCpuBackend:
    return ExactCpuBackend()


def serial_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def conservative_parallel_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    return ExactCpuBackend(config^)


def fixed_work_item_backend(work_items: Int) -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = work_items
    return ExactCpuBackend(config^)


def benchmark_backend(
    read task_source: String,
    read index_source: String,
    backend_name: String,
    read backend: ExactCpuBackend,
) raises -> BackendPolicyBenchmarkSummary:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var query_index = 0

    print("backend: ", backend_name)

    def score_once() capturing raises:
        _ = search_exact(backend, task.queries[query_index].query, index, task.k)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")

    return build_backend_policy_benchmark_summary(
        "BrowseComp-Plus Gold",
        task,
        task_source,
        index_source,
        backend_name,
        Float64(report.mean()),
    )


def main() raises:
    print("loading official BrowseComp-Plus gold slice with storage...")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    print("slice: ", cache.stored_task.task.slice_name)
    print("queries: ", len(cache.stored_task.task.queries))
    print("documents: ", len(cache.stored_task.task.documents))
    print("query_vectors≈ ", cache.stored_task.task.nominal_query_vector_count)
    print("doc_vectors≈ ", cache.stored_task.task.nominal_document_vector_count)
    print("vector_dim: ", cache.stored_task.task.vector_dim)
    print("")

    var summaries = List[BackendPolicyBenchmarkSummary]()
    summaries.append(
        benchmark_backend(
            task_source,
            index_source,
            "default",
            default_backend(),
        )
    )
    summaries.append(
        benchmark_backend(
            task_source,
            index_source,
            "serial",
            serial_backend(),
        )
    )
    summaries.append(
        benchmark_backend(
            task_source,
            index_source,
            "parallel_oversubscription_disabled",
            conservative_parallel_backend(),
        )
    )

    for work_items in [2, 4, 8, 16]:
        summaries.append(
            benchmark_backend(
                task_source,
                index_source,
                "work_items=" + String(work_items),
                fixed_work_item_backend(work_items),
            )
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "browsecomp_plus_gold_exact_policy_sweep.json"
    output_path.write_text(backend_policy_benchmark_summaries_json(summaries))
    print("wrote ", String(output_path))
