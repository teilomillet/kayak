import std.benchmark as benchmark

from kayak import (
    ExactCpuBackend,
    ExactScoringConfig,
    JudgedTask,
    PackedIndex,
)
from kayak.search import search_exact
from kayak.storage import (
    ensure_fiqa_real_subset_cache,
    ensure_scifact_real_subset_cache,
)


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def default_backend() -> ExactCpuBackend:
    return ExactCpuBackend()


def conservative_parallel_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    return ExactCpuBackend(config^)


def benchmark_task(
    dataset_name: String,
    slice_name: String,
    task_source: String,
    index_source: String,
    read task: JudgedTask,
    read index: PackedIndex,
    backend_name: String,
    read backend: ExactCpuBackend,
) raises:
    print("dataset: ", dataset_name)
    print("slice: ", slice_name)
    print("task_source: ", task_source)
    print("index_source: ", index_source)
    print("backend: ", backend_name)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)

    var query_index = 0

    def score_once() capturing raises:
        _ = search_exact(backend, task.queries[query_index].query, index, task.k)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")


def benchmark_scifact_real_subset() raises:
    print("loading real BEIR/SciFact subset with storage...")
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "default",
        default_backend(),
    )
    benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "parallel_oversubscription_disabled",
        conservative_parallel_backend(),
    )


def benchmark_fiqa_real_subset() raises:
    print("loading real BEIR/FIQA subset with storage...")
    var cache = ensure_fiqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "default",
        default_backend(),
    )
    benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "parallel_oversubscription_disabled",
        conservative_parallel_backend(),
    )


def main() raises:
    benchmark_scifact_real_subset()
    benchmark_fiqa_real_subset()
