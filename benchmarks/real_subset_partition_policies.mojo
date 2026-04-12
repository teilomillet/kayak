import std.benchmark as benchmark

from std.collections import List
from std.os import makedirs
from std.pathlib import Path
from std.runtime.asyncrt import parallelism_level

from kayak import ExactCpuBackend, ExactScoringConfig, JudgedTask, PackedIndex
from kayak.benchmarks import (
    BackendPolicyBenchmarkSummary,
    backend_policy_benchmark_summaries_json,
    build_backend_policy_benchmark_summary,
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


def serial_backend() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def fixed_work_item_backend(work_items: Int) -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.parallel_work_item_count_override = work_items
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
) raises -> BackendPolicyBenchmarkSummary:
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
    return build_backend_policy_benchmark_summary(
        dataset_name,
        task,
        task_source,
        index_source,
        backend_name,
        Float64(report.mean()),
    )


def benchmark_scifact_real_subset(
    mut summaries: List[BackendPolicyBenchmarkSummary]
) raises:
    print("loading real BEIR/SciFact subset with storage...")
    print("parallelism_level: ", parallelism_level())
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    summaries.append(benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "default",
        default_backend(),
    ))
    summaries.append(benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "serial",
        serial_backend(),
    ))
    summaries.append(benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "parallel_oversubscription_disabled",
        conservative_parallel_backend(),
    ))
    summaries.append(benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "work_items=" + String(parallelism_level()),
        fixed_work_item_backend(parallelism_level()),
    ))
    summaries.append(benchmark_task(
        "SciFact",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "work_items=" + String(parallelism_level() * 4),
        fixed_work_item_backend(parallelism_level() * 4),
    ))


def benchmark_fiqa_real_subset(
    mut summaries: List[BackendPolicyBenchmarkSummary]
) raises:
    print("loading real BEIR/FIQA subset with storage...")
    print("parallelism_level: ", parallelism_level())
    var cache = ensure_fiqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    summaries.append(benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "default",
        default_backend(),
    ))
    summaries.append(benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "serial",
        serial_backend(),
    ))
    summaries.append(benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "parallel_oversubscription_disabled",
        conservative_parallel_backend(),
    ))
    summaries.append(benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "work_items=" + String(parallelism_level()),
        fixed_work_item_backend(parallelism_level()),
    ))
    summaries.append(benchmark_task(
        "FIQA",
        task.slice_name.copy(),
        task_source,
        index_source,
        task,
        index,
        "work_items=" + String(parallelism_level() * 4),
        fixed_work_item_backend(parallelism_level() * 4),
    ))


def main() raises:
    var summaries = List[BackendPolicyBenchmarkSummary]()
    benchmark_scifact_real_subset(summaries)
    benchmark_fiqa_real_subset(summaries)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "public_partition_policy_benchmarks.json"
    output_path.write_text(backend_policy_benchmark_summaries_json(summaries))
    print("wrote ", String(output_path))
