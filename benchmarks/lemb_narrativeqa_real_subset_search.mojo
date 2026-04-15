import std.benchmark as benchmark
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    build_real_slice_benchmark_summary_from_measurement,
    real_slice_benchmark_summary_json,
)
from kayak.eval import evaluate_task
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import ensure_lemb_narrativeqa_real_subset_cache


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def main() raises:
    print("loading real LEMB NarrativeQA subset with storage...")
    var cache = ensure_lemb_narrativeqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var backend = ExactCpuBackend()
    var query_index = 0
    var task_source = source_label(
        cache.loaded_task_from_storage, "colbert_cpu_encode"
    )
    var index_source = source_label(
        cache.loaded_index_from_storage, "pack_documents"
    )

    print("family: ", task.family)
    print("slice: ", task.slice_name)
    print("task_source: ", task_source)
    print("index_source: ", index_source)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)

    def score_once() capturing raises:
        _ = search_exact(backend, task.queries[query_index].query, index, task.k)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()

    var evaluation = evaluate_task(backend, task)
    var summary = build_real_slice_benchmark_summary_from_measurement(
        cache.stored_task.dataset_id.copy(),
        cache.stored_task.model_name.copy(),
        task,
        task_source,
        index_source,
        evaluation,
        Float64(report.mean()),
    )
    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "lemb_narrativeqa_real_subset_benchmark.json"
    output_path.write_text(real_slice_benchmark_summary_json(summary))
    print("wrote ", String(output_path))
