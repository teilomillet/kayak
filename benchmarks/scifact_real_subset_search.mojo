import std.benchmark as benchmark

from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import ensure_scifact_real_subset_cache


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def main() raises:
    print("loading real BEIR/SciFact subset with storage...")
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var backend = ExactCpuBackend()
    var query_index = 0

    print("family: ", task.family)
    print("slice: ", task.slice_name)
    print(
        "task_source: ",
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
    )
    print(
        "index_source: ",
        source_label(cache.loaded_index_from_storage, "pack_documents"),
    )
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)

    def score_once() capturing raises:
        _ = search_exact(
            backend, task.queries[query_index].query.copy(), index.copy(), task.k
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
