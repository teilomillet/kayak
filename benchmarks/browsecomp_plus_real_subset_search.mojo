import std.benchmark as benchmark

from kayak.eval import evaluate_task
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import ensure_browsecomp_plus_real_subset_cache


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def main() raises:
    print("loading official BrowseComp-Plus evidence slice with storage...")
    var cache = ensure_browsecomp_plus_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var backend = ExactCpuBackend()
    var evaluation = evaluate_task(backend, task)
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
    print("primary@", task.k, ": ", evaluation.primary_metric, " = ", evaluation.primary_value)
    print("ndcg@", task.k, " = ", evaluation.mean_ndcg_at_k)
    print("mrr@", task.k, " = ", evaluation.mean_reciprocal_rank)
    print("recall@", task.k, " = ", evaluation.mean_recall_at_k)
    print("success@", task.k, " = ", evaluation.success_rate_at_k)

    def score_once() capturing raises:
        _ = search_exact(backend, task.queries[query_index].query, index, task.k)
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
