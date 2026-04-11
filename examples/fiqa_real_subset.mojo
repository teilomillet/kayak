from kayak.eval import evaluate_task
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import ensure_fiqa_real_subset_cache


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def main() raises:
    print("loading real BEIR/FIQA subset with storage...")
    var cache = ensure_fiqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var backend = ExactCpuBackend()
    var evaluation = evaluate_task(backend, task.copy())
    var hits = search_exact(backend, task.queries[0].query.copy(), index, task.k)

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
    print("mrr@", task.k, " = ", evaluation.mean_reciprocal_rank)
    print("recall@", task.k, " = ", evaluation.mean_recall_at_k)
    print("success@", task.k, " = ", evaluation.success_rate_at_k)
    print("")
    print("top hits for first query: ", task.queries[0].query_id)

    var limit = 3
    if len(hits) < limit:
        limit = len(hits)

    for hit_index in range(limit):
        print(hits[hit_index].doc_id, " : ", hits[hit_index].score)
