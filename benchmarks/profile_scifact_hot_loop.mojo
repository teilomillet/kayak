from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import ensure_scifact_real_subset_cache


def main() raises:
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var backend = ExactCpuBackend()
    var iterations = 2000000
    var hit_count = 0

    print("profiling SciFact hot loop")
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("iterations: ", iterations)

    for iteration in range(iterations):
        var hits = search_exact(
            backend,
            task.queries[iteration % len(task.queries)].query,
            index,
            task.k,
        )
        hit_count += len(hits)

    print("completed iterations: ", iterations)
    print("total_hits: ", hit_count)
