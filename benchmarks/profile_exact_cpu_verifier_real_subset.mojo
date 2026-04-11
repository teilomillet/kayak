import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler
from std.collections import List

from kayak import ExactCpuBackend, JudgedTask, PackedIndex, SearchHit
from kayak.search import search_exact
from kayak.storage import (
    ensure_fiqa_real_subset_cache,
    ensure_scifact_real_subset_cache,
)
from kayak.verifier import (
    exact_late_interaction_verifier,
    no_verifier,
    rerank_hits_with_verifier,
    search_exact_with_verifier,
)


def source_label(loaded_from_storage: Bool, fresh_label: String) -> String:
    if loaded_from_storage:
        return "storage"

    return fresh_label.copy()


def default_backend() -> ExactCpuBackend:
    return ExactCpuBackend()


def print_fixture_header(
    dataset_name: String,
    task: JudgedTask,
    task_source: String,
    index_source: String,
):
    print("dataset: ", dataset_name)
    print("slice: ", task.slice_name)
    print("task_source: ", task_source)
    print("index_source: ", index_source)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("final_k: ", task.k)
    print("")


def precompute_candidate_hits(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    candidate_k: Int,
) raises -> List[List[SearchHit]]:
    var candidate_hits_by_query = List[List[SearchHit]]()

    for judged_query in task.queries:
        candidate_hits_by_query.append(
            search_exact(backend, judged_query.query, index, candidate_k)
        )

    return candidate_hits_by_query^


def benchmark_search_exact(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
) raises:
    print("== search_exact ==")
    var query_index = 0

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact(backend, task.queries[query_index].query, index, task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[search_once]().print()
    print("")


def benchmark_rerank_noop(
    read task: JudgedTask,
    read index: PackedIndex,
    read candidate_hits_by_query: List[List[SearchHit]],
) raises:
    print("== rerank noop verifier ==")
    var query_index = 0
    var verifier = no_verifier()

    def rerank_once() capturing raises:
        bench_compiler.keep(
            rerank_hits_with_verifier(
                task.queries[query_index].query,
                index,
                candidate_hits_by_query[query_index],
                task.k,
                verifier,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[rerank_once]().print()
    print("")


def benchmark_rerank_exact(
    read task: JudgedTask,
    read index: PackedIndex,
    read candidate_hits_by_query: List[List[SearchHit]],
    candidate_k: Int,
) raises:
    print("== rerank exact late interaction ==")
    print("candidate_k: ", candidate_k)
    var query_index = 0
    var verifier = exact_late_interaction_verifier(candidate_k)

    def rerank_once() capturing raises:
        bench_compiler.keep(
            rerank_hits_with_verifier(
                task.queries[query_index].query,
                index,
                candidate_hits_by_query[query_index],
                task.k,
                verifier,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[rerank_once]().print()
    print("")


def benchmark_search_exact_with_exact_verifier(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    candidate_k: Int,
) raises:
    print("== search_exact_with_verifier(exact late interaction) ==")
    print("candidate_k: ", candidate_k)
    var query_index = 0
    var verifier = exact_late_interaction_verifier(candidate_k)

    def search_once() capturing raises:
        bench_compiler.keep(
            search_exact_with_verifier(
                backend,
                task.queries[query_index].query,
                index,
                task.k,
                verifier,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[search_once]().print()
    print("")


def benchmark_candidate_window(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    candidate_k: Int,
) raises:
    print("## verifier candidate window = ", candidate_k, " ##")
    var candidate_hits_by_query = precompute_candidate_hits(
        backend, task, index, candidate_k
    )
    benchmark_rerank_noop(task, index, candidate_hits_by_query)
    benchmark_rerank_exact(task, index, candidate_hits_by_query, candidate_k)
    benchmark_search_exact_with_exact_verifier(
        backend, task, index, candidate_k
    )


def benchmark_task(
    dataset_name: String, read task: JudgedTask, read index: PackedIndex
) raises:
    var backend = default_backend()
    benchmark_search_exact(backend, task, index)
    benchmark_candidate_window(backend, task, index, task.k)
    benchmark_candidate_window(backend, task, index, task.k * 4)


def benchmark_scifact_real_subset() raises:
    print("loading real BEIR/SciFact subset with storage...")
    var cache = ensure_scifact_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    print_fixture_header(
        "SciFact",
        task,
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
    )
    benchmark_task("SciFact", task, index)


def benchmark_fiqa_real_subset() raises:
    print("loading real BEIR/FIQA subset with storage...")
    var cache = ensure_fiqa_real_subset_cache()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    print_fixture_header(
        "FIQA",
        task,
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
    )
    benchmark_task("FIQA", task, index)


def main() raises:
    benchmark_scifact_real_subset()
    benchmark_fiqa_real_subset()
