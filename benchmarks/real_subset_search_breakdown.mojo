import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List

from kayak import ExactCpuBackend, JudgedTask, PackedIndex, ScoreScalar
from kayak.search import search_exact
from kayak.search.topk import top_k_hits
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


def precompute_scores(
    read backend: ExactCpuBackend, read task: JudgedTask, read index: PackedIndex
) raises -> List[List[ScoreScalar]]:
    var all_scores = List[List[ScoreScalar]]()
    for judged_query in task.queries:
        all_scores.append(backend.score_all(judged_query.query, index))
    return all_scores^


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
    print("")


def benchmark_query_copy(task: JudgedTask) raises:
    print("== query.copy() ==")
    var query_index = 0

    def copy_once() capturing:
        bench_compiler.keep(task.queries[query_index].query.copy())
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[copy_once]().print()
    print("")


def benchmark_index_copy(index: PackedIndex) raises:
    print("== index.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.copy())

    benchmark.run[copy_once]().print()
    print("")


def benchmark_doc_id_copy(index: PackedIndex) raises:
    print("== doc_ids.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.doc_ids.copy())

    benchmark.run[copy_once]().print()
    print("")


def benchmark_doc_offset_copy(index: PackedIndex) raises:
    print("== doc_offsets.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.doc_offsets.copy())

    benchmark.run[copy_once]().print()
    print("")


def benchmark_token_vector_copy(index: PackedIndex) raises:
    print("== token_vectors.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.token_vectors.copy())

    benchmark.run[copy_once]().print()
    print("")


def benchmark_score_copy(
    read task: JudgedTask, read precomputed_scores: List[List[ScoreScalar]]
) raises:
    print("== scores.copy() ==")
    var query_index = 0

    def copy_once() capturing:
        bench_compiler.keep(precomputed_scores[query_index].copy())
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[copy_once]().print()
    print("")


def benchmark_score_all(
    read backend: ExactCpuBackend, read task: JudgedTask, read index: PackedIndex
) raises:
    print("== score_all ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(backend.score_all(task.queries[query_index].query, index))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_top_k(
    read task: JudgedTask,
    read index: PackedIndex,
    read precomputed_scores: List[List[ScoreScalar]],
) raises:
    print("== top_k_hits(precomputed_scores) ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            top_k_hits(index.doc_ids, precomputed_scores[query_index], task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_search_exact(
    read backend: ExactCpuBackend, read task: JudgedTask, read index: PackedIndex
) raises:
    print("== search_exact ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            search_exact(backend, task.queries[query_index].query, index, task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_task(
    dataset_name: String, read task: JudgedTask, read index: PackedIndex
) raises:
    var backend = default_backend()
    var precomputed_scores = precompute_scores(backend, task, index)

    benchmark_query_copy(task.copy())
    benchmark_index_copy(index.copy())
    benchmark_doc_id_copy(index.copy())
    benchmark_doc_offset_copy(index.copy())
    benchmark_token_vector_copy(index.copy())
    benchmark_score_copy(task, precomputed_scores)
    benchmark_score_all(backend, task, index)
    benchmark_top_k(task, index, precomputed_scores)
    benchmark_search_exact(backend, task, index)


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
