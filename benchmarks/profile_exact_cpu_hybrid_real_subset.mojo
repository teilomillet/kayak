import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler
from std.pathlib import Path

from kayak import (
    ExactCpuBackend,
    FlatQueryDim128,
    HybridFlatDim128Index,
    JudgedTask,
    PackedIndex,
    build_flat_query_dim128,
    build_hybrid_flat_dim128_index,
    ensure_stored_hybrid_flat_dim128_index,
    exact_scores_for_hybrid_flat_index_dim128,
    exact_scores_for_hybrid_flat_index_dim128_with_flat_query,
    load_stored_hybrid_flat_dim128_index,
    search_exact_hybrid_flat_dim128,
    search_exact_hybrid_flat_dim128_with_flat_query,
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


def print_fixture_header(
    dataset_name: String,
    task: JudgedTask,
    task_source: String,
    index_source: String,
    hybrid_index_source: String,
):
    print("dataset: ", dataset_name)
    print("slice: ", task.slice_name)
    print("task_source: ", task_source)
    print("index_source: ", index_source)
    print("hybrid_index_source: ", hybrid_index_source)
    print("queries: ", len(task.queries))
    print("documents: ", len(task.documents))
    print("query_vectors≈ ", task.nominal_query_vector_count)
    print("doc_vectors≈ ", task.nominal_document_vector_count)
    print("vector_dim: ", task.vector_dim)
    print("")


def benchmark_build_hybrid_index(read index: PackedIndex) raises:
    print("== build_hybrid_flat_dim128_index ==")

    def build_once() capturing raises:
        bench_compiler.keep(build_hybrid_flat_dim128_index(index))

    benchmark.run[build_once]().print()
    print("")


def benchmark_load_hybrid_index(root: Path) raises:
    print("== load_stored_hybrid_flat_dim128_index ==")

    def load_once() capturing raises:
        bench_compiler.keep(load_stored_hybrid_flat_dim128_index(root))

    benchmark.run[load_once]().print()
    print("")


def benchmark_nested_score_all(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
) raises:
    print("== nested score_all ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(backend.score_all(task.queries[query_index].query, index))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_build_flat_query(
    read task: JudgedTask
) raises:
    print("== build_flat_query_dim128 ==")
    var query_index = 0

    def build_once() capturing raises:
        bench_compiler.keep(build_flat_query_dim128(task.queries[query_index].query))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[build_once]().print()
    print("")


def benchmark_hybrid_score_all(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
) raises:
    print("== hybrid score_all ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_index_dim128(
                task.queries[query_index].query,
                index,
                hybrid_index,
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_hybrid_score_all_with_flat_query(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
) raises:
    print("== hybrid score_all with flat query ==")
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            exact_scores_for_hybrid_flat_index_dim128_with_flat_query(
                flat_query,
                index,
                hybrid_index,
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_nested_search_exact(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
) raises:
    print("== nested search_exact ==")
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


def benchmark_hybrid_search_exact(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
) raises:
    print("== hybrid search_exact ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            search_exact_hybrid_flat_dim128(
                task.queries[query_index].query,
                index,
                hybrid_index,
                task.k,
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_hybrid_search_exact_with_flat_query(
    read backend: ExactCpuBackend,
    read task: JudgedTask,
    read index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
) raises:
    print("== hybrid search_exact with flat query ==")
    var query_index = 0

    def score_once() capturing raises:
        var flat_query = build_flat_query_dim128(task.queries[query_index].query)
        bench_compiler.keep(
            search_exact_hybrid_flat_dim128_with_flat_query(
                flat_query,
                index,
                hybrid_index,
                task.k,
                backend.scoring_config,
            )
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    benchmark.run[score_once]().print()
    print("")


def benchmark_task(
    hybrid_root: Path,
    read task: JudgedTask,
    read index: PackedIndex,
    read hybrid_index: HybridFlatDim128Index,
) raises:
    var backend = default_backend()
    benchmark_build_hybrid_index(index)
    benchmark_load_hybrid_index(hybrid_root)
    benchmark_build_flat_query(task)
    benchmark_nested_score_all(backend, task, index)
    benchmark_hybrid_score_all(backend, task, index, hybrid_index)
    benchmark_hybrid_score_all_with_flat_query(backend, task, index, hybrid_index)
    benchmark_nested_search_exact(backend, task, index)
    benchmark_hybrid_search_exact(backend, task, index, hybrid_index)
    benchmark_hybrid_search_exact_with_flat_query(
        backend, task, index, hybrid_index
    )


def benchmark_scifact_real_subset() raises:
    print("loading real BEIR/SciFact subset with storage...")
    var cache = ensure_scifact_real_subset_cache()
    var hybrid_cache = ensure_stored_hybrid_flat_dim128_index(
        Path(".cache/kayak/scifact_real_subset/hybrid_flat_dim128_index"),
        cache.stored_index,
    )
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    print_fixture_header(
        "SciFact",
        task,
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
        source_label(
            hybrid_cache.loaded_from_storage, "build_hybrid_flat_dim128_index"
        ),
    )
    benchmark_task(
        Path(".cache/kayak/scifact_real_subset/hybrid_flat_dim128_index"),
        task,
        index,
        hybrid_cache.stored_index.index,
    )


def benchmark_fiqa_real_subset() raises:
    print("loading real BEIR/FIQA subset with storage...")
    var cache = ensure_fiqa_real_subset_cache()
    var hybrid_cache = ensure_stored_hybrid_flat_dim128_index(
        Path(".cache/kayak/fiqa_real_subset/hybrid_flat_dim128_index"),
        cache.stored_index,
    )
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    print_fixture_header(
        "FIQA",
        task,
        source_label(cache.loaded_task_from_storage, "colbert_cpu_encode"),
        source_label(cache.loaded_index_from_storage, "pack_documents"),
        source_label(
            hybrid_cache.loaded_from_storage, "build_hybrid_flat_dim128_index"
        ),
    )
    benchmark_task(
        Path(".cache/kayak/fiqa_real_subset/hybrid_flat_dim128_index"),
        task,
        index,
        hybrid_cache.stored_index.index,
    )


def main() raises:
    benchmark_scifact_real_subset()
    benchmark_fiqa_real_subset()
