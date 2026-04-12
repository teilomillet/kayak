import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak import ExactCpuBackend, JudgedTask, PackedIndex, ScoreScalar
from kayak.benchmarks import (
    SearchBreakdownBenchmarkSummary,
    build_search_breakdown_benchmark_summary,
    search_breakdown_benchmark_summaries_json,
)
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


def benchmark_query_copy(task: JudgedTask) raises -> Float64:
    print("== query.copy() ==")
    var query_index = 0

    def copy_once() capturing:
        bench_compiler.keep(task.queries[query_index].query.copy())
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_index_copy(index: PackedIndex) raises -> Float64:
    print("== index.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.copy())

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_doc_id_copy(index: PackedIndex) raises -> Float64:
    print("== doc_ids.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.doc_ids.copy())

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_doc_offset_copy(index: PackedIndex) raises -> Float64:
    print("== doc_offsets.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.doc_offsets.copy())

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_token_vector_copy(index: PackedIndex) raises -> Float64:
    print("== token_vectors.copy() ==")

    def copy_once() capturing:
        bench_compiler.keep(index.token_vectors.copy())

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_score_copy(
    read task: JudgedTask, read precomputed_scores: List[List[ScoreScalar]]
) raises -> Float64:
    print("== scores.copy() ==")
    var query_index = 0

    def copy_once() capturing:
        bench_compiler.keep(precomputed_scores[query_index].copy())
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[copy_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_score_all(
    read backend: ExactCpuBackend, read task: JudgedTask, read index: PackedIndex
) raises -> Float64:
    print("== score_all ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(backend.score_all(task.queries[query_index].query, index))
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_top_k(
    read task: JudgedTask,
    read index: PackedIndex,
    read precomputed_scores: List[List[ScoreScalar]],
) raises -> Float64:
    print("== top_k_hits(precomputed_scores) ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            top_k_hits(index.doc_ids, precomputed_scores[query_index], task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_search_exact(
    read backend: ExactCpuBackend, read task: JudgedTask, read index: PackedIndex
) raises -> Float64:
    print("== search_exact ==")
    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            search_exact(backend, task.queries[query_index].query, index, task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_task(
    dataset_name: String,
    read task: JudgedTask,
    read index: PackedIndex,
    task_source: String,
    index_source: String,
) raises -> List[SearchBreakdownBenchmarkSummary]:
    var backend = default_backend()
    var precomputed_scores = precompute_scores(backend, task, index)
    var summaries = List[SearchBreakdownBenchmarkSummary]()

    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "query.copy()",
            benchmark_query_copy(task.copy()),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "index.copy()",
            benchmark_index_copy(index.copy()),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "doc_ids.copy()",
            benchmark_doc_id_copy(index.copy()),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "doc_offsets.copy()",
            benchmark_doc_offset_copy(index.copy()),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "token_vectors.copy()",
            benchmark_token_vector_copy(index.copy()),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "scores.copy()",
            benchmark_score_copy(task, precomputed_scores),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "score_all",
            benchmark_score_all(backend, task, index),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "top_k_hits(precomputed_scores)",
            benchmark_top_k(task, index, precomputed_scores),
        )
    )
    summaries.append(
        build_search_breakdown_benchmark_summary(
            dataset_name,
            task,
            task_source,
            index_source,
            "search_exact",
            benchmark_search_exact(backend, task, index),
        )
    )
    return summaries^


def append_summaries(
    mut all_summaries: List[SearchBreakdownBenchmarkSummary],
    read dataset_summaries: List[SearchBreakdownBenchmarkSummary],
):
    for summary in dataset_summaries:
        all_summaries.append(summary.copy())


def benchmark_scifact_real_subset(
    mut summaries: List[SearchBreakdownBenchmarkSummary]
) raises:
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
    print_fixture_header(
        "SciFact",
        task,
        task_source,
        index_source,
    )
    append_summaries(
        summaries, benchmark_task("SciFact", task, index, task_source, index_source)
    )


def benchmark_fiqa_real_subset(
    mut summaries: List[SearchBreakdownBenchmarkSummary]
) raises:
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
    print_fixture_header(
        "FIQA",
        task,
        task_source,
        index_source,
    )
    append_summaries(
        summaries, benchmark_task("FIQA", task, index, task_source, index_source)
    )


def main() raises:
    var summaries = List[SearchBreakdownBenchmarkSummary]()
    benchmark_scifact_real_subset(summaries)
    benchmark_fiqa_real_subset(summaries)

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "public_search_breakdown.json"
    output_path.write_text(search_breakdown_benchmark_summaries_json(summaries))
    print("wrote ", String(output_path))
