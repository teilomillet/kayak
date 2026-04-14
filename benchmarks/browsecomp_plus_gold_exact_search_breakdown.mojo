import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.os import makedirs
from std.pathlib import Path

from kayak import ExactCpuBackend, ScoreScalar
from kayak.search import search_exact
from kayak.search.topk import top_k_hits
from kayak.storage import ensure_browsecomp_plus_gold_real_subset_cache


def benchmark_score_all() raises -> Float64:
    print("== score_all ==")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
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


def benchmark_top_k() raises -> Float64:
    print("== top_k_hits(precomputed_scores) ==")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
    var all_scores = List[List[ScoreScalar]]()
    for judged_query in task.queries:
        all_scores.append(backend.score_all(judged_query.query, index))

    var query_index = 0

    def score_once() capturing raises:
        bench_compiler.keep(
            top_k_hits(index.doc_ids, all_scores[query_index], task.k)
        )
        query_index += 1
        if query_index == len(task.queries):
            query_index = 0

    var report = benchmark.run[score_once]()
    report.print()
    print("")
    return Float64(report.mean())


def benchmark_search_exact() raises -> Float64:
    print("== search_exact ==")
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var backend = ExactCpuBackend()
    var task = cache.stored_task.task.copy()
    var index = cache.stored_index.index.copy()
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


def main() raises:
    var cache = ensure_browsecomp_plus_gold_real_subset_cache()
    var task = cache.stored_task.task.copy()
    print("dataset= BrowseComp-Plus Gold")
    print("slice= ", task.slice_name)
    print("docs= ", len(task.documents))
    print("query_vectors= ", task.nominal_query_vector_count)
    print("doc_vectors= ", task.nominal_document_vector_count)
    print("vector_dim= ", task.vector_dim)
    print("")

    var score_all_mean = benchmark_score_all()
    var top_k_mean = benchmark_top_k()
    var search_exact_mean = benchmark_search_exact()

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    var output_path = output_root / "browsecomp_plus_gold_exact_search_breakdown.tsv"
    var lines = String()
    lines += "benchmark_kind\tmean_seconds\n"
    lines += "score_all\t" + String(score_all_mean) + "\n"
    lines += "top_k_hits(precomputed_scores)\t" + String(top_k_mean) + "\n"
    lines += "search_exact\t" + String(search_exact_mean) + "\n"
    output_path.write_text(lines)
    print("wrote ", String(output_path))
