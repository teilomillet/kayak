import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from kayak import ExactCpuBackend, ExactScoringConfig
from kayak.benchmarks import (
    default_exact_search_profiles,
    make_exact_search_fixture_for_profiled_search,
)
from kayak.search import search_exact


def default_backend() -> ExactCpuBackend:
    return ExactCpuBackend()


def backend_without_dim128_fast_path() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_dim128_fast_path = False
    return ExactCpuBackend(config^)


def backend_without_parallel_scoring() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_scoring = False
    return ExactCpuBackend(config^)


def backend_without_parallel_oversubscription() -> ExactCpuBackend:
    var config = ExactScoringConfig()
    config.enable_parallel_work_item_oversubscription = False
    return ExactCpuBackend(config^)


def benchmark_backend_score_all(name: String, backend: ExactCpuBackend) raises:
    print("== ExactCpuBackend.score_all:", name, "==")

    for profile in default_exact_search_profiles():
        var fixture = make_exact_search_fixture_for_profiled_search(profile)

        print(
            "shape: docs=",
            profile.document_count,
            " doc_vecs=",
            profile.document_vector_count,
            " query_vecs=",
            profile.query_vector_count,
            " dim=",
            profile.vector_dim,
        )

        def score_once() capturing raises:
            bench_compiler.keep(backend.score_all(fixture.query, fixture.index))

        var report = benchmark.run[score_once]()
        report.print()
        print("")


def benchmark_backend_search_exact(name: String, backend: ExactCpuBackend) raises:
    print("== search_exact:", name, "==")

    for profile in default_exact_search_profiles():
        var fixture = make_exact_search_fixture_for_profiled_search(profile)

        print(
            "shape: docs=",
            profile.document_count,
            " doc_vecs=",
            profile.document_vector_count,
            " query_vecs=",
            profile.query_vector_count,
            " dim=",
            profile.vector_dim,
            " top_k=",
            profile.top_k,
        )

        def score_once() capturing raises:
            bench_compiler.keep(
                search_exact(backend, fixture.query, fixture.index, fixture.top_k)
            )

        var report = benchmark.run[score_once]()
        report.print()
        print("")


def main() raises:
    benchmark_backend_score_all("default", default_backend())
    benchmark_backend_score_all(
        "dim128_fast_path_disabled", backend_without_dim128_fast_path()
    )
    benchmark_backend_score_all(
        "parallel_scoring_disabled", backend_without_parallel_scoring()
    )
    benchmark_backend_score_all(
        "parallel_oversubscription_disabled",
        backend_without_parallel_oversubscription(),
    )
    benchmark_backend_search_exact("default", default_backend())
    benchmark_backend_search_exact(
        "dim128_fast_path_disabled", backend_without_dim128_fast_path()
    )
    benchmark_backend_search_exact(
        "parallel_scoring_disabled", backend_without_parallel_scoring()
    )
    benchmark_backend_search_exact(
        "parallel_oversubscription_disabled",
        backend_without_parallel_oversubscription(),
    )
