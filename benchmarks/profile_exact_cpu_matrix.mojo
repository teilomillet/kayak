import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from kayak.benchmarks import (
    default_dot_product_profiles,
    default_exact_search_profiles,
    default_per_document_score_profiles,
    make_dot_product_fixture,
    make_exact_search_fixture_for_profiled_search,
    make_per_document_score_fixture,
)
from kayak.runtime import ExactCpuBackend
from kayak.scoring.dot import dot_product
from kayak.scoring.maxsim import exact_score_for_document
from kayak.search import search_exact


def benchmark_dot_product() raises:
    print("== dot_product ==")

    for profile in default_dot_product_profiles():
        var fixture = make_dot_product_fixture(profile)

        print("shape: dim=", profile.vector_dim)

        def score_once() capturing:
            bench_compiler.keep(dot_product(fixture.lhs, fixture.rhs))

        var report = benchmark.run[score_once]()
        report.print()
        print("")


def benchmark_per_document_maxsim() raises:
    print("== exact_score_for_document ==")

    for profile in default_per_document_score_profiles():
        var fixture = make_per_document_score_fixture(profile)

        print(
            "shape: query_vecs=",
            profile.query_vector_count,
            " doc_vecs=",
            profile.document_vector_count,
            " dim=",
            profile.vector_dim,
        )

        def score_once() capturing:
            bench_compiler.keep(
                exact_score_for_document(fixture.query, fixture.index, 0)
            )

        var report = benchmark.run[score_once]()
        report.print()
        print("")


def benchmark_score_all() raises:
    print("== ExactCpuBackend.score_all ==")

    var backend = ExactCpuBackend()
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


def benchmark_search_exact() raises:
    print("== search_exact ==")

    var backend = ExactCpuBackend()
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
    benchmark_dot_product()
    benchmark_per_document_maxsim()
    benchmark_score_all()
    benchmark_search_exact()
