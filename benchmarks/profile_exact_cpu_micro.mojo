import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List
from std.runtime.asyncrt import parallelism_level

from kayak.benchmarks import (
    DotProductProfile,
    default_exact_search_profiles,
    default_per_document_score_profiles,
    make_dot_product_fixture,
    make_exact_search_fixture_for_profiled_search,
    make_per_document_score_fixture,
)
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM, dot_product_dim128
from kayak.scoring.maxsim import (
    build_vector_balanced_boundaries,
    exact_score_for_document_dim128,
)


def append_capped_candidate(
    mut candidates: List[Int], raw_candidate: Int, maximum: Int
):
    if raw_candidate <= 0:
        return

    var candidate = raw_candidate
    if candidate > maximum:
        candidate = maximum

    for existing in candidates:
        if existing == candidate:
            return

    candidates.append(candidate)


def boundary_work_item_candidates(document_count: Int) -> List[Int]:
    var candidates = List[Int]()
    var worker_count = parallelism_level()

    append_capped_candidate(candidates, 1, document_count)
    append_capped_candidate(candidates, worker_count, document_count)
    append_capped_candidate(candidates, worker_count * 4, document_count)

    return candidates^


def best_similarity_for_query_token_dim128(
    read query_token: List[VectorScalar],
    read document_tokens: List[List[VectorScalar]],
) -> ScoreScalar:
    var best_similarity = min_score_scalar()

    for document_token in document_tokens:
        var similarity = dot_product_dim128(query_token, document_token)
        if similarity > best_similarity:
            best_similarity = similarity

    return best_similarity


def benchmark_dot_product_dim128() raises:
    print("== dot_product_dim128 ==")
    var fixture = make_dot_product_fixture(DotProductProfile(COLBERT_VECTOR_DIM))

    print("shape: dim=", COLBERT_VECTOR_DIM)

    def score_once() capturing:
        bench_compiler.keep(dot_product_dim128(fixture.lhs, fixture.rhs))

    benchmark.run[score_once]().print()
    print("")


def benchmark_single_query_token_sweep_dim128() raises:
    print("== best_similarity_for_query_token_dim128 ==")

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
                best_similarity_for_query_token_dim128(
                    fixture.query.token_vectors[0], fixture.index.token_vectors
                )
            )

        benchmark.run[score_once]().print()
        print("")


def benchmark_exact_score_for_document_dim128() raises:
    print("== exact_score_for_document_dim128 ==")

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
                exact_score_for_document_dim128(fixture.query, fixture.index, 0)
            )

        benchmark.run[score_once]().print()
        print("")


def benchmark_boundary_construction() raises:
    print("== build_vector_balanced_boundaries ==")

    for profile in default_exact_search_profiles():
        var fixture = make_exact_search_fixture_for_profiled_search(profile)

        for work_items in boundary_work_item_candidates(profile.document_count):
            print(
                "shape: docs=",
                profile.document_count,
                " doc_vecs=",
                profile.document_vector_count,
                " query_vecs=",
                profile.query_vector_count,
                " dim=",
                profile.vector_dim,
                " work_items=",
                work_items,
            )

            def score_once() capturing:
                bench_compiler.keep(
                    build_vector_balanced_boundaries(fixture.index, work_items)
                )

            benchmark.run[score_once]().print()
            print("")


def main() raises:
    print("parallelism_level: ", parallelism_level())
    print("")
    benchmark_dot_product_dim128()
    benchmark_single_query_token_sweep_dim128()
    benchmark_exact_score_for_document_dim128()
    benchmark_boundary_construction()
