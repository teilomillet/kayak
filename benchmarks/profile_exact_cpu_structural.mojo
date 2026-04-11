import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from kayak.benchmarks import default_per_document_score_profiles, make_per_document_score_fixture
from kayak.numeric import ScoreScalar, VectorScalar, min_score_scalar
from kayak.scoring.dot128 import dot_product_dim128
from kayak.scoring.maxsim import exact_score_for_document_dim128
from benchmarks.structural.flat_dim128 import (
    best_similarity_for_query_token_dim128_flat,
    exact_score_for_flat_document_dim128,
    flatten_document_tokens,
)


def best_similarity_for_query_token_dim128_nested(
    read query_token: List[VectorScalar],
    read document_tokens: List[List[VectorScalar]],
) -> ScoreScalar:
    var best_similarity = min_score_scalar()

    for document_token in document_tokens:
        var similarity = dot_product_dim128(query_token, document_token)
        if similarity > best_similarity:
            best_similarity = similarity

    return best_similarity


def benchmark_single_query_token_sweep_structural() raises:
    print("== best_similarity_for_query_token_dim128: nested vs flat ==")

    for profile in default_per_document_score_profiles():
        var fixture = make_per_document_score_fixture(profile)
        var flat_document_tokens = flatten_document_tokens(fixture.index.token_vectors)

        print(
            "shape: query_vecs=",
            profile.query_vector_count,
            " doc_vecs=",
            profile.document_vector_count,
            " dim=",
            profile.vector_dim,
            " layout=nested",
        )

        def nested_once() capturing:
            bench_compiler.keep(
                best_similarity_for_query_token_dim128_nested(
                    fixture.query.token_vectors[0], fixture.index.token_vectors
                )
            )

        benchmark.run[nested_once]().print()
        print("")

        print(
            "shape: query_vecs=",
            profile.query_vector_count,
            " doc_vecs=",
            profile.document_vector_count,
            " dim=",
            profile.vector_dim,
            " layout=flat",
        )

        def flat_once() capturing:
            bench_compiler.keep(
                best_similarity_for_query_token_dim128_flat(
                    fixture.query.token_vectors[0],
                    flat_document_tokens,
                    profile.document_vector_count,
                )
            )

        benchmark.run[flat_once]().print()
        print("")


def benchmark_per_document_structural() raises:
    print("== exact_score_for_document_dim128: nested vs flat ==")

    for profile in default_per_document_score_profiles():
        var fixture = make_per_document_score_fixture(profile)
        var flat_document_tokens = flatten_document_tokens(fixture.index.token_vectors)

        print(
            "shape: query_vecs=",
            profile.query_vector_count,
            " doc_vecs=",
            profile.document_vector_count,
            " dim=",
            profile.vector_dim,
            " layout=nested",
        )

        def nested_once() capturing:
            bench_compiler.keep(
                exact_score_for_document_dim128(fixture.query, fixture.index, 0)
            )

        benchmark.run[nested_once]().print()
        print("")

        print(
            "shape: query_vecs=",
            profile.query_vector_count,
            " doc_vecs=",
            profile.document_vector_count,
            " dim=",
            profile.vector_dim,
            " layout=flat",
        )

        def flat_once() capturing:
            bench_compiler.keep(
                exact_score_for_flat_document_dim128(
                    fixture.query,
                    flat_document_tokens,
                    profile.document_vector_count,
                )
            )

        benchmark.run[flat_once]().print()
        print("")


def main() raises:
    benchmark_single_query_token_sweep_structural()
    benchmark_per_document_structural()
