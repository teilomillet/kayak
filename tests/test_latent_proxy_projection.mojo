from std.math import erf, sqrt
from std.testing import TestSuite, assert_equal

from kayak import EncodedQuery
from kayak.index import (
    LATENT_PROXY_ACTIVATION_GELU,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
    LatentQueryProjection,
    LatentQueryProjectionBlock,
    build_query_latent_proxy_vector,
)
from kayak.index.latent_proxy import (
    MutableLatentQueryProjectionScratch,
    build_query_latent_proxy_vector_multi_block,
    build_query_latent_proxy_vector_multi_block_reference,
    build_query_latent_proxy_vector_multi_block_with_scratch,
)


def test_latent_proxy_gelu_matches_exact_erf_formulation() raises:
    var projection = LatentQueryProjection(
        1,
        2,
        1.0,
        [
            LatentQueryProjectionBlock(
                LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                LATENT_PROXY_ACTIVATION_GELU,
                1,
                2,
                [[1.0], [0.0]],
                [0.0, 0.0],
                1.0,
                False,
                0.00001,
                [],
                [],
            )
        ],
    )
    var query = EncodedQuery([[1.0]])
    var projected = build_query_latent_proxy_vector(query, projection)
    var gelu = Float64(0.5) * Float64(1.0) * (
        Float64(1.0) + erf(Float64(1.0) / sqrt(Float64(2.0)))
    )
    var mean = gelu / Float64(2.0)
    var variance = (gelu * gelu) / Float64(4.0)
    var denom = sqrt(variance + Float64(0.00001))
    var expected = (gelu - mean) / denom
    assert_equal(abs(Float64(projected[0]) - expected) < 0.0000001, True)


def test_single_block_fast_path_matches_generic_multi_block_path() raises:
    var projection = LatentQueryProjection(
        2,
        3,
        4.0,
        [
            LatentQueryProjectionBlock(
                LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                LATENT_PROXY_ACTIVATION_GELU,
                2,
                3,
                [[1.0, 0.5], [0.0, 1.0], [0.25, -0.25]],
                [0.0, 0.25, -0.5],
                1.0,
                True,
                0.00001,
                [1.0, 0.75, 1.25],
                [0.0, 0.125, -0.25],
            )
        ],
    )
    var query = EncodedQuery(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.25, -0.75],
        ]
    )
    var optimized = build_query_latent_proxy_vector(query, projection)
    var generic = build_query_latent_proxy_vector_multi_block(query, projection)

    assert_equal(len(optimized), len(generic))
    for index in range(len(optimized)):
        assert_equal(
            abs(Float64(optimized[index]) - Float64(generic[index])) < 0.0000001,
            True,
        )


def test_multi_block_scratch_path_matches_reference_multi_block_path() raises:
    var projection = LatentQueryProjection(
        2,
        2,
        3.0,
        [
            LatentQueryProjectionBlock(
                LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                LATENT_PROXY_ACTIVATION_GELU,
                2,
                3,
                [[1.0, 0.5], [0.0, 1.0], [0.25, -0.25]],
                [0.0, 0.25, -0.5],
                1.0,
                True,
                0.00001,
                [1.0, 0.75, 1.25],
                [0.0, 0.125, -0.25],
            ),
            LatentQueryProjectionBlock(
                LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
                LATENT_PROXY_ACTIVATION_GELU,
                3,
                2,
                [[0.5, -0.25, 1.0], [1.0, 0.25, -0.5]],
                [0.1, -0.2],
                1.0,
                True,
                0.00001,
                [1.0, 0.9],
                [0.0, 0.05],
            ),
        ],
    )
    var query = EncodedQuery(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.25, -0.75],
        ]
    )
    var scratch = MutableLatentQueryProjectionScratch()
    var optimized = build_query_latent_proxy_vector_multi_block_with_scratch(
        query,
        projection,
        scratch,
    )
    var reference = build_query_latent_proxy_vector_multi_block_reference(
        query, projection
    )
    var selected = build_query_latent_proxy_vector_multi_block(query, projection)

    assert_equal(len(optimized), len(reference))
    assert_equal(len(selected), len(reference))
    for index in range(len(reference)):
        assert_equal(
            abs(Float64(optimized[index]) - Float64(reference[index])) < 0.0000001,
            True,
        )
        assert_equal(
            abs(Float64(selected[index]) - Float64(reference[index])) < 0.0000001,
            True,
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
