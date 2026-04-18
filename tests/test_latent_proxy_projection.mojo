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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
