from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak.benchmarks.latent_proxy_linear_profile_support import (
    flatten_linear_rows,
    linear_block_output_flat_rows,
    linear_block_output_flat_rows_tiled4,
)
from kayak.benchmarks.latent_proxy_profile_fixtures import (
    make_projection,
    make_projection_block,
    make_query,
    make_vector,
    require_latent_proxy_primitive_profile,
)
from kayak.index.latent_proxy import (
    linear_block_output_reference,
    projected_query_token_for_block,
)
from kayak.numeric import VectorScalar


def assert_vector_lists_close(
    read expected: List[VectorScalar], read observed: List[VectorScalar]
) raises:
    assert_equal(len(expected), len(observed))
    for index in range(len(expected)):
        assert_equal(
            abs(Float64(expected[index]) - Float64(observed[index])) < 0.00001,
            True,
        )


def test_flat_row_kernels_match_reference_for_tail_dims_and_remainder_rows() raises:
    var block = make_projection_block(7, 6, 11)
    var input_vector = make_vector(3000, 0, 7)
    var flat_rows = flatten_linear_rows(block.linear_rows, block.input_dim)

    var expected = linear_block_output_reference(input_vector, block)
    assert_vector_lists_close(
        expected, linear_block_output_flat_rows(input_vector, block, flat_rows)
    )
    assert_vector_lists_close(
        expected,
        linear_block_output_flat_rows_tiled4(input_vector, block, flat_rows),
    )


def test_tiled4_flat_rows_match_reference_for_hotspot_second_block() raises:
    var profile = require_latent_proxy_primitive_profile("q32_lat2048_b2_docs128")
    var projection = make_projection(profile)
    var query = make_query(profile.query_vector_count, profile.vector_dim)
    var block0 = projection.blocks[0].copy()
    var block1 = projection.blocks[1].copy()
    var input_vector = projected_query_token_for_block(query.token_vectors[0], block0)
    var flat_rows = flatten_linear_rows(block1.linear_rows, block1.input_dim)

    var expected = linear_block_output_reference(input_vector, block1)
    assert_vector_lists_close(
        expected,
        linear_block_output_flat_rows_tiled4(input_vector, block1, flat_rows),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
