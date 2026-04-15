from std.collections import List
from std.testing import TestSuite, assert_equal

from kayak import VectorScalar
from kayak.storage.binary_vector_codec import (
    decode_binary_vector_payload_with_encoding,
    decode_vector_scalar_with_encoding_le,
    encode_binary_vector_payload_with_encoding,
    vector_payload_scalar_byte_width,
)
from kayak.storage.vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
)


def reference_decode_binary_vector_payload_with_encoding(
    read bytes: List[Byte],
    vector_dim: Int,
    vector_payload_encoding: String,
) raises -> List[List[VectorScalar]]:
    if vector_dim <= 0:
        raise Error("binary vector payload requires a positive vector_dim")

    var scalar_width = vector_payload_scalar_byte_width(vector_payload_encoding)
    if len(bytes) % scalar_width != 0:
        raise Error("binary vector payload byte length does not match scalar width")

    var scalar_count = len(bytes) // scalar_width
    if scalar_count % vector_dim != 0:
        raise Error("binary vector payload scalar count does not match vector_dim")

    var vector_count = scalar_count // vector_dim
    var vectors = List[List[VectorScalar]]()
    var cursor = 0

    for _ in range(vector_count):
        var vector = List[VectorScalar]()
        for _ in range(vector_dim):
            vector.append(
                decode_vector_scalar_with_encoding_le(
                    bytes,
                    cursor,
                    vector_payload_encoding,
                )
            )
            cursor += scalar_width
        vectors.append(vector^)

    return vectors^


def assert_vector_payloads_equal(
    read lhs: List[List[VectorScalar]], read rhs: List[List[VectorScalar]]
) raises:
    assert_equal(len(lhs), len(rhs))
    for vector_index in range(len(lhs)):
        assert_equal(len(lhs[vector_index]), len(rhs[vector_index]))
        for dim_index in range(len(lhs[vector_index])):
            assert_equal(
                lhs[vector_index][dim_index],
                rhs[vector_index][dim_index],
            )


def codec_fixture_vectors() -> List[List[VectorScalar]]:
    return [
        [1.0, -2.5, 3.125, 4.0],
        [-0.5, 0.25, -7.75, 8.5],
        [9.0, -10.0, 11.5, -12.25],
    ]


def test_decode_binary_vector_payload_matches_reference_for_binary_le() raises:
    var vectors = codec_fixture_vectors()
    var bytes = encode_binary_vector_payload_with_encoding(
        vectors,
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )
    var decoded = decode_binary_vector_payload_with_encoding(
        bytes,
        4,
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )
    var reference = reference_decode_binary_vector_payload_with_encoding(
        bytes,
        4,
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )

    assert_vector_payloads_equal(decoded, reference)


def test_decode_binary_vector_payload_matches_reference_for_binary_f16_le() raises:
    var vectors = codec_fixture_vectors()
    var bytes = encode_binary_vector_payload_with_encoding(
        vectors,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    )
    var decoded = decode_binary_vector_payload_with_encoding(
        bytes,
        4,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    )
    var reference = reference_decode_binary_vector_payload_with_encoding(
        bytes,
        4,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    )

    assert_vector_payloads_equal(decoded, reference)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
