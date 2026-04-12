from std.collections import List
from std.ffi import UnsafeUnion
from std.memory import Span
from std.pathlib import Path

from kayak.numeric import VECTOR_SCALAR_NAME, VectorScalar

from .vector_payload_encoding import (
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
)


def native_vector_scalar_byte_width() raises -> Int:
    if VECTOR_SCALAR_NAME == "Float32":
        return 4

    if VECTOR_SCALAR_NAME == "Float64":
        return 8

    raise Error("unsupported vector scalar type for binary storage")


def vector_payload_scalar_byte_width(
    vector_payload_encoding: String
) raises -> Int:
    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_LE:
        return native_vector_scalar_byte_width()

    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE:
        return 2

    raise Error(
        "unsupported binary vector payload encoding: " + vector_payload_encoding
    )


def append_uint16_le(mut bytes: List[Byte], value: UInt16):
    bytes.append(Byte(value & 0xFF))
    bytes.append(Byte((value >> 8) & 0xFF))


def append_uint32_le(mut bytes: List[Byte], value: UInt32):
    bytes.append(Byte(value & 0xFF))
    bytes.append(Byte((value >> 8) & 0xFF))
    bytes.append(Byte((value >> 16) & 0xFF))
    bytes.append(Byte((value >> 24) & 0xFF))


def append_uint64_le(mut bytes: List[Byte], value: UInt64):
    bytes.append(Byte(value & 0xFF))
    bytes.append(Byte((value >> 8) & 0xFF))
    bytes.append(Byte((value >> 16) & 0xFF))
    bytes.append(Byte((value >> 24) & 0xFF))
    bytes.append(Byte((value >> 32) & 0xFF))
    bytes.append(Byte((value >> 40) & 0xFF))
    bytes.append(Byte((value >> 48) & 0xFF))
    bytes.append(Byte((value >> 56) & 0xFF))


def encode_vector_scalar_le(
    mut bytes: List[Byte], value: VectorScalar
) raises:
    if VECTOR_SCALAR_NAME == "Float32":
        var bits_union = UnsafeUnion[Float32, UInt32](Float32(value))
        append_uint32_le(bytes, bits_union.unsafe_get[UInt32]())
        return

    if VECTOR_SCALAR_NAME == "Float64":
        var bits_union = UnsafeUnion[Float64, UInt64](Float64(value))
        append_uint64_le(bytes, bits_union.unsafe_get[UInt64]())
        return

    raise Error("unsupported vector scalar type for binary storage")


def encode_vector_scalar_with_encoding_le(
    mut bytes: List[Byte],
    value: VectorScalar,
    vector_payload_encoding: String,
) raises:
    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_LE:
        encode_vector_scalar_le(bytes, value)
        return

    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE:
        var bits_union = UnsafeUnion[Float16, UInt16](Float16(value))
        append_uint16_le(bytes, bits_union.unsafe_get[UInt16]())
        return

    raise Error(
        "unsupported binary vector payload encoding: " + vector_payload_encoding
    )


def encode_binary_vector_payload(
    read vectors: List[List[VectorScalar]]
) raises -> List[Byte]:
    return encode_binary_vector_payload_with_encoding(
        vectors, VECTOR_PAYLOAD_ENCODING_BINARY_LE
    )


def encode_binary_vector_payload_with_encoding(
    read vectors: List[List[VectorScalar]],
    vector_payload_encoding: String,
) raises -> List[Byte]:
    var bytes = List[Byte]()

    for vector in vectors:
        for value in vector:
            encode_vector_scalar_with_encoding_le(
                bytes, value, vector_payload_encoding
            )

    return bytes^


def encode_binary_scalar_payload(
    read values: List[VectorScalar]
) raises -> List[Byte]:
    var bytes = List[Byte]()

    for value in values:
        encode_vector_scalar_le(bytes, value)

    return bytes^


def read_uint32_le(read bytes: List[Byte], offset: Int) raises -> UInt32:
    if offset + 4 > len(bytes):
        raise Error("binary vector payload ended early")

    return (
        UInt32(bytes[offset])
        | (UInt32(bytes[offset + 1]) << 8)
        | (UInt32(bytes[offset + 2]) << 16)
        | (UInt32(bytes[offset + 3]) << 24)
    )


def read_uint16_le(read bytes: List[Byte], offset: Int) raises -> UInt16:
    if offset + 2 > len(bytes):
        raise Error("binary vector payload ended early")

    return UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)


def read_uint64_le(read bytes: List[Byte], offset: Int) raises -> UInt64:
    if offset + 8 > len(bytes):
        raise Error("binary vector payload ended early")

    return (
        UInt64(bytes[offset])
        | (UInt64(bytes[offset + 1]) << 8)
        | (UInt64(bytes[offset + 2]) << 16)
        | (UInt64(bytes[offset + 3]) << 24)
        | (UInt64(bytes[offset + 4]) << 32)
        | (UInt64(bytes[offset + 5]) << 40)
        | (UInt64(bytes[offset + 6]) << 48)
        | (UInt64(bytes[offset + 7]) << 56)
    )


def decode_vector_scalar_le(
    read bytes: List[Byte], offset: Int
) raises -> VectorScalar:
    if VECTOR_SCALAR_NAME == "Float32":
        var bits_union = UnsafeUnion[Float32, UInt32](
            read_uint32_le(bytes, offset)
        )
        return VectorScalar(bits_union.unsafe_get[Float32]())

    if VECTOR_SCALAR_NAME == "Float64":
        var bits_union = UnsafeUnion[Float64, UInt64](
            read_uint64_le(bytes, offset)
        )
        return VectorScalar(bits_union.unsafe_get[Float64]())

    raise Error("unsupported vector scalar type for binary storage")


def decode_vector_scalar_with_encoding_le(
    read bytes: List[Byte],
    offset: Int,
    vector_payload_encoding: String,
) raises -> VectorScalar:
    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_LE:
        return decode_vector_scalar_le(bytes, offset)

    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE:
        var bits_union = UnsafeUnion[Float16, UInt16](
            read_uint16_le(bytes, offset)
        )
        return VectorScalar(bits_union.unsafe_get[Float16]())

    raise Error(
        "unsupported binary vector payload encoding: " + vector_payload_encoding
    )


def decode_binary_vector_payload(
    read bytes: List[Byte], vector_dim: Int
) raises -> List[List[VectorScalar]]:
    return decode_binary_vector_payload_with_encoding(
        bytes, vector_dim, VECTOR_PAYLOAD_ENCODING_BINARY_LE
    )


def decode_binary_vector_payload_with_encoding(
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
                    bytes, cursor, vector_payload_encoding
                )
            )
            cursor += scalar_width

        vectors.append(vector^)

    return vectors^


def decode_binary_scalar_payload(
    read bytes: List[Byte]
) raises -> List[VectorScalar]:
    var scalar_width = native_vector_scalar_byte_width()
    if len(bytes) % scalar_width != 0:
        raise Error("binary scalar payload byte length does not match scalar width")

    var values = List[VectorScalar]()
    var cursor = 0

    while cursor < len(bytes):
        values.append(decode_vector_scalar_le(bytes, cursor))
        cursor += scalar_width

    return values^


def write_binary_vector_payload(
    path: Path, read vectors: List[List[VectorScalar]]
) raises:
    write_binary_vector_payload_with_encoding(
        path, vectors, VECTOR_PAYLOAD_ENCODING_BINARY_LE
    )


def write_binary_vector_payload_with_encoding(
    path: Path,
    read vectors: List[List[VectorScalar]],
    vector_payload_encoding: String,
) raises:
    var bytes = encode_binary_vector_payload_with_encoding(
        vectors, vector_payload_encoding
    )
    path.write_bytes(Span(bytes))


def write_binary_scalar_payload(
    path: Path, read values: List[VectorScalar]
) raises:
    var bytes = encode_binary_scalar_payload(values)
    path.write_bytes(Span(bytes))


def read_binary_vector_payload(
    path: Path, vector_dim: Int
) raises -> List[List[VectorScalar]]:
    return read_binary_vector_payload_with_encoding(
        path, vector_dim, VECTOR_PAYLOAD_ENCODING_BINARY_LE
    )


def read_binary_vector_payload_with_encoding(
    path: Path,
    vector_dim: Int,
    vector_payload_encoding: String,
) raises -> List[List[VectorScalar]]:
    return decode_binary_vector_payload_with_encoding(
        path.read_bytes(), vector_dim, vector_payload_encoding
    )


def read_binary_scalar_payload(path: Path) raises -> List[VectorScalar]:
    return decode_binary_scalar_payload(path.read_bytes())
