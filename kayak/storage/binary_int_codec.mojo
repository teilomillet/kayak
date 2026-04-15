from std.collections import List
from std.memory import Span
from std.pathlib import Path

from .binary_vector_codec import append_uint64_le, read_uint64_le

comptime NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE = "binary_u64_le"


def require_supported_non_negative_int_payload_encoding(encoding: String) raises:
    if encoding == NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE:
        return

    raise Error("unsupported non-negative int payload encoding: " + encoding)


def encode_non_negative_int_payload_with_encoding(
    read values: List[Int], encoding: String
) raises -> List[Byte]:
    require_supported_non_negative_int_payload_encoding(encoding)

    var bytes = List[Byte]()
    for value in values:
        if value < 0:
            raise Error("binary non-negative int payload does not support negatives")

        append_uint64_le(bytes, UInt64(value))

    return bytes^


def decode_non_negative_int_payload_with_encoding(
    read bytes: List[Byte], encoding: String
) raises -> List[Int]:
    require_supported_non_negative_int_payload_encoding(encoding)

    if len(bytes) % 8 != 0:
        raise Error("binary non-negative int payload byte length must be 8-byte aligned")

    var values = List[Int]()
    var cursor = 0
    while cursor < len(bytes):
        values.append(Int(read_uint64_le(bytes, cursor)))
        cursor += 8

    return values^


def write_non_negative_int_payload_with_encoding(
    path: Path, read values: List[Int], encoding: String
) raises:
    var bytes = encode_non_negative_int_payload_with_encoding(values, encoding)
    path.write_bytes(Span(bytes))


def read_non_negative_int_payload_with_encoding(
    path: Path, encoding: String
) raises -> List[Int]:
    return decode_non_negative_int_payload_with_encoding(path.read_bytes(), encoding)
