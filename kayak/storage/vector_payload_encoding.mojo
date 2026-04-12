comptime VECTOR_PAYLOAD_ENCODING_BINARY_LE = "binary_le"
comptime VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE = "binary_f16_le"


def require_supported_packed_index_vector_payload_encoding(
    vector_payload_encoding: String
) raises:
    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_LE:
        return

    if vector_payload_encoding == VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE:
        return

    raise Error(
        "unsupported packed index vector payload encoding: "
        + vector_payload_encoding
    )
