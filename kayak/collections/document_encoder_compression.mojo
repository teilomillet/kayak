# Records document-side encoder compression provenance that cannot be recovered
# from a packed vector payload alone. This is distinct from document-side
# transforms over already stored vectors.

from std.collections import List

from .validation import require_non_empty_string, require_positive_int


comptime DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET = (
    "document_vector_budget"
)

comptime DOCUMENT_ENCODER_COMPRESSION_KIND_NONE = "none"
comptime DOCUMENT_ENCODER_COMPRESSION_KIND_SEQUENCE_RESIZING = (
    "sequence_resizing"
)
comptime DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS = "memory_tokens"
comptime DOCUMENT_ENCODER_COMPRESSION_KIND_ATTENTION_GUIDED_CLUSTERING = (
    "attention_guided_clustering"
)


struct DocumentEncoderCompressionConfigEntry(Copyable):
    var key: String
    var value: String

    def __init__(out self, var key: String, var value: String) raises:
        self.key = require_non_empty_string(
            key, "document encoder compression config key"
        )
        self.value = require_non_empty_string(
            value, "document encoder compression config value"
        )


def require_unique_document_encoder_compression_config_keys(
    read entries: List[DocumentEncoderCompressionConfigEntry]
) raises:
    for left_index in range(len(entries)):
        for right_index in range(left_index + 1, len(entries)):
            if entries[left_index].key == entries[right_index].key:
                raise Error(
                    "duplicate document encoder compression config key: "
                    + entries[left_index].key
                )


def copy_document_encoder_compression_config_entries(
    read entries: List[DocumentEncoderCompressionConfigEntry]
) raises -> List[DocumentEncoderCompressionConfigEntry]:
    require_unique_document_encoder_compression_config_keys(entries)

    var copied = List[DocumentEncoderCompressionConfigEntry]()
    for entry in entries:
        copied.append(entry.copy())
    return copied^


def same_document_encoder_compression_config_entries(
    read left: List[DocumentEncoderCompressionConfigEntry],
    read right: List[DocumentEncoderCompressionConfigEntry],
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].key != right[index].key:
            return False
        if left[index].value != right[index].value:
            return False

    return True


def require_document_encoder_compression_kind_supported(value: String) raises -> String:
    var normalized = require_non_empty_string(
        value, "document encoder compression kind"
    )
    if (
        normalized != DOCUMENT_ENCODER_COMPRESSION_KIND_NONE
        and normalized != DOCUMENT_ENCODER_COMPRESSION_KIND_SEQUENCE_RESIZING
        and normalized != DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS
        and normalized
        != DOCUMENT_ENCODER_COMPRESSION_KIND_ATTENTION_GUIDED_CLUSTERING
    ):
        raise Error(
            "unsupported document encoder compression kind: " + normalized
        )

    return normalized^


struct DocumentEncoderCompressionManifest(Copyable):
    var kind: String
    var config: List[DocumentEncoderCompressionConfigEntry]

    def __init__(out self) raises:
        self = default_document_encoder_compression_manifest()

    def __init__(out self, var kind: String) raises:
        self = DocumentEncoderCompressionManifest(kind, [])

    def __init__(
        out self,
        var kind: String,
        read config: List[DocumentEncoderCompressionConfigEntry],
    ) raises:
        self.kind = require_document_encoder_compression_kind_supported(kind)
        self.config = copy_document_encoder_compression_config_entries(config)
        if self.kind == DOCUMENT_ENCODER_COMPRESSION_KIND_NONE and len(self.config) != 0:
            raise Error(
                "document encoder compression kind none must not define config entries"
            )


def default_document_encoder_compression_manifest(
) raises -> DocumentEncoderCompressionManifest:
    return DocumentEncoderCompressionManifest(
        DOCUMENT_ENCODER_COMPRESSION_KIND_NONE,
        [],
    )


def same_document_encoder_compression_manifest(
    read left: DocumentEncoderCompressionManifest,
    read right: DocumentEncoderCompressionManifest,
) -> Bool:
    return (
        left.kind == right.kind
        and same_document_encoder_compression_config_entries(
            left.config,
            right.config,
        )
    )


def has_document_encoder_compression(
    read manifest: DocumentEncoderCompressionManifest
) -> Bool:
    return manifest.kind != DOCUMENT_ENCODER_COMPRESSION_KIND_NONE


def document_encoder_compression_config_value(
    read manifest: DocumentEncoderCompressionManifest, key: String
) -> String:
    for entry in manifest.config:
        if entry.key == key:
            return entry.value.copy()

    return ""


def budgeted_document_encoder_compression_manifest(
    kind: String, document_vector_budget: Int
) raises -> DocumentEncoderCompressionManifest:
    var supported = require_document_encoder_compression_kind_supported(kind)
    if supported == DOCUMENT_ENCODER_COMPRESSION_KIND_NONE:
        raise Error(
            "budgeted document encoder compression manifest requires a compression kind"
        )

    _ = require_positive_int(
        document_vector_budget,
        "document encoder compression document_vector_budget",
    )
    return DocumentEncoderCompressionManifest(
        supported^,
        [
            DocumentEncoderCompressionConfigEntry(
                DOCUMENT_ENCODER_COMPRESSION_CONFIG_DOCUMENT_VECTOR_BUDGET,
                String(document_vector_budget),
            )
        ],
    )


def sequence_resizing_document_encoder_compression(
    document_vector_budget: Int
) raises -> DocumentEncoderCompressionManifest:
    return budgeted_document_encoder_compression_manifest(
        DOCUMENT_ENCODER_COMPRESSION_KIND_SEQUENCE_RESIZING,
        document_vector_budget,
    )


def memory_tokens_document_encoder_compression(
    document_vector_budget: Int
) raises -> DocumentEncoderCompressionManifest:
    return budgeted_document_encoder_compression_manifest(
        DOCUMENT_ENCODER_COMPRESSION_KIND_MEMORY_TOKENS,
        document_vector_budget,
    )


def attention_guided_clustering_document_encoder_compression(
    document_vector_budget: Int
) raises -> DocumentEncoderCompressionManifest:
    return budgeted_document_encoder_compression_manifest(
        DOCUMENT_ENCODER_COMPRESSION_KIND_ATTENTION_GUIDED_CLUSTERING,
        document_vector_budget,
    )
