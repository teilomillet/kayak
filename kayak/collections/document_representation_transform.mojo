from std.collections import List

from .validation import require_non_empty_string, require_positive_int


comptime DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET = (
    "document_vector_budget"
)
comptime DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR = "pool_factor"
comptime DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY = "policy"

comptime DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING = "prefix_pruning"
comptime DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING = "token_pooling"


struct DocumentRepresentationTransformConfigEntry(Copyable):
    var key: String
    var value: String

    def __init__(out self, var key: String, var value: String) raises:
        self.key = require_non_empty_string(
            key, "document representation transform config key"
        )
        self.value = require_non_empty_string(
            value, "document representation transform config value"
        )


def require_unique_document_representation_transform_config_keys(
    read entries: List[DocumentRepresentationTransformConfigEntry]
) raises:
    for left_index in range(len(entries)):
        for right_index in range(left_index + 1, len(entries)):
            if entries[left_index].key == entries[right_index].key:
                raise Error(
                    "duplicate document representation transform config key: "
                    + entries[left_index].key
                )


def copy_document_representation_transform_config_entries(
    read entries: List[DocumentRepresentationTransformConfigEntry]
) raises -> List[DocumentRepresentationTransformConfigEntry]:
    require_unique_document_representation_transform_config_keys(entries)

    var copied = List[DocumentRepresentationTransformConfigEntry]()
    for entry in entries:
        copied.append(entry.copy())
    return copied^


def same_document_representation_transform_config_entries(
    read left: List[DocumentRepresentationTransformConfigEntry],
    read right: List[DocumentRepresentationTransformConfigEntry],
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].key != right[index].key:
            return False
        if left[index].value != right[index].value:
            return False

    return True


struct DocumentRepresentationTransformManifest(Copyable):
    var kind: String
    var config: List[DocumentRepresentationTransformConfigEntry]

    def __init__(out self, var kind: String) raises:
        self = DocumentRepresentationTransformManifest(kind, [])

    def __init__(
        out self,
        var kind: String,
        read config: List[DocumentRepresentationTransformConfigEntry],
    ) raises:
        self.kind = require_non_empty_string(
            kind, "document representation transform kind"
        )
        self.config = copy_document_representation_transform_config_entries(config)


def copy_document_representation_transforms(
    read transforms: List[DocumentRepresentationTransformManifest]
) raises -> List[DocumentRepresentationTransformManifest]:
    var copied = List[DocumentRepresentationTransformManifest]()
    for transform in transforms:
        copied.append(transform.copy())
    return copied^


def same_document_representation_transforms(
    read left: List[DocumentRepresentationTransformManifest],
    read right: List[DocumentRepresentationTransformManifest],
) -> Bool:
    if len(left) != len(right):
        return False

    for index in range(len(left)):
        if left[index].kind != right[index].kind:
            return False
        if not same_document_representation_transform_config_entries(
            left[index].config, right[index].config
        ):
            return False

    return True


def document_representation_transform_config_value(
    read transform: DocumentRepresentationTransformManifest, key: String
) -> String:
    for entry in transform.config:
        if entry.key == key:
            return entry.value.copy()

    return ""


def document_representation_transforms_have_kind(
    read transforms: List[DocumentRepresentationTransformManifest], kind: String
) -> Bool:
    for transform in transforms:
        if transform.kind == kind:
            return True

    return False


def prefix_pruning_document_representation_transform(
    document_vector_budget: Int, policy: String = "prefix"
) raises -> DocumentRepresentationTransformManifest:
    _ = require_positive_int(
        document_vector_budget,
        "document representation transform document_vector_budget",
    )
    return DocumentRepresentationTransformManifest(
        DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
        [
            DocumentRepresentationTransformConfigEntry(
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
                String(document_vector_budget),
            ),
            DocumentRepresentationTransformConfigEntry(
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
                require_non_empty_string(
                    policy, "document representation transform policy"
                ),
            ),
        ],
    )


def token_pooling_document_representation_transform(
    pool_factor: Int, policy: String = "hierarchical"
) raises -> DocumentRepresentationTransformManifest:
    _ = require_positive_int(
        pool_factor,
        "document representation transform pool_factor",
    )
    return DocumentRepresentationTransformManifest(
        DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
        [
            DocumentRepresentationTransformConfigEntry(
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
                String(pool_factor),
            ),
            DocumentRepresentationTransformConfigEntry(
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
                require_non_empty_string(
                    policy, "document representation transform policy"
                ),
            ),
        ],
    )
