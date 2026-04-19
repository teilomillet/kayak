# Owns the paper-shaped method surface for constant-budget multi-vector
# index compression. It classifies which methods can be lowered from stored
# document vectors versus which still require encoder-bound learned behavior.

from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.index import PackedIndex

from .document_representation_transform import (
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    DocumentRepresentationTransformManifest,
    budgeted_token_pooling_document_representation_transform,
    copy_document_representation_transforms,
    require_document_representation_transform_protected_token_position_supported,
)
from .document_encoder_compression import (
    DocumentEncoderCompressionManifest,
    attention_guided_clustering_document_encoder_compression,
    default_document_encoder_compression_manifest,
    has_document_encoder_compression,
    memory_tokens_document_encoder_compression,
    sequence_resizing_document_encoder_compression,
)
from .document_representation_transform_runtime import (
    apply_document_representation_transforms_to_documents,
    apply_document_representation_transforms_to_packed_index,
)


comptime MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT = "full_exact"
comptime MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING = (
    "hierarchical_pooling"
)
comptime MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING = (
    "sequence_resizing"
)
comptime MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS = "memory_tokens"
comptime MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING = (
    "attention_guided_clustering"
)

comptime MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION = (
    "stored_representation"
)
comptime MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER = "encoder"

comptime MULTI_VECTOR_INDEX_COMPRESSION_DEFAULT_PROTECTED_TOKEN_COUNT = 1
comptime MULTI_VECTOR_INDEX_COMPRESSION_DEFAULT_PROTECTED_TOKEN_POSITION = (
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
)


struct MultiVectorIndexCompressionSpec(Copyable):
    var available_document_vector_count: Int
    var requested_document_vector_budget: Int
    var method_kind: String
    var execution_boundary: String
    var transform_policy: String
    var derived_pool_factor: Int
    var protected_token_count: Int
    var protected_token_position: String

    def __init__(
        out self,
        available_document_vector_count: Int,
        requested_document_vector_budget: Int,
        var method_kind: String,
        var execution_boundary: String,
        var transform_policy: String,
        derived_pool_factor: Int,
        protected_token_count: Int,
        var protected_token_position: String,
    ) raises:
        if available_document_vector_count <= 0:
            raise Error(
                "multi-vector index compression available_document_vector_count must be positive"
            )
        if requested_document_vector_budget <= 0:
            raise Error(
                "multi-vector index compression requested_document_vector_budget must be positive"
            )
        _ = require_multi_vector_index_compression_method_supported(method_kind)
        if (
            execution_boundary
            != MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION
            and execution_boundary
            != MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER
        ):
            raise Error(
                "unsupported multi-vector index compression execution boundary: "
                + execution_boundary
            )
        if protected_token_count < 0:
            raise Error(
                "multi-vector index compression protected_token_count must be non-negative"
            )
        if derived_pool_factor < 0:
            raise Error(
                "multi-vector index compression derived_pool_factor must be non-negative"
            )
        self.available_document_vector_count = available_document_vector_count
        self.requested_document_vector_budget = requested_document_vector_budget
        self.method_kind = method_kind^
        self.execution_boundary = execution_boundary^
        self.transform_policy = transform_policy^
        self.derived_pool_factor = derived_pool_factor
        self.protected_token_count = protected_token_count
        self.protected_token_position = protected_token_position^


struct MultiVectorIndexCompressionLowering(Copyable):
    var spec: MultiVectorIndexCompressionSpec
    var document_representation_transforms: List[
        DocumentRepresentationTransformManifest
    ]
    var document_encoder_compression: DocumentEncoderCompressionManifest

    def __init__(
        out self,
        read spec: MultiVectorIndexCompressionSpec,
        read document_representation_transforms: List[
            DocumentRepresentationTransformManifest
        ],
        read document_encoder_compression: DocumentEncoderCompressionManifest,
    ) raises:
        self.spec = spec.copy()
        self.document_representation_transforms = (
            copy_document_representation_transforms(
                document_representation_transforms
            )
        )
        self.document_encoder_compression = document_encoder_compression.copy()

        if (
            self.spec.execution_boundary
            == MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION
        ):
            if has_document_encoder_compression(self.document_encoder_compression):
                raise Error(
                    "stored-representation multi-vector compression lowering must not define document encoder compression"
                )
        else:
            if len(self.document_representation_transforms) != 0:
                raise Error(
                    "encoder-bound multi-vector compression lowering must not define stored-document transforms"
                )
            if not has_document_encoder_compression(
                self.document_encoder_compression
            ):
                raise Error(
                    "encoder-bound multi-vector compression lowering must define document encoder compression provenance"
                )

        if self.spec.method_kind == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT:
            if len(self.document_representation_transforms) != 0:
                raise Error(
                    "full_exact multi-vector compression lowering must not define stored-document transforms"
                )
            if has_document_encoder_compression(self.document_encoder_compression):
                raise Error(
                    "full_exact multi-vector compression lowering must not define document encoder compression"
                )


def supported_multi_vector_index_compression_methods() -> List[String]:
    return [
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING,
    ]


def stored_representation_multi_vector_index_compression_methods() -> List[String]:
    return [
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT,
        MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING,
    ]


def require_multi_vector_index_compression_method_supported(
    value: String
) raises -> String:
    for method_kind in supported_multi_vector_index_compression_methods():
        if value == method_kind:
            return value.copy()

    raise Error("unsupported multi-vector index compression method kind: " + value)


def multi_vector_index_compression_method_execution_boundary(
    method_kind: String
) raises -> String:
    var supported = require_multi_vector_index_compression_method_supported(
        method_kind
    )
    if (
        supported == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING
        or supported == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS
        or supported
        == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_ATTENTION_GUIDED_CLUSTERING
    ):
        return MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER

    return MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION


def multi_vector_index_compression_method_is_stored_representation_executable(
    method_kind: String
) raises -> Bool:
    return (
        multi_vector_index_compression_method_execution_boundary(method_kind)
        == MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION
    )


def require_multi_vector_index_compression_method_stored_representation_executable(
    method_kind: String
) raises -> String:
    var supported = require_multi_vector_index_compression_method_supported(
        method_kind
    )
    if multi_vector_index_compression_method_is_stored_representation_executable(
        supported
    ):
        return supported^

    raise Error(
        "multi-vector index compression method "
        + supported
        + " requires encoder-bound learned compression and cannot be lowered from stored document vectors in kayak yet"
    )


def pool_factor_for_multi_vector_index_compression_budget(
    available_count: Int, requested_budget: Int
) raises -> Int:
    if available_count <= 0:
        raise Error(
            "multi-vector index compression requires a positive available vector count"
        )
    if requested_budget <= 0:
        raise Error(
            "multi-vector index compression requested budget must be positive"
        )
    if requested_budget >= available_count:
        return 1

    var factor = available_count // requested_budget
    if available_count % requested_budget != 0:
        factor += 1
    if factor <= 0:
        return 1
    return factor


def multi_vector_index_compression_spec(
    available_document_vector_count: Int,
    method_kind: String,
    requested_document_vector_budget: Int,
    protected_token_count: Int = (
        MULTI_VECTOR_INDEX_COMPRESSION_DEFAULT_PROTECTED_TOKEN_COUNT
    ),
    protected_token_position: String = (
        MULTI_VECTOR_INDEX_COMPRESSION_DEFAULT_PROTECTED_TOKEN_POSITION
    ),
) raises -> MultiVectorIndexCompressionSpec:
    var supported = require_multi_vector_index_compression_method_supported(
        method_kind
    )

    if supported == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT:
        return MultiVectorIndexCompressionSpec(
            available_document_vector_count,
            available_document_vector_count,
            supported^,
            MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
            "",
            0,
            0,
            "",
        )

    if supported == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_HIERARCHICAL_POOLING:
        return MultiVectorIndexCompressionSpec(
            available_document_vector_count,
            requested_document_vector_budget,
            supported^,
            MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION,
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
            pool_factor_for_multi_vector_index_compression_budget(
                available_document_vector_count,
                requested_document_vector_budget,
            ),
            protected_token_count,
            require_document_representation_transform_protected_token_position_supported(
                protected_token_position
            ),
        )

    return MultiVectorIndexCompressionSpec(
        available_document_vector_count,
        requested_document_vector_budget,
        supported^,
        MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_ENCODER,
        "",
        0,
        0,
        "",
    )


def multi_vector_index_compression_transforms(
    read spec: MultiVectorIndexCompressionSpec
) raises -> List[DocumentRepresentationTransformManifest]:
    var lowering = multi_vector_index_compression_lowering(spec)

    if (
        spec.execution_boundary
        != MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION
    ):
        raise Error(
            "multi-vector index compression method "
            + spec.method_kind
            + " requires encoder-bound learned compression and cannot be lowered from stored document vectors in kayak yet"
        )

    return lowering.document_representation_transforms.copy()


def multi_vector_index_compression_document_encoder_compression(
    read spec: MultiVectorIndexCompressionSpec
) raises -> DocumentEncoderCompressionManifest:
    _ = require_multi_vector_index_compression_method_supported(spec.method_kind)

    return multi_vector_index_compression_lowering(
        spec
    ).document_encoder_compression.copy()


def multi_vector_index_compression_lowering(
    read spec: MultiVectorIndexCompressionSpec
) raises -> MultiVectorIndexCompressionLowering:
    _ = require_multi_vector_index_compression_method_supported(spec.method_kind)

    if (
        spec.execution_boundary
        == MULTI_VECTOR_INDEX_COMPRESSION_EXECUTION_BOUNDARY_STORED_REPRESENTATION
    ):
        if spec.method_kind == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_FULL_EXACT:
            return MultiVectorIndexCompressionLowering(
                spec,
                [],
                default_document_encoder_compression_manifest(),
            )
        return MultiVectorIndexCompressionLowering(
            spec,
            [
                budgeted_token_pooling_document_representation_transform(
                    spec.requested_document_vector_budget,
                    spec.transform_policy,
                    spec.protected_token_count,
                    spec.protected_token_position,
                )
            ],
            default_document_encoder_compression_manifest(),
        )

    if spec.method_kind == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_SEQUENCE_RESIZING:
        return MultiVectorIndexCompressionLowering(
            spec,
            [],
            sequence_resizing_document_encoder_compression(
                spec.requested_document_vector_budget
            ),
        )
    if spec.method_kind == MULTI_VECTOR_INDEX_COMPRESSION_METHOD_MEMORY_TOKENS:
        return MultiVectorIndexCompressionLowering(
            spec,
            [],
            memory_tokens_document_encoder_compression(
                spec.requested_document_vector_budget
            ),
        )
    return MultiVectorIndexCompressionLowering(
        spec,
        [],
        attention_guided_clustering_document_encoder_compression(
            spec.requested_document_vector_budget
        ),
    )


def apply_multi_vector_index_compression_to_documents(
    read documents: List[EncodedDocument],
    read spec: MultiVectorIndexCompressionSpec,
) raises -> List[EncodedDocument]:
    return apply_document_representation_transforms_to_documents(
        documents,
        multi_vector_index_compression_transforms(spec),
    )


def apply_multi_vector_index_compression_to_packed_index(
    read index: PackedIndex,
    read spec: MultiVectorIndexCompressionSpec,
) raises -> PackedIndex:
    return apply_document_representation_transforms_to_packed_index(
        index,
        multi_vector_index_compression_transforms(spec),
    )
