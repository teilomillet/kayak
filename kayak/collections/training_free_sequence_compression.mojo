# Owns the paper-shaped method surface for training-free sequence compression.
# It lowers explicit method specs into document-representation transforms.
# It does not own benchmark evaluation or search metrics.

from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.index import PackedIndex

from .document_representation_transform import (
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    DocumentRepresentationTransformManifest,
    budgeted_token_pooling_document_representation_transform,
    prefix_pruning_document_representation_transform,
    require_document_representation_transform_protected_token_position_supported,
)
from .document_representation_transform_runtime import (
    apply_document_representation_transforms_to_documents,
    apply_document_representation_transforms_to_packed_index,
)


comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT = "full_exact"
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING = (
    "prefix_pruning"
)
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING = "token_pooling"

comptime TRAINING_FREE_SEQUENCE_COMPRESSION_DEFAULT_PROTECTED_TOKEN_COUNT = 1
comptime TRAINING_FREE_SEQUENCE_COMPRESSION_DEFAULT_PROTECTED_TOKEN_POSITION = (
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
)


struct TrainingFreeSequenceCompressionSpec(Copyable):
    var available_document_vector_count: Int
    var requested_document_vector_budget: Int
    var method_kind: String
    var transform_policy: String
    var derived_pool_factor: Int
    var protected_token_count: Int
    var protected_token_position: String

    def __init__(
        out self,
        available_document_vector_count: Int,
        requested_document_vector_budget: Int,
        var method_kind: String,
        var transform_policy: String,
        derived_pool_factor: Int,
        protected_token_count: Int,
        var protected_token_position: String,
    ) raises:
        if available_document_vector_count <= 0:
            raise Error(
                "training-free sequence compression available_document_vector_count must be positive"
            )
        if requested_document_vector_budget <= 0:
            raise Error(
                "training-free sequence compression requested_document_vector_budget must be positive"
            )
        if (
            method_kind != TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT
            and method_kind
            != TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING
            and method_kind
            != TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING
        ):
            raise Error(
                "unsupported training-free sequence compression method kind: "
                + method_kind
            )
        if protected_token_count < 0:
            raise Error(
                "training-free sequence compression protected_token_count must be non-negative"
            )
        if derived_pool_factor < 0:
            raise Error(
                "training-free sequence compression derived_pool_factor must be non-negative"
            )
        self.available_document_vector_count = available_document_vector_count
        self.requested_document_vector_budget = requested_document_vector_budget
        self.method_kind = method_kind^
        self.transform_policy = transform_policy^
        self.derived_pool_factor = derived_pool_factor
        self.protected_token_count = protected_token_count
        self.protected_token_position = protected_token_position^


def supported_training_free_sequence_compression_token_pooling_policies(
) -> List[String]:
    return [
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    ]


def default_training_free_sequence_compression_policies() -> List[String]:
    return [
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
        DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    ]


def pool_factor_for_target_document_vector_budget(
    available_count: Int, requested_budget: Int
) raises -> Int:
    if available_count <= 0:
        raise Error("sequence compression requires a positive available vector count")
    if requested_budget <= 0:
        raise Error("sequence compression requested budget must be positive")
    if requested_budget >= available_count:
        return 1

    var factor = available_count // requested_budget
    if available_count % requested_budget != 0:
        factor += 1
    if factor <= 0:
        return 1
    return factor


def training_free_sequence_compression_spec(
    available_document_vector_count: Int,
    method_kind: String,
    requested_document_vector_budget: Int,
    transform_policy: String = "",
    protected_token_count: Int = (
        TRAINING_FREE_SEQUENCE_COMPRESSION_DEFAULT_PROTECTED_TOKEN_COUNT
    ),
    protected_token_position: String = (
        TRAINING_FREE_SEQUENCE_COMPRESSION_DEFAULT_PROTECTED_TOKEN_POSITION
    ),
) raises -> TrainingFreeSequenceCompressionSpec:
    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT:
        return TrainingFreeSequenceCompressionSpec(
            available_document_vector_count,
            available_document_vector_count,
            method_kind.copy(),
            "",
            0,
            0,
            "",
        )

    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING:
        return TrainingFreeSequenceCompressionSpec(
            available_document_vector_count,
            requested_document_vector_budget,
            method_kind.copy(),
            DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
            0,
            0,
            "",
        )

    if method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_TOKEN_POOLING:
        var policy = transform_policy.copy()
        if policy.byte_length() == 0:
            policy = DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL
        var supported = False
        for candidate in supported_training_free_sequence_compression_token_pooling_policies():
            if policy == candidate:
                supported = True
                break
        if not supported:
            raise Error(
                "unsupported training-free sequence compression token-pooling policy: "
                + policy
            )
        return TrainingFreeSequenceCompressionSpec(
            available_document_vector_count,
            requested_document_vector_budget,
            method_kind.copy(),
            policy^,
            pool_factor_for_target_document_vector_budget(
                available_document_vector_count,
                requested_document_vector_budget,
            ),
            protected_token_count,
            require_document_representation_transform_protected_token_position_supported(
                protected_token_position
            ),
        )

    raise Error("unsupported training-free sequence compression method kind: " + method_kind)


def training_free_sequence_compression_transforms(
    read spec: TrainingFreeSequenceCompressionSpec
) raises -> List[DocumentRepresentationTransformManifest]:
    if spec.method_kind == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_FULL_EXACT:
        return []
    if (
        spec.method_kind
        == TRAINING_FREE_SEQUENCE_COMPRESSION_METHOD_PREFIX_PRUNING
    ):
        return [
            prefix_pruning_document_representation_transform(
                spec.requested_document_vector_budget,
                spec.transform_policy,
            )
        ]
    return [
        budgeted_token_pooling_document_representation_transform(
            spec.requested_document_vector_budget,
            spec.transform_policy,
            spec.protected_token_count,
            spec.protected_token_position,
        )
    ]


def apply_training_free_sequence_compression_to_documents(
    read documents: List[EncodedDocument],
    read spec: TrainingFreeSequenceCompressionSpec,
) raises -> List[EncodedDocument]:
    return apply_document_representation_transforms_to_documents(
        documents,
        training_free_sequence_compression_transforms(spec),
    )


def apply_training_free_sequence_compression_to_packed_index(
    read index: PackedIndex,
    read spec: TrainingFreeSequenceCompressionSpec,
) raises -> PackedIndex:
    return apply_document_representation_transforms_to_packed_index(
        index,
        training_free_sequence_compression_transforms(spec),
    )
