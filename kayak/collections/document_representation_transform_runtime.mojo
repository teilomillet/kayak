from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.index import PackedIndex, pack_documents, unpack_documents
from kayak.numeric import VectorScalar
from kayak.storage.text_codec import parse_int

from .document_representation_transform import (
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    DocumentRepresentationTransformManifest,
    document_representation_transform_config_value,
)
from .token_pooling_runtime import (
    hierarchical_token_pool_document,
    hierarchical_token_pool_document_to_target,
    sequential_token_pool_document,
    sequential_token_pool_document_to_target,
)

def prefix_prune_document(
    read document: EncodedDocument, document_vector_budget: Int
) raises -> EncodedDocument:
    if document_vector_budget <= 0:
        raise Error("prefix pruning budget must be positive")

    var budget = document_vector_budget
    if budget > document.vector_count:
        budget = document.vector_count

    if budget == document.vector_count:
        return document.copy()

    var token_vectors = List[List[VectorScalar]]()
    for vector_index in range(budget):
        token_vectors.append(document.token_vectors[vector_index].copy())
    return EncodedDocument(document.doc_id.copy(), token_vectors^)


def parse_optional_document_representation_transform_config_int(
    read transform: DocumentRepresentationTransformManifest,
    key: String,
    owner: String,
) raises -> Int:
    var value = document_representation_transform_config_value(transform, key)
    if value.byte_length() == 0:
        return 0

    return parse_int(value, owner)


def apply_document_representation_transform_to_document(
    read document: EncodedDocument,
    read transform: DocumentRepresentationTransformManifest,
) raises -> EncodedDocument:
    if transform.kind == DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING:
        var pool_factor_text = document_representation_transform_config_value(
            transform,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
        )
        var document_vector_budget_text = (
            document_representation_transform_config_value(
                transform,
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
            )
        )
        var policy = document_representation_transform_config_value(
            transform,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
        )
        var protected_token_count = (
            parse_optional_document_representation_transform_config_int(
                transform,
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_COUNT,
                "document representation transform protected_token_count",
            )
        )
        var protected_token_position = document_representation_transform_config_value(
            transform,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_PROTECTED_TOKEN_POSITION,
        )
        if protected_token_position.byte_length() == 0:
            protected_token_position = (
                DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
            )

        if (
            pool_factor_text.byte_length() != 0
            and document_vector_budget_text.byte_length() != 0
        ):
            raise Error(
                "token pooling transform requires exactly one of pool_factor or document_vector_budget"
            )
        if (
            pool_factor_text.byte_length() == 0
            and document_vector_budget_text.byte_length() == 0
        ):
            raise Error(
                "token pooling transform requires pool_factor or document_vector_budget"
            )

        if document_vector_budget_text.byte_length() != 0:
            var document_vector_budget = parse_int(
                document_vector_budget_text,
                "document representation transform document_vector_budget",
            )
            if policy == DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL:
                return sequential_token_pool_document_to_target(
                    document,
                    document_vector_budget,
                    protected_token_count,
                    protected_token_position,
                )
            if policy == DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL:
                return hierarchical_token_pool_document_to_target(
                    document,
                    document_vector_budget,
                    protected_token_count,
                    protected_token_position,
                )
            raise Error("unsupported token pooling policy: " + policy)

        var pool_factor = parse_int(
            pool_factor_text,
            "document representation transform pool_factor",
        )
        if policy == DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL:
            return sequential_token_pool_document(document, pool_factor)
        if policy == DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL:
            return hierarchical_token_pool_document(document, pool_factor)
        raise Error("unsupported token pooling policy: " + policy)

    if transform.kind == DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING:
        var policy = document_representation_transform_config_value(
            transform,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
        )
        if policy != DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX:
            raise Error("unsupported prefix pruning policy: " + policy)
        var document_vector_budget = parse_int(
            document_representation_transform_config_value(
                transform,
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
            ),
            "document representation transform document_vector_budget",
        )
        return prefix_prune_document(document, document_vector_budget)

    raise Error(
        "unsupported document representation transform kind: " + transform.kind
    )


def apply_document_representation_transforms_to_document(
    read document: EncodedDocument,
    read transforms: List[DocumentRepresentationTransformManifest],
) raises -> EncodedDocument:
    var current = document.copy()
    for transform in transforms:
        current = apply_document_representation_transform_to_document(
            current,
            transform,
        )
    return current^


def apply_document_representation_transforms_to_documents(
    read documents: List[EncodedDocument],
    read transforms: List[DocumentRepresentationTransformManifest],
) raises -> List[EncodedDocument]:
    if len(transforms) == 0:
        return documents.copy()

    var transformed = List[EncodedDocument]()
    for document in documents:
        transformed.append(
            apply_document_representation_transforms_to_document(document, transforms)
        )
    return transformed^


def apply_document_representation_transforms_to_packed_index(
    read index: PackedIndex,
    read transforms: List[DocumentRepresentationTransformManifest],
) raises -> PackedIndex:
    if len(transforms) == 0:
        return index.copy()

    return pack_documents(
        apply_document_representation_transforms_to_documents(
            unpack_documents(index),
            transforms,
        )
    )
