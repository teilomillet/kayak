from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.index import PackedIndex, pack_documents, unpack_documents
from kayak.numeric import MetricScalar, VectorScalar, zero_metric_scalar, zero_vector_scalar
from kayak.storage.text_codec import parse_int

from .document_representation_transform import (
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_DOCUMENT_VECTOR_BUDGET,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
    DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_PREFIX_PRUNING,
    DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_HIERARCHICAL,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_PREFIX,
    DOCUMENT_REPRESENTATION_TRANSFORM_POLICY_SEQUENTIAL,
    DocumentRepresentationTransformManifest,
    document_representation_transform_config_value,
)


struct TokenPoolCluster(Copyable):
    var centroid: List[VectorScalar]
    var member_count: Int
    var min_member_index: Int

    def __init__(
        out self,
        read centroid: List[VectorScalar],
        member_count: Int,
        min_member_index: Int,
    ) raises:
        if member_count <= 0:
            raise Error("token pool cluster member_count must be positive")
        if min_member_index < 0:
            raise Error("token pool cluster min_member_index must be non-negative")

        self.centroid = centroid.copy()
        self.member_count = member_count
        self.min_member_index = min_member_index


def target_pooled_vector_count(
    available_count: Int, pool_factor: Int
) raises -> Int:
    if available_count <= 0:
        raise Error("token pooling requires at least one available vector")
    if pool_factor <= 0:
        raise Error("token pooling pool_factor must be positive")

    var target = available_count // pool_factor
    if available_count % pool_factor != 0:
        target += 1
    if target <= 0:
        return 1
    return target


def squared_l2_distance(
    read left: List[VectorScalar], read right: List[VectorScalar]
) raises -> MetricScalar:
    if len(left) != len(right):
        raise Error("squared_l2_distance requires matching dimensions")

    var total = zero_metric_scalar()
    for dim_index in range(len(left)):
        var delta = MetricScalar(left[dim_index]) - MetricScalar(right[dim_index])
        total += delta * delta
    return total


def ward_merge_cost(
    read left: TokenPoolCluster, read right: TokenPoolCluster
) raises -> MetricScalar:
    var left_count = MetricScalar(left.member_count)
    var right_count = MetricScalar(right.member_count)
    return (
        (left_count * right_count) / (left_count + right_count)
    ) * squared_l2_distance(left.centroid, right.centroid)


def average_vectors_in_range(
    read vectors: List[List[VectorScalar]], start: Int, stop: Int
) raises -> List[VectorScalar]:
    if start < 0 or stop <= start or stop > len(vectors):
        raise Error("invalid vector averaging range")

    var vector_dim = len(vectors[start])
    if vector_dim == 0:
        raise Error("token pooling vectors must not be empty")

    var pooled = List[VectorScalar]()
    for _ in range(vector_dim):
        pooled.append(zero_vector_scalar())

    for vector_index in range(start, stop):
        if len(vectors[vector_index]) != vector_dim:
            raise Error("token pooling requires a consistent vector dimension")
        for dim_index in range(vector_dim):
            pooled[dim_index] += vectors[vector_index][dim_index]

    var divisor = VectorScalar(stop - start)
    for dim_index in range(vector_dim):
        pooled[dim_index] = pooled[dim_index] / divisor

    return pooled^


def build_singleton_cluster(
    read vectors: List[List[VectorScalar]], vector_index: Int
) raises -> TokenPoolCluster:
    return TokenPoolCluster(vectors[vector_index].copy(), 1, vector_index)


def merge_token_pool_clusters(
    read left: TokenPoolCluster, read right: TokenPoolCluster
) raises -> TokenPoolCluster:
    if len(left.centroid) != len(right.centroid):
        raise Error("token pool cluster merge requires matching dimensions")

    var merged = List[VectorScalar]()
    var total_count = left.member_count + right.member_count
    var left_weight = VectorScalar(left.member_count)
    var right_weight = VectorScalar(right.member_count)
    var divisor = VectorScalar(total_count)

    for dim_index in range(len(left.centroid)):
        merged.append(
            (
                left.centroid[dim_index] * left_weight
                + right.centroid[dim_index] * right_weight
            )
            / divisor
        )

    var min_member_index = left.min_member_index
    if right.min_member_index < min_member_index:
        min_member_index = right.min_member_index

    return TokenPoolCluster(merged^, total_count, min_member_index)


def sort_clusters_by_min_member_index(
    mut clusters: List[TokenPoolCluster]
):
    for left_index in range(len(clusters)):
        var best_index = left_index
        for right_index in range(left_index + 1, len(clusters)):
            if (
                clusters[right_index].min_member_index
                < clusters[best_index].min_member_index
            ):
                best_index = right_index
        if best_index != left_index:
            var swap = clusters[left_index].copy()
            clusters[left_index] = clusters[best_index].copy()
            clusters[best_index] = swap^


def sequential_token_pool_document(
    read document: EncodedDocument, pool_factor: Int
) raises -> EncodedDocument:
    if pool_factor <= 0:
        raise Error("sequential token pooling pool_factor must be positive")

    if pool_factor == 1 or document.vector_count <= 1:
        return document.copy()

    var pooled_vectors = List[List[VectorScalar]]()
    var start = 0
    while start < document.vector_count:
        var stop = start + pool_factor
        if stop > document.vector_count:
            stop = document.vector_count
        pooled_vectors.append(
            average_vectors_in_range(document.token_vectors, start, stop)
        )
        start = stop

    return EncodedDocument(document.doc_id.copy(), pooled_vectors^)


def hierarchical_token_pool_document(
    read document: EncodedDocument, pool_factor: Int
) raises -> EncodedDocument:
    var target = target_pooled_vector_count(document.vector_count, pool_factor)
    if target >= document.vector_count or document.vector_count <= 1:
        return document.copy()

    var clusters = List[TokenPoolCluster]()
    for vector_index in range(document.vector_count):
        clusters.append(build_singleton_cluster(document.token_vectors, vector_index))

    while len(clusters) > target:
        var best_left_index = 0
        var best_right_index = 1
        var best_cost = ward_merge_cost(clusters[0], clusters[1])

        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                var cost = ward_merge_cost(clusters[left_index], clusters[right_index])
                if cost < best_cost:
                    best_cost = cost
                    best_left_index = left_index
                    best_right_index = right_index
                    continue

                if cost == best_cost:
                    var current_left_min = clusters[left_index].min_member_index
                    var current_right_min = clusters[right_index].min_member_index
                    var best_left_min = clusters[best_left_index].min_member_index
                    var best_right_min = clusters[best_right_index].min_member_index
                    if (
                        current_left_min < best_left_min
                        or (
                            current_left_min == best_left_min
                            and current_right_min < best_right_min
                        )
                    ):
                        best_left_index = left_index
                        best_right_index = right_index

        var merged = merge_token_pool_clusters(
            clusters[best_left_index], clusters[best_right_index]
        )
        var next_clusters = List[TokenPoolCluster]()
        for cluster_index in range(len(clusters)):
            if (
                cluster_index == best_left_index
                or cluster_index == best_right_index
            ):
                continue
            next_clusters.append(clusters[cluster_index].copy())
        next_clusters.append(merged^)
        clusters = next_clusters^
    sort_clusters_by_min_member_index(clusters)

    var pooled_vectors = List[List[VectorScalar]]()
    for cluster in clusters:
        pooled_vectors.append(cluster.centroid.copy())
    return EncodedDocument(document.doc_id.copy(), pooled_vectors^)


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


def apply_document_representation_transform_to_document(
    read document: EncodedDocument,
    read transform: DocumentRepresentationTransformManifest,
) raises -> EncodedDocument:
    if transform.kind == DOCUMENT_REPRESENTATION_TRANSFORM_KIND_TOKEN_POOLING:
        var pool_factor = parse_int(
            document_representation_transform_config_value(
                transform,
                DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POOL_FACTOR,
            ),
            "document representation transform pool_factor",
        )
        var policy = document_representation_transform_config_value(
            transform,
            DOCUMENT_REPRESENTATION_TRANSFORM_CONFIG_POLICY,
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
