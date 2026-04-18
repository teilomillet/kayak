from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.numeric import MetricScalar, VectorScalar, zero_metric_scalar, zero_vector_scalar

from .document_representation_transform import (
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST,
    DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST,
    require_document_representation_transform_protected_token_position_supported,
)
from .validation import require_non_negative_int, require_positive_int


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


def resolved_token_pooling_protected_token_count(
    target_vector_count: Int, requested_protected_token_count: Int
) raises -> Int:
    _ = require_positive_int(
        target_vector_count,
        "token pooling target_vector_count",
    )
    _ = require_non_negative_int(
        requested_protected_token_count,
        "token pooling requested_protected_token_count",
    )

    # Mirror the paper's budget contract so protected tokens never consume the
    # full target budget and there is always room for at least one pooled vector.
    if target_vector_count == 1:
        return 0
    if requested_protected_token_count < target_vector_count:
        return requested_protected_token_count
    return target_vector_count - 1


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


def append_document_vectors_in_range(
    mut output: List[List[VectorScalar]],
    read document: EncodedDocument,
    start: Int,
    stop: Int,
):
    for vector_index in range(start, stop):
        output.append(document.token_vectors[vector_index].copy())


def append_sequential_pooled_ranges(
    mut output: List[List[VectorScalar]],
    read document: EncodedDocument,
    pool_start: Int,
    pool_stop: Int,
    target_cluster_count: Int,
) raises:
    if target_cluster_count <= 0:
        raise Error("sequential token pooling target_cluster_count must be positive")

    var available_count = pool_stop - pool_start
    var base_chunk_size = available_count // target_cluster_count
    var remainder = available_count % target_cluster_count
    var start = pool_start

    for chunk_index in range(target_cluster_count):
        var chunk_size = base_chunk_size
        if chunk_index < remainder:
            chunk_size += 1
        var stop = start + chunk_size
        output.append(
            average_vectors_in_range(document.token_vectors, start, stop)
        )
        start = stop


def sequential_token_pool_document_to_target(
    read document: EncodedDocument,
    target_vector_count: Int,
    protected_token_count: Int = 0,
    protected_token_position: String = (
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
    ),
) raises -> EncodedDocument:
    _ = require_positive_int(
        target_vector_count,
        "sequential token pooling target_vector_count",
    )
    var protected_count = resolved_token_pooling_protected_token_count(
        target_vector_count,
        protected_token_count,
    )
    var protected_position = (
        require_document_representation_transform_protected_token_position_supported(
            protected_token_position
        )
    )

    if target_vector_count >= document.vector_count or document.vector_count <= 1:
        return document.copy()

    var pool_start = 0
    var pool_stop = document.vector_count
    if protected_count > 0:
        if (
            protected_position
            == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
        ):
            pool_start = protected_count
        else:
            pool_stop = document.vector_count - protected_count

    var pooled_vectors = List[List[VectorScalar]]()
    if (
        protected_position
        == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
    ):
        append_document_vectors_in_range(
            pooled_vectors,
            document,
            0,
            pool_start,
        )

    append_sequential_pooled_ranges(
        pooled_vectors,
        document,
        pool_start,
        pool_stop,
        target_vector_count - protected_count,
    )

    if (
        protected_position
        == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST
    ):
        append_document_vectors_in_range(
            pooled_vectors,
            document,
            pool_stop,
            document.vector_count,
        )

    return EncodedDocument(document.doc_id.copy(), pooled_vectors^)


def sequential_token_pool_document(
    read document: EncodedDocument, pool_factor: Int
) raises -> EncodedDocument:
    if pool_factor <= 0:
        raise Error("sequential token pooling pool_factor must be positive")

    if pool_factor == 1 or document.vector_count <= 1:
        return document.copy()

    return sequential_token_pool_document_to_target(
        document,
        target_pooled_vector_count(document.vector_count, pool_factor),
    )


def hierarchical_token_pool_document_to_target(
    read document: EncodedDocument,
    target_vector_count: Int,
    protected_token_count: Int = 0,
    protected_token_position: String = (
        DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
    ),
) raises -> EncodedDocument:
    _ = require_positive_int(
        target_vector_count,
        "hierarchical token pooling target_vector_count",
    )
    var protected_count = resolved_token_pooling_protected_token_count(
        target_vector_count,
        protected_token_count,
    )
    var protected_position = (
        require_document_representation_transform_protected_token_position_supported(
            protected_token_position
        )
    )

    if target_vector_count >= document.vector_count or document.vector_count <= 1:
        return document.copy()

    var pool_start = 0
    var pool_stop = document.vector_count
    if protected_count > 0:
        if (
            protected_position
            == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
        ):
            pool_start = protected_count
        else:
            pool_stop = document.vector_count - protected_count

    var clusters = List[TokenPoolCluster]()
    for vector_index in range(pool_start, pool_stop):
        clusters.append(build_singleton_cluster(document.token_vectors, vector_index))

    while len(clusters) > target_vector_count - protected_count:
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
    if (
        protected_position
        == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_FIRST
    ):
        append_document_vectors_in_range(
            pooled_vectors,
            document,
            0,
            pool_start,
        )

    for cluster in clusters:
        pooled_vectors.append(cluster.centroid.copy())

    if (
        protected_position
        == DOCUMENT_REPRESENTATION_TRANSFORM_PROTECTED_TOKEN_POSITION_LAST
    ):
        append_document_vectors_in_range(
            pooled_vectors,
            document,
            pool_stop,
            document.vector_count,
        )

    return EncodedDocument(document.doc_id.copy(), pooled_vectors^)


def hierarchical_token_pool_document(
    read document: EncodedDocument, pool_factor: Int
) raises -> EncodedDocument:
    return hierarchical_token_pool_document_to_target(
        document,
        target_pooled_vector_count(document.vector_count, pool_factor),
    )
