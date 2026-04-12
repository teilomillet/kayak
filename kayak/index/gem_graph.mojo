# GEM-family stage-1 artifact.
#
# Owns:
# - quantized per-document code histograms
# - coarse cluster profiles used for cluster filtering
# - graph adjacency used for native stage-1 traversal
#
# Does not own:
# - exact late-interaction token vectors for final scoring
#
# Assumptions:
# - clustering is deterministic k-means over the current packed index
# - qCH follows the paper's quantized Chamfer shape
# - graph construction uses a greedy quantized-EMD approximation; the paper uses
#   optimal qEMD, which would require a dedicated transport solver that does not
#   yet exist in this repo

from std.collections import List
from std.math import log2

from kayak.contracts import EncodedQuery
from kayak.numeric import (
    MetricScalar,
    ScoreScalar,
    VectorScalar,
    min_score_scalar,
    zero_metric_scalar,
    zero_vector_scalar,
)
from kayak.scoring.dot import dot_product

from .packed_index import PackedIndex


comptime DEFAULT_GEM_GRAPH_FINE_CLUSTER_REFINEMENT_STEPS = 4
comptime DEFAULT_GEM_GRAPH_COARSE_CLUSTER_REFINEMENT_STEPS = 4
comptime DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT = 6
comptime DEFAULT_GEM_GRAPH_DEGREE_LIMIT = 8
comptime DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K = 2
comptime DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH = 32


struct GemGraphIndex(Copyable):
    var doc_ids: List[String]
    var doc_code_offsets: List[Int]
    var doc_code_ids: List[Int]
    var doc_code_counts: List[Int]
    var quantization_centroids: List[List[VectorScalar]]
    var index_centroids: List[List[VectorScalar]]
    var quantization_to_index: List[Int]
    var doc_profile_offsets: List[Int]
    var doc_profile_cluster_ids: List[Int]
    var doc_profile_scores: List[ScoreScalar]
    var cluster_offsets: List[Int]
    var cluster_doc_indices: List[Int]
    var entry_doc_indices: List[Int]
    var neighbor_offsets: List[Int]
    var neighbor_doc_indices: List[Int]
    var vector_dim: Int
    var document_count: Int
    var total_token_count: Int
    var quantization_centroid_count: Int
    var cluster_count: Int
    var graph_edge_count: Int
    var shortcut_edge_count: Int
    var cluster_cutoff: Int
    var construction_neighbor_count: Int
    var degree_limit: Int

    def __init__(out self):
        self.doc_ids = List[String]()
        self.doc_code_offsets = [0]
        self.doc_code_ids = List[Int]()
        self.doc_code_counts = List[Int]()
        self.quantization_centroids = List[List[VectorScalar]]()
        self.index_centroids = List[List[VectorScalar]]()
        self.quantization_to_index = List[Int]()
        self.doc_profile_offsets = [0]
        self.doc_profile_cluster_ids = List[Int]()
        self.doc_profile_scores = List[ScoreScalar]()
        self.cluster_offsets = [0]
        self.cluster_doc_indices = List[Int]()
        self.entry_doc_indices = List[Int]()
        self.neighbor_offsets = [0]
        self.neighbor_doc_indices = List[Int]()
        self.vector_dim = 0
        self.document_count = 0
        self.total_token_count = 0
        self.quantization_centroid_count = 0
        self.cluster_count = 0
        self.graph_edge_count = 0
        self.shortcut_edge_count = 0
        self.cluster_cutoff = 0
        self.construction_neighbor_count = 0
        self.degree_limit = 0

    def __init__(
        out self,
        var doc_ids: List[String],
        var doc_code_offsets: List[Int],
        var doc_code_ids: List[Int],
        var doc_code_counts: List[Int],
        var quantization_centroids: List[List[VectorScalar]],
        var index_centroids: List[List[VectorScalar]],
        var quantization_to_index: List[Int],
        var doc_profile_offsets: List[Int],
        var doc_profile_cluster_ids: List[Int],
        var doc_profile_scores: List[ScoreScalar],
        var cluster_offsets: List[Int],
        var cluster_doc_indices: List[Int],
        var entry_doc_indices: List[Int],
        var neighbor_offsets: List[Int],
        var neighbor_doc_indices: List[Int],
        vector_dim: Int,
        shortcut_edge_count: Int,
        cluster_cutoff: Int,
        construction_neighbor_count: Int,
        degree_limit: Int,
    ) raises:
        require_valid_gem_graph_index(
            doc_ids,
            doc_code_offsets,
            doc_code_ids,
            doc_code_counts,
            quantization_centroids,
            index_centroids,
            quantization_to_index,
            doc_profile_offsets,
            doc_profile_cluster_ids,
            doc_profile_scores,
            cluster_offsets,
            cluster_doc_indices,
            entry_doc_indices,
            neighbor_offsets,
            neighbor_doc_indices,
            vector_dim,
            shortcut_edge_count,
            cluster_cutoff,
            construction_neighbor_count,
            degree_limit,
        )

        self.doc_ids = doc_ids^
        self.doc_code_offsets = doc_code_offsets^
        self.doc_code_ids = doc_code_ids^
        self.doc_code_counts = doc_code_counts^
        self.quantization_centroids = quantization_centroids^
        self.index_centroids = index_centroids^
        self.quantization_to_index = quantization_to_index^
        self.doc_profile_offsets = doc_profile_offsets^
        self.doc_profile_cluster_ids = doc_profile_cluster_ids^
        self.doc_profile_scores = doc_profile_scores^
        self.cluster_offsets = cluster_offsets^
        self.cluster_doc_indices = cluster_doc_indices^
        self.entry_doc_indices = entry_doc_indices^
        self.neighbor_offsets = neighbor_offsets^
        self.neighbor_doc_indices = neighbor_doc_indices^
        self.vector_dim = vector_dim
        self.document_count = len(self.doc_ids)
        self.total_token_count = sum_ints(self.doc_code_counts)
        self.quantization_centroid_count = len(self.quantization_centroids)
        self.cluster_count = len(self.index_centroids)
        self.graph_edge_count = len(self.neighbor_doc_indices)
        self.shortcut_edge_count = shortcut_edge_count
        self.cluster_cutoff = cluster_cutoff
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit


def sum_ints(read values: List[Int]) -> Int:
    var total = 0
    for value in values:
        total += value
    return total


def require_valid_offsets(
    read offsets: List[Int], terminal: Int, owner: String
) raises:
    if len(offsets) == 0:
        raise Error(owner + " offsets must not be empty")
    if offsets[0] != 0:
        raise Error(owner + " offsets must start at 0")
    for index in range(1, len(offsets)):
        if offsets[index] < offsets[index - 1]:
            raise Error(owner + " offsets must be monotonic")
    if offsets[len(offsets) - 1] != terminal:
        raise Error(owner + " offsets must end at the payload length")


def require_valid_gem_graph_index(
    read doc_ids: List[String],
    read doc_code_offsets: List[Int],
    read doc_code_ids: List[Int],
    read doc_code_counts: List[Int],
    read quantization_centroids: List[List[VectorScalar]],
    read index_centroids: List[List[VectorScalar]],
    read quantization_to_index: List[Int],
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    read doc_profile_scores: List[ScoreScalar],
    read cluster_offsets: List[Int],
    read cluster_doc_indices: List[Int],
    read entry_doc_indices: List[Int],
    read neighbor_offsets: List[Int],
    read neighbor_doc_indices: List[Int],
    vector_dim: Int,
    shortcut_edge_count: Int,
    cluster_cutoff: Int,
    construction_neighbor_count: Int,
    degree_limit: Int,
) raises:
    if shortcut_edge_count < 0:
        raise Error("gem graph shortcut_edge_count must be non-negative")
    if cluster_cutoff < 0:
        raise Error("gem graph cluster_cutoff must be non-negative")
    if construction_neighbor_count < 0:
        raise Error("gem graph construction_neighbor_count must be non-negative")
    if degree_limit < 0:
        raise Error("gem graph degree_limit must be non-negative")

    if len(doc_ids) == 0:
        if vector_dim != 0:
            raise Error("empty gem graph must have vector_dim = 0")
        if len(doc_code_offsets) != 1 or doc_code_offsets[0] != 0:
            raise Error("empty gem graph doc_code_offsets must be [0]")
        if len(doc_code_ids) != 0 or len(doc_code_counts) != 0:
            raise Error("empty gem graph cannot have document code payloads")
        if len(quantization_centroids) != 0 or len(index_centroids) != 0:
            raise Error("empty gem graph cannot have centroids")
        if len(quantization_to_index) != 0:
            raise Error("empty gem graph cannot have quantization_to_index")
        if len(doc_profile_offsets) != 1 or doc_profile_offsets[0] != 0:
            raise Error("empty gem graph doc_profile_offsets must be [0]")
        if len(doc_profile_cluster_ids) != 0 or len(doc_profile_scores) != 0:
            raise Error("empty gem graph cannot have profile payloads")
        if len(cluster_offsets) != 1 or cluster_offsets[0] != 0:
            raise Error("empty gem graph cluster_offsets must be [0]")
        if len(cluster_doc_indices) != 0 or len(entry_doc_indices) != 0:
            raise Error("empty gem graph cannot have cluster members")
        if len(neighbor_offsets) != 1 or neighbor_offsets[0] != 0:
            raise Error("empty gem graph neighbor_offsets must be [0]")
        if len(neighbor_doc_indices) != 0:
            raise Error("empty gem graph cannot have neighbor payloads")
        return

    if vector_dim <= 0:
        raise Error("gem graph vector_dim must be positive")
    if len(doc_code_ids) != len(doc_code_counts):
        raise Error("gem graph doc code ids and counts must match")
    require_valid_offsets(doc_code_offsets, len(doc_code_ids), "gem graph doc_code")
    require_valid_offsets(
        doc_profile_offsets, len(doc_profile_cluster_ids), "gem graph doc_profile"
    )
    if len(doc_profile_cluster_ids) != len(doc_profile_scores):
        raise Error("gem graph doc profile ids and scores must match")
    require_valid_offsets(
        cluster_offsets, len(cluster_doc_indices), "gem graph cluster"
    )
    require_valid_offsets(
        neighbor_offsets, len(neighbor_doc_indices), "gem graph neighbor"
    )
    if len(doc_code_offsets) != len(doc_ids) + 1:
        raise Error("gem graph doc_code_offsets must match document_count + 1")
    if len(doc_profile_offsets) != len(doc_ids) + 1:
        raise Error("gem graph doc_profile_offsets must match document_count + 1")
    if len(neighbor_offsets) != len(doc_ids) + 1:
        raise Error("gem graph neighbor_offsets must match document_count + 1")
    if len(quantization_centroids) == 0:
        raise Error("gem graph requires at least one quantization centroid")
    if len(index_centroids) == 0:
        raise Error("gem graph requires at least one index centroid")
    if len(quantization_to_index) != len(quantization_centroids):
        raise Error("gem graph quantization_to_index must match fine centroid count")
    if len(cluster_offsets) != len(index_centroids) + 1:
        raise Error("gem graph cluster_offsets must match cluster_count + 1")
    if len(entry_doc_indices) != len(index_centroids):
        raise Error("gem graph entry_doc_indices must match cluster_count")

    for centroid in quantization_centroids:
        if len(centroid) != vector_dim:
            raise Error("gem graph quantization centroid dimension mismatch")
    for centroid in index_centroids:
        if len(centroid) != vector_dim:
            raise Error("gem graph index centroid dimension mismatch")

    for code_index in range(len(doc_code_ids)):
        if doc_code_ids[code_index] < 0 or doc_code_ids[code_index] >= len(quantization_centroids):
            raise Error("gem graph document code id is out of range")
        if doc_code_counts[code_index] <= 0:
            raise Error("gem graph document code counts must be positive")

    for cluster_index in quantization_to_index:
        if cluster_index < 0 or cluster_index >= len(index_centroids):
            raise Error("gem graph quantization_to_index is out of range")

    for profile_cluster in doc_profile_cluster_ids:
        if profile_cluster < 0 or profile_cluster >= len(index_centroids):
            raise Error("gem graph profile cluster id is out of range")

    for doc_index in cluster_doc_indices:
        if doc_index < 0 or doc_index >= len(doc_ids):
            raise Error("gem graph cluster member doc index is out of range")

    for cluster_index in range(len(entry_doc_indices)):
        var entry_doc = entry_doc_indices[cluster_index]
        if entry_doc == -1:
            if cluster_offsets[cluster_index] != cluster_offsets[cluster_index + 1]:
                raise Error(
                    "gem graph empty cluster entry can only be -1 for empty clusters"
                )
            continue

        if entry_doc < 0 or entry_doc >= len(doc_ids):
            raise Error("gem graph entry document index is out of range")

    for doc_index in neighbor_doc_indices:
        if doc_index < 0 or doc_index >= len(doc_ids):
            raise Error("gem graph neighbor doc index is out of range")


def max_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0e300)


def choose_effective_cluster_count(
    requested_count: Int, available_count: Int
) raises -> Int:
    if requested_count < 0:
        raise Error("cluster count must be non-negative")
    if available_count <= 0:
        raise Error("cluster count requires at least one vector")
    if requested_count == 0 or requested_count > available_count:
        return available_count
    return requested_count


def squared_l2_distance(
    read lhs: List[VectorScalar], read rhs: List[VectorScalar]
) raises -> MetricScalar:
    if len(lhs) != len(rhs):
        raise Error("squared_l2_distance requires matching dimensions")

    var total = zero_metric_scalar()
    for dim_index in range(len(lhs)):
        var delta = MetricScalar(lhs[dim_index]) - MetricScalar(rhs[dim_index])
        total += delta * delta

    return total


def nearest_centroid_index(
    read vector: List[VectorScalar], read centroids: List[List[VectorScalar]]
) raises -> Int:
    if len(centroids) == 0:
        raise Error("nearest_centroid_index requires at least one centroid")

    var best_index = 0
    var best_distance = max_metric_scalar()
    for centroid_index in range(len(centroids)):
        var distance = squared_l2_distance(vector, centroids[centroid_index])
        if distance < best_distance:
            best_distance = distance
            best_index = centroid_index

    return best_index


def average_selected_vectors(
    read vectors: List[List[VectorScalar]], read selected_indices: List[Int]
) raises -> List[VectorScalar]:
    if len(selected_indices) == 0:
        raise Error("average_selected_vectors requires at least one vector")

    var vector_dim = len(vectors[selected_indices[0]])
    var total = List[VectorScalar]()
    for _ in range(vector_dim):
        total.append(zero_vector_scalar())

    for selected_index in selected_indices:
        for dim_index in range(vector_dim):
            total[dim_index] += vectors[selected_index][dim_index]

    var divisor = VectorScalar(len(selected_indices))
    for dim_index in range(vector_dim):
        total[dim_index] = total[dim_index] / divisor

    return total^


def build_kmeans_centroids(
    read vectors: List[List[VectorScalar]],
    requested_cluster_count: Int,
    refinement_steps: Int,
) raises -> List[List[VectorScalar]]:
    if refinement_steps < 0:
        raise Error("k-means refinement_steps must be non-negative")

    var cluster_count = choose_effective_cluster_count(
        requested_cluster_count, len(vectors)
    )
    var centroids = List[List[VectorScalar]]()
    for cluster_index in range(cluster_count):
        var sample_index = (cluster_index * len(vectors)) // cluster_count
        centroids.append(vectors[sample_index].copy())

    var assignments = List[Int]()
    for _ in range(len(vectors)):
        assignments.append(0)

    for _ in range(refinement_steps):
        for vector_index in range(len(vectors)):
            assignments[vector_index] = nearest_centroid_index(
                vectors[vector_index], centroids
            )

        var members = List[List[Int]]()
        for _ in range(cluster_count):
            members.append(List[Int]())

        for vector_index in range(len(vectors)):
            members[assignments[vector_index]].append(vector_index)

        for cluster_index in range(cluster_count):
            if len(members[cluster_index]) == 0:
                continue
            centroids[cluster_index] = average_selected_vectors(
                vectors, members[cluster_index]
            )

    return centroids^


def append_or_increment_count(
    mut ids: List[Int], mut counts: List[Int], item_id: Int, increment: Int = 1
) raises:
    if increment <= 0:
        raise Error("count increment must be positive")

    for index in range(len(ids)):
        if ids[index] == item_id:
            counts[index] += increment
            return

    ids.append(item_id)
    counts.append(increment)


def insert_descending_score(
    mut ids: List[Int], mut scores: List[ScoreScalar], item_id: Int, score: ScoreScalar, k: Int
):
    if k == 0:
        return

    var insert_at = 0
    while insert_at < len(scores) and scores[insert_at] >= score:
        insert_at += 1

    if insert_at >= k:
        if len(scores) < k:
            ids.append(item_id)
            scores.append(score)
        return

    if len(scores) < k:
        ids.append(item_id)
        scores.append(score)

    var current = len(scores) - 1
    while current > insert_at:
        ids[current] = ids[current - 1]
        scores[current] = scores[current - 1]
        current -= 1

    ids[insert_at] = item_id
    scores[insert_at] = score


def insert_ascending_metric(
    mut ids: List[Int], mut distances: List[MetricScalar], item_id: Int, distance: MetricScalar, k: Int
):
    if k == 0:
        return

    for index in range(len(ids)):
        if ids[index] == item_id:
            if distance >= distances[index]:
                return
            distances[index] = distance
            while index > 0 and distances[index] < distances[index - 1]:
                var swap_distance = distances[index - 1]
                distances[index - 1] = distances[index]
                distances[index] = swap_distance
                var swap_id = ids[index - 1]
                ids[index - 1] = ids[index]
                ids[index] = swap_id
            return

    var insert_at = 0
    while insert_at < len(distances) and distances[insert_at] <= distance:
        insert_at += 1

    if insert_at >= k:
        if len(distances) < k:
            ids.append(item_id)
            distances.append(distance)
        return

    if len(distances) < k:
        ids.append(item_id)
        distances.append(distance)

    var current = len(distances) - 1
    while current > insert_at:
        ids[current] = ids[current - 1]
        distances[current] = distances[current - 1]
        current -= 1

    ids[insert_at] = item_id
    distances[insert_at] = distance


struct QuantizedCodeHistogram(Copyable):
    var code_ids: List[Int]
    var counts: List[Int]

    def __init__(out self):
        self.code_ids = List[Int]()
        self.counts = List[Int]()

    def __init__(out self, var code_ids: List[Int], var counts: List[Int]) raises:
        if len(code_ids) != len(counts):
            raise Error("quantized code histogram ids and counts must match")
        for count in counts:
            if count <= 0:
                raise Error("quantized code histogram counts must be positive")
        self.code_ids = code_ids^
        self.counts = counts^


def build_document_quantized_histogram(
    read doc_code_ids: List[Int],
    read doc_code_counts: List[Int],
    start: Int,
    stop: Int,
) raises -> QuantizedCodeHistogram:
    var ids = List[Int]()
    var counts = List[Int]()
    for index in range(start, stop):
        append_or_increment_count(ids, counts, doc_code_ids[index], doc_code_counts[index])
    return QuantizedCodeHistogram(ids^, counts^)


def build_quantization_distance_matrix(
    read centroids: List[List[VectorScalar]]
) raises -> List[MetricScalar]:
    var distances = List[MetricScalar]()
    for left_index in range(len(centroids)):
        for right_index in range(len(centroids)):
            distances.append(
                squared_l2_distance(centroids[left_index], centroids[right_index])
            )
    return distances^


def centroid_distance_at(
    read matrix: List[MetricScalar], centroid_count: Int, left_index: Int, right_index: Int
) raises -> MetricScalar:
    if left_index < 0 or left_index >= centroid_count:
        raise Error("left centroid index is out of range")
    if right_index < 0 or right_index >= centroid_count:
        raise Error("right centroid index is out of range")
    return matrix[left_index * centroid_count + right_index]


def greedy_quantized_emd_distance(
    read left_histogram: QuantizedCodeHistogram,
    left_total_count: Int,
    read right_histogram: QuantizedCodeHistogram,
    right_total_count: Int,
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises -> MetricScalar:
    if left_total_count <= 0 or right_total_count <= 0:
        raise Error("qEMD requires positive token counts")

    var left_masses = List[MetricScalar]()
    var right_masses = List[MetricScalar]()
    for count in left_histogram.counts:
        left_masses.append(MetricScalar(count) / MetricScalar(left_total_count))
    for count in right_histogram.counts:
        right_masses.append(MetricScalar(count) / MetricScalar(right_total_count))

    var total = zero_metric_scalar()
    while True:
        var best_left = -1
        var best_right = -1
        var best_distance = max_metric_scalar()

        for left_index in range(len(left_histogram.code_ids)):
            if left_masses[left_index] <= zero_metric_scalar():
                continue
            for right_index in range(len(right_histogram.code_ids)):
                if right_masses[right_index] <= zero_metric_scalar():
                    continue
                var distance = centroid_distance_at(
                    distance_matrix,
                    centroid_count,
                    left_histogram.code_ids[left_index],
                    right_histogram.code_ids[right_index],
                )
                if distance < best_distance:
                    best_distance = distance
                    best_left = left_index
                    best_right = right_index

        if best_left < 0 or best_right < 0:
            break

        var flow = left_masses[best_left]
        if right_masses[best_right] < flow:
            flow = right_masses[best_right]

        total += flow * best_distance
        left_masses[best_left] -= flow
        right_masses[best_right] -= flow
    return total


def quantized_chamfer_distance_for_document(
    read query_codes: List[Int],
    read index: GemGraphIndex,
    document_index: Int,
    read distance_matrix: List[MetricScalar],
) raises -> MetricScalar:
    if document_index < 0 or document_index >= index.document_count:
        raise Error("gem graph document index is out of range")
    if len(query_codes) == 0:
        raise Error("qCH requires at least one query code")

    var start = index.doc_code_offsets[document_index]
    var stop = index.doc_code_offsets[document_index + 1]
    if start == stop:
        raise Error("qCH requires the document to have at least one code")

    var total = zero_metric_scalar()
    for query_code in query_codes:
        var best_distance = max_metric_scalar()
        for code_index in range(start, stop):
            var distance = centroid_distance_at(
                distance_matrix,
                index.quantization_centroid_count,
                query_code,
                index.doc_code_ids[code_index],
            )
            if distance < best_distance:
                best_distance = distance
        total += best_distance

    return total


def document_profile_has_cluster_arrays(
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    document_index: Int,
    cluster_index: Int,
) raises -> Bool:
    if document_index < 0 or document_index >= len(doc_profile_offsets) - 1:
        raise Error("document profile document index is out of range")
    for profile_index in range(
        doc_profile_offsets[document_index],
        doc_profile_offsets[document_index + 1],
    ):
        if doc_profile_cluster_ids[profile_index] == cluster_index:
            return True
    return False


def document_profile_has_cluster(
    read index: GemGraphIndex, document_index: Int, cluster_index: Int
) raises -> Bool:
    if document_index < 0 or document_index >= index.document_count:
        raise Error("document profile document index is out of range")
    return document_profile_has_cluster_arrays(
        index.doc_profile_offsets,
        index.doc_profile_cluster_ids,
        document_index,
        cluster_index,
    )


def document_profile_intersects_clusters(
    read index: GemGraphIndex, document_index: Int, read relevant_clusters: List[Int]
) raises -> Bool:
    if len(relevant_clusters) == 0:
        return False
    for relevant_cluster in relevant_clusters:
        if document_profile_has_cluster(index, document_index, relevant_cluster):
            return True
    return False


def quantize_query_codes(
    read query: EncodedQuery, read index: GemGraphIndex
) raises -> List[Int]:
    if index.quantization_centroid_count == 0:
        raise Error("cannot quantize query against an empty gem graph")

    var codes = List[Int]()
    for query_token in query.token_vectors:
        codes.append(
            nearest_centroid_index(query_token, index.quantization_centroids)
        )
    return codes^


def query_relevant_cluster_ids(
    read query: EncodedQuery,
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> List[Int]:
    if cluster_top_k_per_query_token <= 0:
        raise Error("cluster_top_k_per_query_token must be positive")

    var cluster_ids = List[Int]()
    for query_token in query.token_vectors:
        var top_ids = List[Int]()
        var top_scores = List[ScoreScalar]()
        for cluster_index in range(index.cluster_count):
            if index.cluster_offsets[cluster_index] == index.cluster_offsets[cluster_index + 1]:
                continue
            insert_descending_score(
                top_ids,
                top_scores,
                cluster_index,
                dot_product(query_token, index.index_centroids[cluster_index]),
                cluster_top_k_per_query_token,
            )
        for cluster_index in top_ids:
            var seen = False
            for existing in cluster_ids:
                if existing == cluster_index:
                    seen = True
                    break
            if not seen:
                cluster_ids.append(cluster_index)
    return cluster_ids^


def query_entry_doc_indices(
    read index: GemGraphIndex, read relevant_clusters: List[Int]
) raises -> List[Int]:
    var doc_indices = List[Int]()
    for cluster_index in relevant_clusters:
        var entry_doc = index.entry_doc_indices[cluster_index]
        if entry_doc == -1:
            continue
        var seen = False
        for existing in doc_indices:
            if existing == entry_doc:
                seen = True
                break
        if not seen:
            doc_indices.append(entry_doc)
    return doc_indices^


def profile_score_for_cluster(
    read index: GemGraphIndex, document_index: Int, cluster_index: Int
) raises -> ScoreScalar:
    return profile_score_for_cluster_arrays(
        index.doc_profile_offsets,
        index.doc_profile_cluster_ids,
        index.doc_profile_scores,
        document_index,
        cluster_index,
    )


def profile_score_for_cluster_arrays(
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    read doc_profile_scores: List[ScoreScalar],
    document_index: Int,
    cluster_index: Int,
) raises -> ScoreScalar:
    if document_index < 0 or document_index >= len(doc_profile_offsets) - 1:
        raise Error("profile score document index is out of range")
    for profile_index in range(
        doc_profile_offsets[document_index],
        doc_profile_offsets[document_index + 1],
    ):
        if doc_profile_cluster_ids[profile_index] == cluster_index:
            return doc_profile_scores[profile_index]
    return min_score_scalar()


def build_gem_graph_index(
    read packed_index: PackedIndex,
    fine_cluster_count: Int,
    coarse_cluster_count: Int,
    cluster_cutoff: Int,
    construction_neighbor_count: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    degree_limit: Int = DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
) raises -> GemGraphIndex:
    if packed_index.document_count <= 0:
        raise Error("gem graph requires at least one document")

    var effective_fine_cluster_count = choose_effective_cluster_count(
        fine_cluster_count, packed_index.total_vector_count
    )
    var quantization_centroids = build_kmeans_centroids(
        packed_index.token_vectors,
        effective_fine_cluster_count,
        DEFAULT_GEM_GRAPH_FINE_CLUSTER_REFINEMENT_STEPS,
    )
    var effective_coarse_cluster_count = choose_effective_cluster_count(
        coarse_cluster_count, len(quantization_centroids)
    )
    var index_centroids = build_kmeans_centroids(
        quantization_centroids,
        effective_coarse_cluster_count,
        DEFAULT_GEM_GRAPH_COARSE_CLUSTER_REFINEMENT_STEPS,
    )

    var quantization_to_index = List[Int]()
    for centroid in quantization_centroids:
        quantization_to_index.append(
            nearest_centroid_index(centroid, index_centroids)
        )

    var doc_code_offsets = [0]
    var doc_code_ids = List[Int]()
    var doc_code_counts = List[Int]()
    var doc_cluster_ids = List[List[Int]]()
    var doc_cluster_counts = List[List[Int]]()
    var document_frequencies = List[Int]()
    for _ in range(len(index_centroids)):
        document_frequencies.append(0)

    for document_index in range(packed_index.document_count):
        var doc_ids = List[Int]()
        var doc_counts = List[Int]()
        var cluster_ids = List[Int]()
        var cluster_counts = List[Int]()
        var start = packed_index.doc_offsets[document_index]
        var stop = packed_index.doc_offsets[document_index + 1]
        for token_index in range(start, stop):
            var code_id = nearest_centroid_index(
                packed_index.token_vectors[token_index], quantization_centroids
            )
            append_or_increment_count(doc_ids, doc_counts, code_id)
            append_or_increment_count(
                cluster_ids,
                cluster_counts,
                quantization_to_index[code_id],
            )
        for code_index in range(len(doc_ids)):
            doc_code_ids.append(doc_ids[code_index])
            doc_code_counts.append(doc_counts[code_index])
        doc_code_offsets.append(len(doc_code_ids))
        doc_cluster_ids.append(cluster_ids^)
        doc_cluster_counts.append(cluster_counts^)
        for cluster_id in doc_cluster_ids[document_index]:
            document_frequencies[cluster_id] += 1
    var doc_profile_offsets = [0]
    var doc_profile_cluster_ids = List[Int]()
    var doc_profile_scores = List[ScoreScalar]()
    var cluster_members = List[List[Int]]()
    for _ in range(len(index_centroids)):
        cluster_members.append(List[Int]())

    for document_index in range(packed_index.document_count):
        var profile_ids = List[Int]()
        var profile_scores = List[ScoreScalar]()
        var doc_unique_cluster_count = len(doc_cluster_ids[document_index])
        var effective_cluster_cutoff = doc_unique_cluster_count
        if cluster_cutoff > 0 and cluster_cutoff < effective_cluster_cutoff:
            effective_cluster_cutoff = cluster_cutoff
        for cluster_position in range(doc_unique_cluster_count):
            var coarse_cluster = doc_cluster_ids[document_index][cluster_position]
            var tf = doc_cluster_counts[document_index][cluster_position]
            var idf = MetricScalar(log2(
                MetricScalar(packed_index.document_count)
                / MetricScalar(1 + document_frequencies[coarse_cluster])
            ))
            insert_descending_score(
                profile_ids,
                profile_scores,
                coarse_cluster,
                ScoreScalar(MetricScalar(tf) * idf),
                effective_cluster_cutoff,
            )
        for profile_index in range(len(profile_ids)):
            doc_profile_cluster_ids.append(profile_ids[profile_index])
            doc_profile_scores.append(profile_scores[profile_index])
            cluster_members[profile_ids[profile_index]].append(document_index)
        doc_profile_offsets.append(len(doc_profile_cluster_ids))

    var cluster_offsets = [0]
    var cluster_doc_indices = List[Int]()
    for cluster_index in range(len(index_centroids)):
        for document_index in cluster_members[cluster_index]:
            cluster_doc_indices.append(document_index)
        cluster_offsets.append(len(cluster_doc_indices))

    var entry_doc_indices = List[Int]()
    for cluster_index in range(len(index_centroids)):
        if len(cluster_members[cluster_index]) == 0:
            entry_doc_indices.append(-1)
            continue

        var best_doc = cluster_members[cluster_index][0]
        var best_score = min_score_scalar()
        for document_index in cluster_members[cluster_index]:
            var score = profile_score_for_cluster_arrays(
                doc_profile_offsets,
                doc_profile_cluster_ids,
                doc_profile_scores,
                document_index,
                cluster_index,
            )
            if score > best_score:
                best_score = score
                best_doc = document_index
        entry_doc_indices.append(best_doc)

    var distance_matrix = build_quantization_distance_matrix(quantization_centroids)
    var candidate_neighbor_ids = List[List[Int]]()
    var candidate_neighbor_distances = List[List[MetricScalar]]()
    for _ in range(packed_index.document_count):
        candidate_neighbor_ids.append(List[Int]())
        candidate_neighbor_distances.append(List[MetricScalar]())

    for cluster_index in range(len(index_centroids)):
        var cluster_start = cluster_offsets[cluster_index]
        var cluster_stop = cluster_offsets[cluster_index + 1]
        for left_offset in range(cluster_start, cluster_stop):
            var left_doc = cluster_doc_indices[left_offset]
            var left_histogram = build_document_quantized_histogram(
                doc_code_ids,
                doc_code_counts,
                doc_code_offsets[left_doc],
                doc_code_offsets[left_doc + 1],
            )
            var local_ids = List[Int]()
            var local_distances = List[MetricScalar]()
            for right_offset in range(cluster_start, cluster_stop):
                var right_doc = cluster_doc_indices[right_offset]
                if right_doc == left_doc:
                    continue
                var right_histogram = build_document_quantized_histogram(
                    doc_code_ids,
                    doc_code_counts,
                    doc_code_offsets[right_doc],
                    doc_code_offsets[right_doc + 1],
                )
                insert_ascending_metric(
                    local_ids,
                    local_distances,
                    right_doc,
                    greedy_quantized_emd_distance(
                        left_histogram,
                        sum_ints(left_histogram.counts),
                        right_histogram,
                        sum_ints(right_histogram.counts),
                        distance_matrix,
                        len(quantization_centroids),
                    ),
                    construction_neighbor_count,
                )
            for neighbor_index in range(len(local_ids)):
                insert_ascending_metric(
                    candidate_neighbor_ids[left_doc],
                    candidate_neighbor_distances[left_doc],
                    local_ids[neighbor_index],
                    local_distances[neighbor_index],
                    degree_limit * 4,
                )
                insert_ascending_metric(
                    candidate_neighbor_ids[local_ids[neighbor_index]],
                    candidate_neighbor_distances[local_ids[neighbor_index]],
                    left_doc,
                    local_distances[neighbor_index],
                    degree_limit * 4,
                )

    var neighbor_offsets = [0]
    var neighbor_doc_indices = List[Int]()
    for document_index in range(packed_index.document_count):
        var final_ids = List[Int]()
        var final_distances = List[MetricScalar]()
        for candidate_index in range(len(candidate_neighbor_ids[document_index])):
            insert_ascending_metric(
                final_ids,
                final_distances,
                candidate_neighbor_ids[document_index][candidate_index],
                candidate_neighbor_distances[document_index][candidate_index],
                degree_limit,
            )
        for profile_index in range(
            doc_profile_offsets[document_index],
            doc_profile_offsets[document_index + 1],
        ):
            var coarse_cluster = doc_profile_cluster_ids[profile_index]
            var covered = False
            for neighbor_doc in final_ids:
                if document_profile_has_cluster_arrays(
                    doc_profile_offsets,
                    doc_profile_cluster_ids,
                    neighbor_doc,
                    coarse_cluster,
                ):
                    covered = True
                    break
            if covered:
                continue
            for candidate_index in range(len(candidate_neighbor_ids[document_index])):
                var candidate_doc = candidate_neighbor_ids[document_index][candidate_index]
                if not document_profile_has_cluster_arrays(
                    doc_profile_offsets,
                    doc_profile_cluster_ids,
                    candidate_doc,
                    coarse_cluster,
                ):
                    continue
                insert_ascending_metric(
                    final_ids,
                    final_distances,
                    candidate_doc,
                    candidate_neighbor_distances[document_index][candidate_index],
                    degree_limit,
                )
                break
        for neighbor_doc in final_ids:
            neighbor_doc_indices.append(neighbor_doc)
        neighbor_offsets.append(len(neighbor_doc_indices))

    return GemGraphIndex(
        packed_index.doc_ids.copy(),
        doc_code_offsets^,
        doc_code_ids^,
        doc_code_counts^,
        quantization_centroids^,
        index_centroids^,
        quantization_to_index^,
        doc_profile_offsets^,
        doc_profile_cluster_ids^,
        doc_profile_scores^,
        cluster_offsets^,
        cluster_doc_indices^,
        entry_doc_indices^,
        neighbor_offsets^,
        neighbor_doc_indices^,
        packed_index.vector_dim,
        0,
        cluster_cutoff,
        construction_neighbor_count,
        degree_limit,
    )
