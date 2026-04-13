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
# - qCH and qEMD operate on quantized centroid similarities using `1 - dot`
# - the graph is stored explicitly even though the public GEM reference uses
#   HNSW internals; distance, bridge, and shortcut behavior should still mirror
#   the GEM design

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

from .gem_cutoff_tree import build_adaptive_cutoff_decision_tree
from .gem_transport import min_cost_transport_distance
from .packed_index import PackedIndex


comptime DEFAULT_GEM_GRAPH_FINE_CLUSTER_REFINEMENT_STEPS = 4
comptime DEFAULT_GEM_GRAPH_COARSE_CLUSTER_REFINEMENT_STEPS = 4
comptime DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT = 6
comptime DEFAULT_GEM_GRAPH_DEGREE_LIMIT = 8
comptime DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K = 2
comptime DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH = 32
comptime DEFAULT_GEM_GRAPH_ADAPTIVE_CUTOFF_MAX = 10
comptime DEFAULT_GEM_GRAPH_ADAPTIVE_TREE_MAX_DEPTH = 3


struct GemGraphTrainingPair(Copyable):
    var query: EncodedQuery
    var positive_doc_id: String

    def __init__(
        out self, var query: EncodedQuery, var positive_doc_id: String
    ) raises:
        if positive_doc_id.byte_length() == 0:
            raise Error("gem graph training pair positive_doc_id must not be empty")
        self.query = query^
        self.positive_doc_id = positive_doc_id^


struct GemGraphBuildConfig(Copyable):
    var fine_cluster_count: Int
    var coarse_cluster_count: Int
    var cluster_cutoff: Int
    var construction_neighbor_count: Int
    var degree_limit: Int
    var enable_adaptive_cluster_cutoff: Bool
    var adaptive_cluster_cutoff_max: Int
    var adaptive_tree_max_depth: Int
    var cluster_top_k_per_query_token: Int
    var enable_shortcuts: Bool
    var shortcut_candidate_k: Int
    var shortcut_beam_width: Int
    var training_pairs: List[GemGraphTrainingPair]

    def __init__(
        out self,
        fine_cluster_count: Int,
        coarse_cluster_count: Int,
        cluster_cutoff: Int,
        construction_neighbor_count: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
        degree_limit: Int = DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
        enable_adaptive_cluster_cutoff: Bool = False,
        adaptive_cluster_cutoff_max: Int = DEFAULT_GEM_GRAPH_ADAPTIVE_CUTOFF_MAX,
        adaptive_tree_max_depth: Int = DEFAULT_GEM_GRAPH_ADAPTIVE_TREE_MAX_DEPTH,
        cluster_top_k_per_query_token: Int = DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
        enable_shortcuts: Bool = False,
        shortcut_candidate_k: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
        shortcut_beam_width: Int = DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
        var training_pairs: List[GemGraphTrainingPair] = List[GemGraphTrainingPair](),
    ) raises:
        if fine_cluster_count < 0:
            raise Error("gem graph fine_cluster_count must be non-negative")
        if coarse_cluster_count < 0:
            raise Error("gem graph coarse_cluster_count must be non-negative")
        if cluster_cutoff < 0:
            raise Error("gem graph cluster_cutoff must be non-negative")
        if construction_neighbor_count <= 0:
            raise Error(
                "gem graph construction_neighbor_count must be positive"
            )
        if degree_limit <= 0:
            raise Error("gem graph degree_limit must be positive")
        if adaptive_cluster_cutoff_max <= 0:
            raise Error(
                "gem graph adaptive_cluster_cutoff_max must be positive"
            )
        if adaptive_tree_max_depth < 0:
            raise Error("gem graph adaptive_tree_max_depth must be non-negative")
        if cluster_top_k_per_query_token <= 0:
            raise Error(
                "gem graph cluster_top_k_per_query_token must be positive"
            )
        if shortcut_candidate_k <= 0:
            raise Error("gem graph shortcut_candidate_k must be positive")
        if shortcut_beam_width <= 0:
            raise Error("gem graph shortcut_beam_width must be positive")

        self.fine_cluster_count = fine_cluster_count
        self.coarse_cluster_count = coarse_cluster_count
        self.cluster_cutoff = cluster_cutoff
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit
        self.enable_adaptive_cluster_cutoff = enable_adaptive_cluster_cutoff
        self.adaptive_cluster_cutoff_max = adaptive_cluster_cutoff_max
        self.adaptive_tree_max_depth = adaptive_tree_max_depth
        self.cluster_top_k_per_query_token = cluster_top_k_per_query_token
        self.enable_shortcuts = enable_shortcuts
        self.shortcut_candidate_k = shortcut_candidate_k
        self.shortcut_beam_width = shortcut_beam_width
        self.training_pairs = training_pairs^


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
    var adaptive_cluster_cutoff_enabled: Bool
    var adaptive_cluster_cutoff_max: Int
    var construction_neighbor_count: Int
    var degree_limit: Int
    var shortcuts_enabled: Bool

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
        self.adaptive_cluster_cutoff_enabled = False
        self.adaptive_cluster_cutoff_max = 0
        self.construction_neighbor_count = 0
        self.degree_limit = 0
        self.shortcuts_enabled = False

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
        adaptive_cluster_cutoff_enabled: Bool,
        adaptive_cluster_cutoff_max: Int,
        construction_neighbor_count: Int,
        degree_limit: Int,
        shortcuts_enabled: Bool,
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
            adaptive_cluster_cutoff_enabled,
            adaptive_cluster_cutoff_max,
            construction_neighbor_count,
            degree_limit,
            shortcuts_enabled,
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
        self.adaptive_cluster_cutoff_enabled = adaptive_cluster_cutoff_enabled
        self.adaptive_cluster_cutoff_max = adaptive_cluster_cutoff_max
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit
        self.shortcuts_enabled = shortcuts_enabled


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
    adaptive_cluster_cutoff_enabled: Bool,
    adaptive_cluster_cutoff_max: Int,
    construction_neighbor_count: Int,
    degree_limit: Int,
    shortcuts_enabled: Bool,
) raises:
    if shortcut_edge_count < 0:
        raise Error("gem graph shortcut_edge_count must be non-negative")
    if cluster_cutoff < 0:
        raise Error("gem graph cluster_cutoff must be non-negative")
    if adaptive_cluster_cutoff_max < 0:
        raise Error("gem graph adaptive_cluster_cutoff_max must be non-negative")
    if adaptive_cluster_cutoff_enabled and adaptive_cluster_cutoff_max <= 0:
        raise Error(
            "gem graph adaptive_cluster_cutoff_max must be positive when adaptive cutoff is enabled"
        )
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
                MetricScalar(1.0)
                - MetricScalar(
                    dot_product(centroids[left_index], centroids[right_index])
                )
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


def exact_quantized_emd_distance(
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
    var pair_costs = List[MetricScalar]()
    for count in left_histogram.counts:
        left_masses.append(MetricScalar(count) / MetricScalar(left_total_count))
    for count in right_histogram.counts:
        right_masses.append(MetricScalar(count) / MetricScalar(right_total_count))
    for left_index in range(len(left_histogram.code_ids)):
        for right_index in range(len(right_histogram.code_ids)):
            pair_costs.append(
                centroid_distance_at(
                    distance_matrix,
                    centroid_count,
                    left_histogram.code_ids[left_index],
                    right_histogram.code_ids[right_index],
                )
            )
    return min_cost_transport_distance(left_masses, right_masses, pair_costs)


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

    return total / MetricScalar(len(query_codes))


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


def build_gem_graph_index_legacy(
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
        False,
        0,
        construction_neighbor_count,
        degree_limit,
        False,
    )


def list_contains_int(read values: List[Int], target: Int) -> Bool:
    for value in values:
        if value == target:
            return True
    return False


def insert_unique_int(mut values: List[Int], item: Int):
    if not list_contains_int(values, item):
        values.append(item)


def remove_front(
    mut ids: List[Int], mut distances: List[MetricScalar]
) raises -> Int:
    if len(ids) == 0:
        raise Error("cannot remove from an empty queue")

    var head_id = ids[0]
    for index in range(1, len(ids)):
        ids[index - 1] = ids[index]
        distances[index - 1] = distances[index]
    _ = ids.pop()
    _ = distances.pop()
    return head_id


struct FlattenedNeighborLists(Copyable):
    var offsets: List[Int]
    var doc_indices: List[Int]

    def __init__(out self):
        self.offsets = [0]
        self.doc_indices = List[Int]()

    def __init__(out self, var offsets: List[Int], var doc_indices: List[Int]):
        self.offsets = offsets^
        self.doc_indices = doc_indices^


def flatten_neighbor_lists(
    read neighbor_ids_by_doc: List[List[Int]]
) -> FlattenedNeighborLists:
    var offsets = [0]
    var doc_indices = List[Int]()
    for neighbors in neighbor_ids_by_doc:
        for neighbor_doc in neighbors:
            doc_indices.append(neighbor_doc)
        offsets.append(len(doc_indices))
    return FlattenedNeighborLists(offsets^, doc_indices^)


def qemd_distance_between_documents(
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    left_doc: Int,
    right_doc: Int,
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises -> MetricScalar:
    return exact_quantized_emd_distance(
        histograms_by_doc[left_doc],
        histogram_totals_by_doc[left_doc],
        histograms_by_doc[right_doc],
        histogram_totals_by_doc[right_doc],
        distance_matrix,
        centroid_count,
    )


def choose_profile_limit_for_document(
    doc_unique_cluster_count: Int, max_profile_limit: Int
) -> Int:
    var limit = doc_unique_cluster_count
    if max_profile_limit > 0 and max_profile_limit < limit:
        limit = max_profile_limit
    return limit


def query_relevant_cluster_ids_for_centroids(
    read query: EncodedQuery,
    read index_centroids: List[List[VectorScalar]],
    cluster_top_k_per_query_token: Int,
) raises -> List[Int]:
    if cluster_top_k_per_query_token <= 0:
        raise Error("cluster_top_k_per_query_token must be positive")

    var cluster_ids = List[Int]()
    for query_token in query.token_vectors:
        var top_ids = List[Int]()
        var top_scores = List[ScoreScalar]()
        for cluster_index in range(len(index_centroids)):
            insert_descending_score(
                top_ids,
                top_scores,
                cluster_index,
                dot_product(query_token, index_centroids[cluster_index]),
                cluster_top_k_per_query_token,
            )
        for cluster_index in top_ids:
            insert_unique_int(cluster_ids, cluster_index)
    return cluster_ids^


def adaptive_profile_feature_row(
    read profile_scores: List[ScoreScalar],
    max_profile_limit: Int,
    vector_count: Int,
) raises -> List[MetricScalar]:
    if max_profile_limit <= 0:
        raise Error("adaptive profile max_profile_limit must be positive")
    if vector_count <= 0:
        raise Error("adaptive profile vector_count must be positive")

    var features = List[MetricScalar]()
    for score_index in range(max_profile_limit):
        if score_index < len(profile_scores):
            features.append(MetricScalar(profile_scores[score_index]))
        else:
            features.append(zero_metric_scalar())
    features.append(MetricScalar(vector_count))
    return features^


def adaptive_profile_label(
    read query: EncodedQuery,
    read index_centroids: List[List[VectorScalar]],
    read document_profile_cluster_ids: List[Int],
    cluster_top_k_per_query_token: Int,
    max_profile_limit: Int,
) raises -> Int:
    var relevant_clusters = query_relevant_cluster_ids_for_centroids(
        query, index_centroids, cluster_top_k_per_query_token
    )
    var stop = len(document_profile_cluster_ids)
    if stop > max_profile_limit:
        stop = max_profile_limit
    for rank in range(stop):
        if list_contains_int(relevant_clusters, document_profile_cluster_ids[rank]):
            return rank + 1
    return max_profile_limit


def find_document_index_by_id(read doc_ids: List[String], doc_id: String) raises -> Int:
    for document_index in range(len(doc_ids)):
        if doc_ids[document_index] == doc_id:
            return document_index
    raise Error("training pair positive_doc_id was not found in the packed index")


def document_vector_count(read packed_index: PackedIndex, document_index: Int) -> Int:
    return (
        packed_index.doc_offsets[document_index + 1]
        - packed_index.doc_offsets[document_index]
    )


def build_adaptive_profile_limits(
    read packed_index: PackedIndex,
    read raw_profile_ids_by_doc: List[List[Int]],
    read raw_profile_scores_by_doc: List[List[ScoreScalar]],
    read index_centroids: List[List[VectorScalar]],
    read config: GemGraphBuildConfig,
) raises -> List[Int]:
    if not config.enable_adaptive_cluster_cutoff:
        raise Error("adaptive profile limits require adaptive cutoff to be enabled")
    if len(config.training_pairs) == 0:
        raise Error(
            "adaptive cluster cutoff requires training_pairs because the paper-defined labels are supervised"
        )

    var feature_rows = List[List[MetricScalar]]()
    var labels = List[Int]()
    for training_pair in config.training_pairs:
        var document_index = find_document_index_by_id(
            packed_index.doc_ids, training_pair.positive_doc_id
        )
        feature_rows.append(
            adaptive_profile_feature_row(
                raw_profile_scores_by_doc[document_index],
                config.adaptive_cluster_cutoff_max,
                document_vector_count(packed_index, document_index),
            )
        )
        labels.append(
            adaptive_profile_label(
                training_pair.query,
                index_centroids,
                raw_profile_ids_by_doc[document_index],
                config.cluster_top_k_per_query_token,
                config.adaptive_cluster_cutoff_max,
            )
        )

    var decision_tree = build_adaptive_cutoff_decision_tree(
        feature_rows,
        labels,
        config.adaptive_tree_max_depth,
    )
    var predicted_limits = List[Int]()
    for document_index in range(packed_index.document_count):
        var unique_cluster_count = len(raw_profile_ids_by_doc[document_index])
        if unique_cluster_count == 0:
            predicted_limits.append(0)
            continue
        var predicted_limit = decision_tree.predict_label(
            adaptive_profile_feature_row(
                raw_profile_scores_by_doc[document_index],
                config.adaptive_cluster_cutoff_max,
                document_vector_count(packed_index, document_index),
            )
        )
        if predicted_limit < 1:
            predicted_limit = 1
        if predicted_limit > config.adaptive_cluster_cutoff_max:
            predicted_limit = config.adaptive_cluster_cutoff_max
        if predicted_limit > unique_cluster_count:
            predicted_limit = unique_cluster_count
        predicted_limits.append(predicted_limit)
    return predicted_limits^


def approximate_cluster_neighbor_candidates(
    document_index: Int,
    entry_doc: Int,
    read cluster_inserted_docs: List[Int],
    read neighbor_ids_by_doc: List[List[Int]],
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
    candidate_count: Int,
) raises -> List[Int]:
    if candidate_count <= 0 or len(cluster_inserted_docs) == 0:
        return List[Int]()
    if not list_contains_int(cluster_inserted_docs, entry_doc):
        raise Error("cluster entry_doc must already be present in the cluster graph")

    var queue_ids = List[Int]()
    var queue_distances = List[MetricScalar]()
    var result_ids = List[Int]()
    var result_distances = List[MetricScalar]()
    var visited = List[Int]()

    var entry_distance = qemd_distance_between_documents(
        histograms_by_doc,
        histogram_totals_by_doc,
        document_index,
        entry_doc,
        distance_matrix,
        centroid_count,
    )
    visited.append(entry_doc)
    insert_ascending_metric(
        queue_ids, queue_distances, entry_doc, entry_distance, candidate_count
    )
    insert_ascending_metric(
        result_ids, result_distances, entry_doc, entry_distance, candidate_count
    )

    while len(queue_ids) > 0:
        var current_distance = queue_distances[0]
        var tau = max_metric_scalar()
        if len(result_distances) >= candidate_count:
            tau = result_distances[len(result_distances) - 1]
        if len(result_distances) >= candidate_count and current_distance > tau:
            break

        var current_doc = remove_front(queue_ids, queue_distances)
        for neighbor_doc in neighbor_ids_by_doc[current_doc]:
            if not list_contains_int(cluster_inserted_docs, neighbor_doc):
                continue
            if list_contains_int(visited, neighbor_doc):
                continue
            visited.append(neighbor_doc)
            var neighbor_distance = qemd_distance_between_documents(
                histograms_by_doc,
                histogram_totals_by_doc,
                document_index,
                neighbor_doc,
                distance_matrix,
                centroid_count,
            )
            insert_ascending_metric(
                queue_ids,
                queue_distances,
                neighbor_doc,
                neighbor_distance,
                candidate_count,
            )
            insert_ascending_metric(
                result_ids,
                result_distances,
                neighbor_doc,
                neighbor_distance,
                candidate_count,
            )

    return result_ids^


def select_neighbors_by_cluster_heuristic(
    document_index: Int,
    read candidate_ids: List[Int],
    degree_limit: Int,
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises -> List[Int]:
    if degree_limit <= 0:
        raise Error("cluster heuristic degree_limit must be positive")

    var ranked_ids = List[Int]()
    var ranked_distances = List[MetricScalar]()
    for candidate_doc in candidate_ids:
        if candidate_doc == document_index:
            continue
        insert_ascending_metric(
            ranked_ids,
            ranked_distances,
            candidate_doc,
            qemd_distance_between_documents(
                histograms_by_doc,
                histogram_totals_by_doc,
                document_index,
                candidate_doc,
                distance_matrix,
                centroid_count,
            ),
            len(candidate_ids),
        )

    if len(ranked_ids) <= degree_limit:
        return ranked_ids^

    var selected_ids = List[Int]()
    for ranked_index in range(len(ranked_ids)):
        if len(selected_ids) >= degree_limit:
            break
        var candidate_doc = ranked_ids[ranked_index]
        var candidate_distance = ranked_distances[ranked_index]
        var keep_candidate = True
        for selected_doc in selected_ids:
            if (
                qemd_distance_between_documents(
                    histograms_by_doc,
                    histogram_totals_by_doc,
                    selected_doc,
                    candidate_doc,
                    distance_matrix,
                    centroid_count,
                )
                < candidate_distance
            ):
                keep_candidate = False
                break
        if keep_candidate:
            selected_ids.append(candidate_doc)
    return selected_ids^


def insert_neighbor_with_degree_limit(
    mut neighbor_ids_by_doc: List[List[Int]],
    document_index: Int,
    neighbor_doc: Int,
    degree_limit: Int,
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises:
    if document_index == neighbor_doc:
        return

    var candidate_ids = List[Int]()
    for existing_neighbor in neighbor_ids_by_doc[document_index]:
        insert_unique_int(candidate_ids, existing_neighbor)
    insert_unique_int(candidate_ids, neighbor_doc)
    neighbor_ids_by_doc[document_index] = select_neighbors_by_cluster_heuristic(
        document_index,
        candidate_ids,
        degree_limit,
        histograms_by_doc,
        histogram_totals_by_doc,
        distance_matrix,
        centroid_count,
    )


def add_mutual_connection(
    mut neighbor_ids_by_doc: List[List[Int]],
    left_doc: Int,
    right_doc: Int,
    degree_limit: Int,
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises:
    insert_neighbor_with_degree_limit(
        neighbor_ids_by_doc,
        left_doc,
        right_doc,
        degree_limit,
        histograms_by_doc,
        histogram_totals_by_doc,
        distance_matrix,
        centroid_count,
    )
    insert_neighbor_with_degree_limit(
        neighbor_ids_by_doc,
        right_doc,
        left_doc,
        degree_limit,
        histograms_by_doc,
        histogram_totals_by_doc,
        distance_matrix,
        centroid_count,
    )


def merge_bridge_neighbors(
    mut neighbor_ids_by_doc: List[List[Int]],
    document_index: Int,
    read new_neighbors: List[Int],
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    degree_limit: Int,
    read histograms_by_doc: List[QuantizedCodeHistogram],
    read histogram_totals_by_doc: List[Int],
    read distance_matrix: List[MetricScalar],
    centroid_count: Int,
) raises:
    var combined_ids = List[Int]()
    for neighbor_doc in neighbor_ids_by_doc[document_index]:
        insert_unique_int(combined_ids, neighbor_doc)
    for neighbor_doc in new_neighbors:
        insert_unique_int(combined_ids, neighbor_doc)

    var selected_ids = List[Int]()
    for profile_index in range(
        doc_profile_offsets[document_index],
        doc_profile_offsets[document_index + 1],
    ):
        if len(selected_ids) >= degree_limit:
            break
        var coarse_cluster = doc_profile_cluster_ids[profile_index]
        var best_neighbor = -1
        var best_distance = max_metric_scalar()
        for candidate_doc in combined_ids:
            if not document_profile_has_cluster_arrays(
                doc_profile_offsets,
                doc_profile_cluster_ids,
                candidate_doc,
                coarse_cluster,
            ):
                continue
            var distance = qemd_distance_between_documents(
                histograms_by_doc,
                histogram_totals_by_doc,
                document_index,
                candidate_doc,
                distance_matrix,
                centroid_count,
            )
            if distance < best_distance:
                best_distance = distance
                best_neighbor = candidate_doc
        if best_neighbor != -1:
            insert_unique_int(selected_ids, best_neighbor)

    var ranked_ids = List[Int]()
    var ranked_distances = List[MetricScalar]()
    for candidate_doc in combined_ids:
        insert_ascending_metric(
            ranked_ids,
            ranked_distances,
            candidate_doc,
            qemd_distance_between_documents(
                histograms_by_doc,
                histogram_totals_by_doc,
                document_index,
                candidate_doc,
                distance_matrix,
                centroid_count,
            ),
            len(combined_ids),
        )
    for candidate_doc in ranked_ids:
        if len(selected_ids) >= degree_limit:
            break
        insert_unique_int(selected_ids, candidate_doc)

    var heuristic_selected_ids = select_neighbors_by_cluster_heuristic(
        document_index,
        selected_ids,
        degree_limit,
        histograms_by_doc,
        histogram_totals_by_doc,
        distance_matrix,
        centroid_count,
    )
    var final_selected_ids = heuristic_selected_ids.copy()
    neighbor_ids_by_doc[document_index] = heuristic_selected_ids^
    for neighbor_doc in final_selected_ids:
        insert_neighbor_with_degree_limit(
            neighbor_ids_by_doc,
            neighbor_doc,
            document_index,
            degree_limit,
            histograms_by_doc,
            histogram_totals_by_doc,
            distance_matrix,
            centroid_count,
        )


def graph_candidate_doc_indices_for_query(
    read query: EncodedQuery,
    read index: GemGraphIndex,
    read neighbor_ids_by_doc: List[List[Int]],
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
) raises -> List[Int]:
    if candidate_k <= 0:
        return List[Int]()

    var relevant_clusters = query_relevant_cluster_ids(
        query,
        index,
        cluster_top_k_per_query_token,
    )
    if len(relevant_clusters) == 0:
        return List[Int]()

    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    if len(entry_docs) == 0:
        return List[Int]()

    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var query_codes = quantize_query_codes(query, index)

    var result_doc_indices = List[Int]()
    var result_distances = List[MetricScalar]()
    var visited = List[Int]()
    var queue_doc_indices = List[List[Int]]()
    var queue_distances = List[List[MetricScalar]]()

    var effective_beam_width = beam_width
    if effective_beam_width < candidate_k:
        effective_beam_width = candidate_k

    for entry_doc in entry_docs:
        var entry_distance = quantized_chamfer_distance_for_document(
            query_codes,
            index,
            entry_doc,
            distance_matrix,
        )
        visited.append(entry_doc)
        insert_ascending_metric(
            result_doc_indices,
            result_distances,
            entry_doc,
            entry_distance,
            effective_beam_width,
        )
        var local_ids = List[Int]()
        var local_distances = List[MetricScalar]()
        insert_ascending_metric(
            local_ids,
            local_distances,
            entry_doc,
            entry_distance,
            effective_beam_width,
        )
        queue_doc_indices.append(local_ids^)
        queue_distances.append(local_distances^)

    while True:
        var any_non_empty = False
        for queue_index in range(len(queue_doc_indices)):
            if len(queue_doc_indices[queue_index]) == 0:
                continue

            any_non_empty = True
            var current_distance = queue_distances[queue_index][0]
            var tau = max_metric_scalar()
            if len(result_distances) >= effective_beam_width:
                tau = result_distances[len(result_distances) - 1]

            if (
                len(result_distances) >= effective_beam_width
                and current_distance > tau
            ):
                queue_doc_indices[queue_index] = List[Int]()
                queue_distances[queue_index] = List[MetricScalar]()
                continue

            var current_doc = remove_front(
                queue_doc_indices[queue_index],
                queue_distances[queue_index],
            )
            for neighbor_doc in neighbor_ids_by_doc[current_doc]:
                if list_contains_int(visited, neighbor_doc):
                    continue
                if not document_profile_intersects_clusters(
                    index, neighbor_doc, relevant_clusters
                ):
                    continue

                var neighbor_distance = quantized_chamfer_distance_for_document(
                    query_codes,
                    index,
                    neighbor_doc,
                    distance_matrix,
                )
                visited.append(neighbor_doc)
                insert_ascending_metric(
                    queue_doc_indices[queue_index],
                    queue_distances[queue_index],
                    neighbor_doc,
                    neighbor_distance,
                    effective_beam_width,
                )
                insert_ascending_metric(
                    result_doc_indices,
                    result_distances,
                    neighbor_doc,
                    neighbor_distance,
                    effective_beam_width,
                )
        if not any_non_empty:
            break

    var top_doc_indices = List[Int]()
    var limit = candidate_k
    if len(result_doc_indices) < limit:
        limit = len(result_doc_indices)
    for result_index in range(limit):
        top_doc_indices.append(result_doc_indices[result_index])
    return top_doc_indices^


def inject_shortcuts(
    read index: GemGraphIndex,
    mut neighbor_ids_by_doc: List[List[Int]],
    read config: GemGraphBuildConfig,
) raises -> Int:
    var injected_shortcut_count = 0
    for training_pair in config.training_pairs:
        var positive_doc = find_document_index_by_id(
            index.doc_ids, training_pair.positive_doc_id
        )
        var candidate_doc_indices = graph_candidate_doc_indices_for_query(
            training_pair.query,
            index,
            neighbor_ids_by_doc,
            config.shortcut_candidate_k,
            config.cluster_top_k_per_query_token,
            config.shortcut_beam_width,
        )
        if len(candidate_doc_indices) == 0:
            continue
        if list_contains_int(candidate_doc_indices, positive_doc):
            continue

        var top_doc = candidate_doc_indices[0]
        if top_doc == positive_doc:
            continue
        if len(neighbor_ids_by_doc[top_doc]) >= config.degree_limit:
            continue
        if len(neighbor_ids_by_doc[positive_doc]) >= config.degree_limit:
            continue
        if list_contains_int(neighbor_ids_by_doc[top_doc], positive_doc):
            continue

        neighbor_ids_by_doc[top_doc].append(positive_doc)
        neighbor_ids_by_doc[positive_doc].append(top_doc)
        injected_shortcut_count += 1
    return injected_shortcut_count


def build_gem_graph_index_with_config(
    read packed_index: PackedIndex,
    read config: GemGraphBuildConfig,
) raises -> GemGraphIndex:
    if packed_index.document_count <= 0:
        raise Error("gem graph requires at least one document")

    var effective_fine_cluster_count = choose_effective_cluster_count(
        config.fine_cluster_count, packed_index.total_vector_count
    )
    var quantization_centroids = build_kmeans_centroids(
        packed_index.token_vectors,
        effective_fine_cluster_count,
        DEFAULT_GEM_GRAPH_FINE_CLUSTER_REFINEMENT_STEPS,
    )
    var effective_coarse_cluster_count = choose_effective_cluster_count(
        config.coarse_cluster_count, len(quantization_centroids)
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
        var histogram_code_ids = List[Int]()
        var histogram_code_counts = List[Int]()
        var cluster_ids = List[Int]()
        var cluster_counts = List[Int]()
        var start = packed_index.doc_offsets[document_index]
        var stop = packed_index.doc_offsets[document_index + 1]
        for token_index in range(start, stop):
            var code_id = nearest_centroid_index(
                packed_index.token_vectors[token_index], quantization_centroids
            )
            append_or_increment_count(
                histogram_code_ids, histogram_code_counts, code_id
            )
            append_or_increment_count(
                cluster_ids,
                cluster_counts,
                quantization_to_index[code_id],
            )
        for code_index in range(len(histogram_code_ids)):
            doc_code_ids.append(histogram_code_ids[code_index])
            doc_code_counts.append(histogram_code_counts[code_index])
        doc_code_offsets.append(len(doc_code_ids))
        doc_cluster_ids.append(cluster_ids^)
        doc_cluster_counts.append(cluster_counts^)
        for cluster_id in doc_cluster_ids[document_index]:
            document_frequencies[cluster_id] += 1

    var raw_profile_ids_by_doc = List[List[Int]]()
    var raw_profile_scores_by_doc = List[List[ScoreScalar]]()
    var raw_profile_limit = config.cluster_cutoff
    if (
        config.enable_adaptive_cluster_cutoff
        and config.adaptive_cluster_cutoff_max > raw_profile_limit
    ):
        raw_profile_limit = config.adaptive_cluster_cutoff_max

    for document_index in range(packed_index.document_count):
        var profile_ids = List[Int]()
        var profile_scores = List[ScoreScalar]()
        var profile_limit = choose_profile_limit_for_document(
            len(doc_cluster_ids[document_index]), raw_profile_limit
        )
        for cluster_position in range(len(doc_cluster_ids[document_index])):
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
                profile_limit,
            )
        raw_profile_ids_by_doc.append(profile_ids^)
        raw_profile_scores_by_doc.append(profile_scores^)

    var profile_limits_by_doc = List[Int]()
    if config.enable_adaptive_cluster_cutoff:
        profile_limits_by_doc = build_adaptive_profile_limits(
            packed_index,
            raw_profile_ids_by_doc,
            raw_profile_scores_by_doc,
            index_centroids,
            config,
        )
    else:
        for document_index in range(packed_index.document_count):
            profile_limits_by_doc.append(
                choose_profile_limit_for_document(
                    len(raw_profile_ids_by_doc[document_index]),
                    config.cluster_cutoff,
                )
            )

    var doc_profile_offsets = [0]
    var doc_profile_cluster_ids = List[Int]()
    var doc_profile_scores = List[ScoreScalar]()
    var cluster_members = List[List[Int]]()
    for _ in range(len(index_centroids)):
        cluster_members.append(List[Int]())

    for document_index in range(packed_index.document_count):
        var effective_cluster_cutoff = profile_limits_by_doc[document_index]
        for profile_index in range(effective_cluster_cutoff):
            doc_profile_cluster_ids.append(
                raw_profile_ids_by_doc[document_index][profile_index]
            )
            doc_profile_scores.append(
                raw_profile_scores_by_doc[document_index][profile_index]
            )
            cluster_members[
                raw_profile_ids_by_doc[document_index][profile_index]
            ].append(document_index)
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
        entry_doc_indices.append(cluster_members[cluster_index][0])

    var histograms_by_doc = List[QuantizedCodeHistogram]()
    var histogram_totals_by_doc = List[Int]()
    for document_index in range(packed_index.document_count):
        var histogram = build_document_quantized_histogram(
            doc_code_ids,
            doc_code_counts,
            doc_code_offsets[document_index],
            doc_code_offsets[document_index + 1],
        )
        histogram_totals_by_doc.append(sum_ints(histogram.counts))
        histograms_by_doc.append(histogram^)

    var distance_matrix = build_quantization_distance_matrix(quantization_centroids)
    var neighbor_ids_by_doc = List[List[Int]]()
    var inserted_globally = List[Bool]()
    for _ in range(packed_index.document_count):
        neighbor_ids_by_doc.append(List[Int]())
        inserted_globally.append(False)

    for cluster_index in range(len(index_centroids)):
        if len(cluster_members[cluster_index]) == 0:
            continue
        var entry_doc = entry_doc_indices[cluster_index]
        var cluster_inserted_docs = List[Int]()
        for cluster_doc in cluster_members[cluster_index]:
            if len(cluster_inserted_docs) == 0:
                inserted_globally[cluster_doc] = True
                cluster_inserted_docs.append(cluster_doc)
                continue

            var new_neighbors = approximate_cluster_neighbor_candidates(
                cluster_doc,
                entry_doc,
                cluster_inserted_docs,
                neighbor_ids_by_doc,
                histograms_by_doc,
                histogram_totals_by_doc,
                distance_matrix,
                len(quantization_centroids),
                config.construction_neighbor_count,
            )
            if len(new_neighbors) == 0:
                new_neighbors.append(entry_doc)

            var selected_new_neighbors = select_neighbors_by_cluster_heuristic(
                cluster_doc,
                new_neighbors,
                config.degree_limit,
                histograms_by_doc,
                histogram_totals_by_doc,
                distance_matrix,
                len(quantization_centroids),
            )
            if len(selected_new_neighbors) == 0:
                selected_new_neighbors.append(new_neighbors[0])

            if not inserted_globally[cluster_doc]:
                inserted_globally[cluster_doc] = True
                for neighbor_doc in selected_new_neighbors:
                    add_mutual_connection(
                        neighbor_ids_by_doc,
                        cluster_doc,
                        neighbor_doc,
                        config.degree_limit,
                        histograms_by_doc,
                        histogram_totals_by_doc,
                        distance_matrix,
                        len(quantization_centroids),
                )
            else:
                merge_bridge_neighbors(
                    neighbor_ids_by_doc,
                    cluster_doc,
                    selected_new_neighbors,
                    doc_profile_offsets,
                    doc_profile_cluster_ids,
                    config.degree_limit,
                    histograms_by_doc,
                    histogram_totals_by_doc,
                    distance_matrix,
                    len(quantization_centroids),
                )
            cluster_inserted_docs.append(cluster_doc)

    var provisional_flattened_neighbors = flatten_neighbor_lists(neighbor_ids_by_doc)
    var provisional_neighbor_offsets = provisional_flattened_neighbors.offsets.copy()
    var provisional_neighbor_doc_indices = (
        provisional_flattened_neighbors.doc_indices.copy()
    )
    var stored_adaptive_cluster_cutoff_max = 0
    if config.enable_adaptive_cluster_cutoff:
        stored_adaptive_cluster_cutoff_max = config.adaptive_cluster_cutoff_max
    var provisional_index = GemGraphIndex(
        packed_index.doc_ids.copy(),
        doc_code_offsets.copy(),
        doc_code_ids.copy(),
        doc_code_counts.copy(),
        quantization_centroids.copy(),
        index_centroids.copy(),
        quantization_to_index.copy(),
        doc_profile_offsets.copy(),
        doc_profile_cluster_ids.copy(),
        doc_profile_scores.copy(),
        cluster_offsets.copy(),
        cluster_doc_indices.copy(),
        entry_doc_indices.copy(),
        provisional_neighbor_offsets^,
        provisional_neighbor_doc_indices^,
        packed_index.vector_dim,
        0,
        config.cluster_cutoff,
        config.enable_adaptive_cluster_cutoff,
        stored_adaptive_cluster_cutoff_max,
        config.construction_neighbor_count,
        config.degree_limit,
        config.enable_shortcuts,
    )
    var shortcut_edge_count = 0
    if config.enable_shortcuts:
        shortcut_edge_count = inject_shortcuts(
            provisional_index,
            neighbor_ids_by_doc,
            config,
        )
    var flattened_neighbors = flatten_neighbor_lists(neighbor_ids_by_doc)
    var flattened_neighbor_offsets = flattened_neighbors.offsets.copy()
    var flattened_neighbor_doc_indices = flattened_neighbors.doc_indices.copy()
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
        flattened_neighbor_offsets^,
        flattened_neighbor_doc_indices^,
        packed_index.vector_dim,
        shortcut_edge_count,
        config.cluster_cutoff,
        config.enable_adaptive_cluster_cutoff,
        stored_adaptive_cluster_cutoff_max,
        config.construction_neighbor_count,
        config.degree_limit,
        config.enable_shortcuts,
    )


def build_gem_graph_index(
    read packed_index: PackedIndex,
    fine_cluster_count: Int,
    coarse_cluster_count: Int,
    cluster_cutoff: Int,
    construction_neighbor_count: Int = DEFAULT_GEM_GRAPH_CONSTRUCTION_NEIGHBOR_COUNT,
    degree_limit: Int = DEFAULT_GEM_GRAPH_DEGREE_LIMIT,
) raises -> GemGraphIndex:
    return build_gem_graph_index_with_config(
        packed_index,
        GemGraphBuildConfig(
            fine_cluster_count,
            coarse_cluster_count,
            cluster_cutoff,
            construction_neighbor_count,
            degree_limit,
        ),
    )
