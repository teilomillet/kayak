# Adaptive supervision diagnostics for held-out GEM ablations.
#
# Owns:
# - benchmark-side reconstruction of the raw GEM document profiles used by the
#   adaptive cutoff trainer
# - compact statistics for adaptive training labels and predicted profile
#   limits
#
# Does not own:
# - graph construction
# - stored artifact serialization
# - faithfulness evaluation
#
# Assumptions:
# - raw profile reconstruction mirrors `build_gem_graph_index_with_config(...)`
# - these diagnostics are only meaningful when adaptive cutoff is enabled with
#   non-empty training pairs

from std.collections import List
from std.math import log2

from kayak.index import GemGraphBuildConfig, PackedIndex
from kayak.index.gem_graph import (
    adaptive_profile_label,
    append_or_increment_count,
    build_adaptive_profile_limits,
    build_kmeans_centroids,
    choose_effective_cluster_count,
    choose_profile_limit_for_document,
    find_document_index_by_id,
    insert_descending_score,
    nearest_centroid_index,
)
from kayak.numeric import MetricScalar, ScoreScalar, VectorScalar


struct GemHeldoutAdaptiveDiagnostics(Copyable):
    var training_label_count: Int
    var training_label_mean: Float64
    var training_label_min: Int
    var training_label_max: Int
    var training_label_one_share: Float64
    var predicted_profile_limit_count: Int
    var predicted_profile_limit_mean: Float64
    var predicted_profile_limit_min: Int
    var predicted_profile_limit_max: Int
    var predicted_profile_limit_one_share: Float64

    def __init__(out self):
        self.training_label_count = 0
        self.training_label_mean = 0.0
        self.training_label_min = 0
        self.training_label_max = 0
        self.training_label_one_share = 0.0
        self.predicted_profile_limit_count = 0
        self.predicted_profile_limit_mean = 0.0
        self.predicted_profile_limit_min = 0
        self.predicted_profile_limit_max = 0
        self.predicted_profile_limit_one_share = 0.0

    def __init__(
        out self,
        training_label_count: Int,
        training_label_mean: Float64,
        training_label_min: Int,
        training_label_max: Int,
        training_label_one_share: Float64,
        predicted_profile_limit_count: Int,
        predicted_profile_limit_mean: Float64,
        predicted_profile_limit_min: Int,
        predicted_profile_limit_max: Int,
        predicted_profile_limit_one_share: Float64,
    ) raises:
        if training_label_count < 0:
            raise Error(
                "held-out adaptive diagnostics training_label_count must be non-negative"
            )
        if predicted_profile_limit_count < 0:
            raise Error(
                "held-out adaptive diagnostics predicted_profile_limit_count must be non-negative"
            )
        self.training_label_count = training_label_count
        self.training_label_mean = training_label_mean
        self.training_label_min = training_label_min
        self.training_label_max = training_label_max
        self.training_label_one_share = training_label_one_share
        self.predicted_profile_limit_count = predicted_profile_limit_count
        self.predicted_profile_limit_mean = predicted_profile_limit_mean
        self.predicted_profile_limit_min = predicted_profile_limit_min
        self.predicted_profile_limit_max = predicted_profile_limit_max
        self.predicted_profile_limit_one_share = predicted_profile_limit_one_share


struct GemHeldoutAdaptiveProfileState(Copyable):
    var index_centroids: List[List[VectorScalar]]
    var raw_profile_ids_by_doc: List[List[Int]]
    var raw_profile_scores_by_doc: List[List[ScoreScalar]]

    def __init__(
        out self,
        var index_centroids: List[List[VectorScalar]],
        var raw_profile_ids_by_doc: List[List[Int]],
        var raw_profile_scores_by_doc: List[List[ScoreScalar]],
    ):
        self.index_centroids = index_centroids^
        self.raw_profile_ids_by_doc = raw_profile_ids_by_doc^
        self.raw_profile_scores_by_doc = raw_profile_scores_by_doc^


def mean_int_values(read values: List[Int]) -> Float64:
    if len(values) == 0:
        return 0.0

    var total = 0
    for value in values:
        total += value
    return Float64(total) / Float64(len(values))


def min_int_value(read values: List[Int]) -> Int:
    if len(values) == 0:
        return 0

    var minimum = values[0]
    for value in values:
        if value < minimum:
            minimum = value
    return minimum


def max_int_value(read values: List[Int]) -> Int:
    if len(values) == 0:
        return 0

    var maximum = values[0]
    for value in values:
        if value > maximum:
            maximum = value
    return maximum


def int_value_share(read values: List[Int], target: Int) -> Float64:
    if len(values) == 0:
        return 0.0

    var hit_count = 0
    for value in values:
        if value == target:
            hit_count += 1
    return Float64(hit_count) / Float64(len(values))


def build_gem_heldout_adaptive_profile_state(
    read packed_index: PackedIndex,
    read config: GemGraphBuildConfig,
) raises -> GemHeldoutAdaptiveProfileState:
    var effective_fine_cluster_count = choose_effective_cluster_count(
        config.fine_cluster_count, packed_index.total_vector_count
    )
    var quantization_centroids = build_kmeans_centroids(
        packed_index.token_vectors,
        effective_fine_cluster_count,
        4,
    )
    var effective_coarse_cluster_count = choose_effective_cluster_count(
        config.coarse_cluster_count, len(quantization_centroids)
    )
    var index_centroids = build_kmeans_centroids(
        quantization_centroids,
        effective_coarse_cluster_count,
        4,
    )

    var quantization_to_index = List[Int]()
    for centroid in quantization_centroids:
        quantization_to_index.append(
            nearest_centroid_index(centroid, index_centroids)
        )

    var doc_cluster_ids = List[List[Int]]()
    var doc_cluster_counts = List[List[Int]]()
    var document_frequencies = List[Int]()
    for _ in range(len(index_centroids)):
        document_frequencies.append(0)

    for document_index in range(packed_index.document_count):
        var cluster_ids = List[Int]()
        var cluster_counts = List[Int]()
        var start = packed_index.doc_offsets[document_index]
        var stop = packed_index.doc_offsets[document_index + 1]
        for token_index in range(start, stop):
            var code_id = nearest_centroid_index(
                packed_index.token_vectors[token_index], quantization_centroids
            )
            append_or_increment_count(
                cluster_ids,
                cluster_counts,
                quantization_to_index[code_id],
            )
        doc_cluster_ids.append(cluster_ids^)
        doc_cluster_counts.append(cluster_counts^)
        for cluster_id in doc_cluster_ids[document_index]:
            document_frequencies[cluster_id] += 1

    var raw_profile_limit = config.cluster_cutoff
    if (
        config.enable_adaptive_cluster_cutoff
        and config.adaptive_cluster_cutoff_max > raw_profile_limit
    ):
        raw_profile_limit = config.adaptive_cluster_cutoff_max

    var raw_profile_ids_by_doc = List[List[Int]]()
    var raw_profile_scores_by_doc = List[List[ScoreScalar]]()
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

    return GemHeldoutAdaptiveProfileState(
        index_centroids^,
        raw_profile_ids_by_doc^,
        raw_profile_scores_by_doc^,
    )


def adaptive_training_labels(
    read packed_index: PackedIndex,
    read state: GemHeldoutAdaptiveProfileState,
    read config: GemGraphBuildConfig,
) raises -> List[Int]:
    var labels = List[Int]()
    for training_pair in config.training_pairs:
        var document_index = find_document_index_by_id(
            packed_index.doc_ids, training_pair.positive_doc_id
        )
        labels.append(
            adaptive_profile_label(
                training_pair.query,
                state.index_centroids,
                state.raw_profile_ids_by_doc[document_index],
                config.cluster_top_k_per_query_token,
                config.adaptive_cluster_cutoff_max,
                config.adaptive_label_policy,
            )
        )
    return labels^


def build_gem_heldout_adaptive_diagnostics(
    read packed_index: PackedIndex,
    read config: GemGraphBuildConfig,
) raises -> GemHeldoutAdaptiveDiagnostics:
    if not config.enable_adaptive_cluster_cutoff:
        return GemHeldoutAdaptiveDiagnostics()
    if len(config.training_pairs) == 0:
        return GemHeldoutAdaptiveDiagnostics()

    var state = build_gem_heldout_adaptive_profile_state(packed_index, config)
    var training_labels = adaptive_training_labels(
        packed_index,
        state,
        config,
    )
    var predicted_limits = build_adaptive_profile_limits(
        packed_index,
        state.raw_profile_ids_by_doc,
        state.raw_profile_scores_by_doc,
        state.index_centroids,
        config,
    )

    return GemHeldoutAdaptiveDiagnostics(
        len(training_labels),
        mean_int_values(training_labels),
        min_int_value(training_labels),
        max_int_value(training_labels),
        int_value_share(training_labels, 1),
        len(predicted_limits),
        mean_int_values(predicted_limits),
        min_int_value(predicted_limits),
        max_int_value(predicted_limits),
        int_value_share(predicted_limits, 1),
    )
