# Held-out synthetic GEM ablation summaries.
#
# Owns:
# - deterministic train/eval query splitting for supervised GEM features
# - GEM build-variant enumeration over a fixed frontier-style base config
# - JSON summaries that combine build metadata with held-out faithfulness
#
# Does not own:
# - public benchmark supervision policy
# - generic faithfulness evaluation logic
#
# Assumptions:
# - supervision positives come from the first relevant doc id per training query
# - training queries are selected by an evenly spaced deterministic split so the
#   held-out evaluation slice is disjoint without introducing randomness
# - query-time GEM search parameters remain explicit and constant so build-time
#   adaptive cutoff and shortcut changes are the variable under test

from std.collections import List
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_resolved_collection_snapshot,
    loaded_segment_stored_gem_graph_index,
)
from kayak.contracts import EncodedDocument
from kayak.eval import JudgedQuery, JudgedTask
from kayak.index import (
    DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
    DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
    GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE,
    GemGraphBuildConfig,
    GemGraphIndex,
    GemGraphTrainingPair,
    query_entry_doc_indices,
    query_representative_doc_indices,
    query_relevant_cluster_ids,
)
from kayak.planning import (
    best_effort_faithfulness_policy,
    gem_graph_search_plan,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .faithfulness_frontier_json import (
    build_faithfulness_frontier_summary_for_plan,
)
from .gem_frontier_config import frontier_gem_graph_build_config
from .gem_heldout_adaptive_diagnostics import (
    build_gem_heldout_adaptive_diagnostics,
)
from .json_common import json_escape


comptime GEM_HELDOUT_TRAINING_SELECTION_POLICY = "evenly_spaced_queries"
comptime GEM_HELDOUT_POSITIVE_SELECTION_POLICY = "first_relevant_doc_id"
comptime GEM_HELDOUT_VARIANT_BASELINE = "baseline"
comptime GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF = "adaptive_cutoff"
comptime GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE = (
    "adaptive_cutoff_relevant_coverage"
)
comptime GEM_HELDOUT_VARIANT_SHORTCUTS = "shortcuts"
comptime GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS = (
    "adaptive_cutoff_shortcuts"
)


struct GemHeldoutQuerySplit(Copyable):
    var training_selection_policy: String
    var positive_selection_policy: String
    var training_query_ids: List[String]
    var evaluation_query_ids: List[String]
    var training_pairs: List[GemGraphTrainingPair]
    var evaluation_task: StoredJudgedTask

    def __init__(
        out self,
        var training_selection_policy: String,
        var positive_selection_policy: String,
        var training_query_ids: List[String],
        var evaluation_query_ids: List[String],
        var training_pairs: List[GemGraphTrainingPair],
        evaluation_task: StoredJudgedTask,
    ):
        self.training_selection_policy = training_selection_policy^
        self.positive_selection_policy = positive_selection_policy^
        self.training_query_ids = training_query_ids^
        self.evaluation_query_ids = evaluation_query_ids^
        self.training_pairs = training_pairs^
        self.evaluation_task = evaluation_task.copy()


struct GemHeldoutAblationVariantSpec(Copyable):
    var variant_kind: String
    var build_config: GemGraphBuildConfig

    def __init__(
        out self, var variant_kind: String, read build_config: GemGraphBuildConfig
    ):
        self.variant_kind = variant_kind^
        self.build_config = build_config.copy()


struct GemHeldoutAblationSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var family: String
    var slice_name: String
    var collection_id: String
    var snapshot_id: String
    var variant_kind: String
    var training_selection_policy: String
    var positive_selection_policy: String
    var training_query_count: Int
    var evaluation_query_count: Int
    var final_k: Int
    var candidate_k: Int
    var query_vector_budget: Int
    var search_cluster_top_k_per_query_token: Int
    var search_beam_width: Int
    var fine_cluster_count: Int
    var coarse_cluster_count: Int
    var cluster_cutoff: Int
    var construction_neighbor_count: Int
    var degree_limit: Int
    var build_cluster_top_k_per_query_token: Int
    var adaptive_cluster_cutoff_enabled: Bool
    var adaptive_cluster_cutoff_max: Int
    var adaptive_label_policy: String
    var shortcuts_enabled: Bool
    var shortcut_candidate_k: Int
    var shortcut_beam_width: Int
    var artifact_document_count: Int
    var artifact_cluster_count: Int
    var artifact_graph_edge_count: Int
    var artifact_shortcut_edge_count: Int
    var artifact_entry_point_count: Int
    var artifact_quantization_centroid_count: Int
    var mean_document_profile_limit: Float64
    var min_document_profile_limit: Int
    var max_document_profile_limit: Int
    var mean_training_positive_profile_limit: Float64
    var mean_evaluation_positive_profile_limit: Float64
    var training_positive_profile_hit_rate: Float64
    var evaluation_positive_profile_hit_rate: Float64
    var evaluation_positive_entry_rate: Float64
    var evaluation_positive_reachable_rate: Float64
    var adaptive_training_label_count: Int
    var adaptive_training_label_mean: Float64
    var adaptive_training_label_min: Int
    var adaptive_training_label_max: Int
    var adaptive_training_label_one_share: Float64
    var adaptive_predicted_profile_limit_count: Int
    var adaptive_predicted_profile_limit_mean: Float64
    var adaptive_predicted_profile_limit_min: Int
    var adaptive_predicted_profile_limit_max: Int
    var adaptive_predicted_profile_limit_one_share: Float64
    var mean_candidate_generation_seconds: Float64
    var mean_search_seconds: Float64
    var mean_candidate_recall_at_final_k: Float64
    var mean_ndcg_at_k: Float64
    var mean_reciprocal_rank: Float64
    var mean_recall_at_k: Float64
    var success_rate_at_k: Float64
    var stage1_vector_count: Int
    var stage1_token_count: Int
    var stage1_byte_size: Int
    var stage1_bytes_per_document: Float64
    var stage1_bytes_per_vector: Float64
    var mean_stage1_graph_visited_vertex_count: Float64
    var mean_stage1_graph_expanded_edge_count: Float64
    var mean_stage1_graph_visited_cluster_count: Float64
    var mean_stage1_graph_entry_point_count: Float64
    var mean_stage1_graph_max_frontier_size: Float64

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var family: String,
        var slice_name: String,
        var collection_id: String,
        var snapshot_id: String,
        var variant_kind: String,
        var training_selection_policy: String,
        var positive_selection_policy: String,
        training_query_count: Int,
        evaluation_query_count: Int,
        final_k: Int,
        candidate_k: Int,
        query_vector_budget: Int,
        search_cluster_top_k_per_query_token: Int,
        search_beam_width: Int,
        fine_cluster_count: Int,
        coarse_cluster_count: Int,
        cluster_cutoff: Int,
        construction_neighbor_count: Int,
        degree_limit: Int,
        build_cluster_top_k_per_query_token: Int,
        adaptive_cluster_cutoff_enabled: Bool,
        adaptive_cluster_cutoff_max: Int,
        var adaptive_label_policy: String,
        shortcuts_enabled: Bool,
        shortcut_candidate_k: Int,
        shortcut_beam_width: Int,
        artifact_document_count: Int,
        artifact_cluster_count: Int,
        artifact_graph_edge_count: Int,
        artifact_shortcut_edge_count: Int,
        artifact_entry_point_count: Int,
        artifact_quantization_centroid_count: Int,
        mean_document_profile_limit: Float64,
        min_document_profile_limit: Int,
        max_document_profile_limit: Int,
        mean_training_positive_profile_limit: Float64,
        mean_evaluation_positive_profile_limit: Float64,
        training_positive_profile_hit_rate: Float64,
        evaluation_positive_profile_hit_rate: Float64,
        evaluation_positive_entry_rate: Float64,
        evaluation_positive_reachable_rate: Float64,
        adaptive_training_label_count: Int,
        adaptive_training_label_mean: Float64,
        adaptive_training_label_min: Int,
        adaptive_training_label_max: Int,
        adaptive_training_label_one_share: Float64,
        adaptive_predicted_profile_limit_count: Int,
        adaptive_predicted_profile_limit_mean: Float64,
        adaptive_predicted_profile_limit_min: Int,
        adaptive_predicted_profile_limit_max: Int,
        adaptive_predicted_profile_limit_one_share: Float64,
        mean_candidate_generation_seconds: Float64,
        mean_search_seconds: Float64,
        mean_candidate_recall_at_final_k: Float64,
        mean_ndcg_at_k: Float64,
        mean_reciprocal_rank: Float64,
        mean_recall_at_k: Float64,
        success_rate_at_k: Float64,
        stage1_vector_count: Int,
        stage1_token_count: Int,
        stage1_byte_size: Int,
        stage1_bytes_per_document: Float64,
        stage1_bytes_per_vector: Float64,
        mean_stage1_graph_visited_vertex_count: Float64,
        mean_stage1_graph_expanded_edge_count: Float64,
        mean_stage1_graph_visited_cluster_count: Float64,
        mean_stage1_graph_entry_point_count: Float64,
        mean_stage1_graph_max_frontier_size: Float64,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.family = family^
        self.slice_name = slice_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.variant_kind = variant_kind^
        self.training_selection_policy = training_selection_policy^
        self.positive_selection_policy = positive_selection_policy^
        self.training_query_count = training_query_count
        self.evaluation_query_count = evaluation_query_count
        self.final_k = final_k
        self.candidate_k = candidate_k
        self.query_vector_budget = query_vector_budget
        self.search_cluster_top_k_per_query_token = (
            search_cluster_top_k_per_query_token
        )
        self.search_beam_width = search_beam_width
        self.fine_cluster_count = fine_cluster_count
        self.coarse_cluster_count = coarse_cluster_count
        self.cluster_cutoff = cluster_cutoff
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit
        self.build_cluster_top_k_per_query_token = (
            build_cluster_top_k_per_query_token
        )
        self.adaptive_cluster_cutoff_enabled = adaptive_cluster_cutoff_enabled
        self.adaptive_cluster_cutoff_max = adaptive_cluster_cutoff_max
        self.adaptive_label_policy = adaptive_label_policy^
        self.shortcuts_enabled = shortcuts_enabled
        self.shortcut_candidate_k = shortcut_candidate_k
        self.shortcut_beam_width = shortcut_beam_width
        self.artifact_document_count = artifact_document_count
        self.artifact_cluster_count = artifact_cluster_count
        self.artifact_graph_edge_count = artifact_graph_edge_count
        self.artifact_shortcut_edge_count = artifact_shortcut_edge_count
        self.artifact_entry_point_count = artifact_entry_point_count
        self.artifact_quantization_centroid_count = (
            artifact_quantization_centroid_count
        )
        self.mean_document_profile_limit = mean_document_profile_limit
        self.min_document_profile_limit = min_document_profile_limit
        self.max_document_profile_limit = max_document_profile_limit
        self.mean_training_positive_profile_limit = (
            mean_training_positive_profile_limit
        )
        self.mean_evaluation_positive_profile_limit = (
            mean_evaluation_positive_profile_limit
        )
        self.training_positive_profile_hit_rate = training_positive_profile_hit_rate
        self.evaluation_positive_profile_hit_rate = evaluation_positive_profile_hit_rate
        self.evaluation_positive_entry_rate = evaluation_positive_entry_rate
        self.evaluation_positive_reachable_rate = (
            evaluation_positive_reachable_rate
        )
        self.adaptive_training_label_count = adaptive_training_label_count
        self.adaptive_training_label_mean = adaptive_training_label_mean
        self.adaptive_training_label_min = adaptive_training_label_min
        self.adaptive_training_label_max = adaptive_training_label_max
        self.adaptive_training_label_one_share = (
            adaptive_training_label_one_share
        )
        self.adaptive_predicted_profile_limit_count = (
            adaptive_predicted_profile_limit_count
        )
        self.adaptive_predicted_profile_limit_mean = (
            adaptive_predicted_profile_limit_mean
        )
        self.adaptive_predicted_profile_limit_min = (
            adaptive_predicted_profile_limit_min
        )
        self.adaptive_predicted_profile_limit_max = (
            adaptive_predicted_profile_limit_max
        )
        self.adaptive_predicted_profile_limit_one_share = (
            adaptive_predicted_profile_limit_one_share
        )
        self.mean_candidate_generation_seconds = mean_candidate_generation_seconds
        self.mean_search_seconds = mean_search_seconds
        self.mean_candidate_recall_at_final_k = mean_candidate_recall_at_final_k
        self.mean_ndcg_at_k = mean_ndcg_at_k
        self.mean_reciprocal_rank = mean_reciprocal_rank
        self.mean_recall_at_k = mean_recall_at_k
        self.success_rate_at_k = success_rate_at_k
        self.stage1_vector_count = stage1_vector_count
        self.stage1_token_count = stage1_token_count
        self.stage1_byte_size = stage1_byte_size
        self.stage1_bytes_per_document = stage1_bytes_per_document
        self.stage1_bytes_per_vector = stage1_bytes_per_vector
        self.mean_stage1_graph_visited_vertex_count = (
            mean_stage1_graph_visited_vertex_count
        )
        self.mean_stage1_graph_expanded_edge_count = (
            mean_stage1_graph_expanded_edge_count
        )
        self.mean_stage1_graph_visited_cluster_count = (
            mean_stage1_graph_visited_cluster_count
        )
        self.mean_stage1_graph_entry_point_count = (
            mean_stage1_graph_entry_point_count
        )
        self.mean_stage1_graph_max_frontier_size = (
            mean_stage1_graph_max_frontier_size
        )


def default_gem_heldout_training_query_count(
    read stored_task: StoredJudgedTask
) raises -> Int:
    var query_count = len(stored_task.task.queries)
    if query_count < 2:
        raise Error("held-out GEM ablation requires at least two judged queries")

    var training_query_count = query_count // 2
    if training_query_count <= 0:
        training_query_count = 1
    if training_query_count >= query_count:
        training_query_count = query_count - 1
    return training_query_count


def list_contains_int(read values: List[Int], needle: Int) -> Bool:
    for value in values:
        if value == needle:
            return True
    return False


def evenly_spaced_training_query_indices(
    total_query_count: Int, training_query_count: Int
) raises -> List[Int]:
    if total_query_count < 2:
        raise Error("held-out GEM ablation requires at least two judged queries")
    if training_query_count <= 0:
        raise Error("held-out GEM training_query_count must be positive")
    if training_query_count >= total_query_count:
        raise Error(
            "held-out GEM training_query_count must leave at least one evaluation query"
        )

    var indices = List[Int]()
    for training_rank in range(training_query_count):
        indices.append((training_rank * total_query_count) // training_query_count)
    return indices^


def first_relevant_doc_id_for_query(read judged_query: JudgedQuery) raises -> String:
    if len(judged_query.relevant_doc_ids) == 0:
        raise Error(
            "held-out GEM training queries must define at least one relevant doc id"
        )
    return judged_query.relevant_doc_ids[0].copy()


def training_pair_for_judged_query(
    read judged_query: JudgedQuery
) raises -> GemGraphTrainingPair:
    return GemGraphTrainingPair(
        judged_query.query.copy(),
        first_relevant_doc_id_for_query(judged_query),
    )


def build_gem_heldout_query_split(
    read stored_task: StoredJudgedTask, training_query_count: Int
) raises -> GemHeldoutQuerySplit:
    var task = stored_task.task.copy()
    var training_query_indices = evenly_spaced_training_query_indices(
        len(task.queries),
        training_query_count,
    )
    var training_query_ids = List[String]()
    var evaluation_query_ids = List[String]()
    var training_pairs = List[GemGraphTrainingPair]()
    var evaluation_queries = List[JudgedQuery]()

    for query_index in range(len(task.queries)):
        var judged_query = task.queries[query_index].copy()
        if list_contains_int(training_query_indices, query_index):
            training_query_ids.append(judged_query.query_id.copy())
            training_pairs.append(training_pair_for_judged_query(judged_query))
            continue

        evaluation_query_ids.append(judged_query.query_id.copy())
        evaluation_queries.append(judged_query.copy())

    if len(evaluation_queries) == 0:
        raise Error("held-out GEM split must leave at least one evaluation query")

    return GemHeldoutQuerySplit(
        GEM_HELDOUT_TRAINING_SELECTION_POLICY,
        GEM_HELDOUT_POSITIVE_SELECTION_POLICY,
        training_query_ids^,
        evaluation_query_ids^,
        training_pairs^,
        StoredJudgedTask(
            stored_task.dataset_id.copy(),
            stored_task.model_name.copy(),
            stored_task.vector_scalar_name.copy(),
            JudgedTask(
                task.family.copy(),
                task.slice_name + "__gem_heldout_eval",
                task.why
                + " GEM supervision uses a disjoint evenly spaced query split; this slice evaluates only the held-out queries.",
                task.primary_metric.copy(),
                task.k,
                task.nominal_query_vector_count,
                task.nominal_document_vector_count,
                task.vector_dim,
                List[EncodedDocument](),
                evaluation_queries^,
            ),
        ),
    )


def default_adaptive_cluster_cutoff_max(read base_config: GemGraphBuildConfig) -> Int:
    var adaptive_max = base_config.cluster_cutoff * 2
    if adaptive_max < base_config.cluster_cutoff:
        adaptive_max = base_config.cluster_cutoff
    if adaptive_max <= 0:
        adaptive_max = 1
    if (
        base_config.coarse_cluster_count > 0
        and adaptive_max > base_config.coarse_cluster_count
    ):
        adaptive_max = base_config.coarse_cluster_count
    if adaptive_max < base_config.cluster_cutoff:
        adaptive_max = base_config.cluster_cutoff
    return adaptive_max


def resolved_adaptive_cluster_cutoff_max(read base_config: GemGraphBuildConfig) -> Int:
    var adaptive_max = base_config.adaptive_cluster_cutoff_max
    if adaptive_max <= 0:
        adaptive_max = default_adaptive_cluster_cutoff_max(base_config)

    if (
        base_config.coarse_cluster_count > 0
        and adaptive_max > base_config.coarse_cluster_count
    ):
        adaptive_max = base_config.coarse_cluster_count
    return adaptive_max


def document_profile_limit_for_index(
    read doc_profile_offsets: List[Int], document_index: Int
) raises -> Int:
    if document_index < 0 or document_index >= len(doc_profile_offsets) - 1:
        raise Error("document profile document_index is out of range")
    return (
        doc_profile_offsets[document_index + 1]
        - doc_profile_offsets[document_index]
    )


def mean_document_profile_limit(
    read doc_profile_offsets: List[Int], document_count: Int
) -> Float64:
    if document_count <= 0:
        return 0.0

    var total = 0
    for document_index in range(document_count):
        total += (
            doc_profile_offsets[document_index + 1]
            - doc_profile_offsets[document_index]
        )
    return Float64(total) / Float64(document_count)


def min_document_profile_limit(
    read doc_profile_offsets: List[Int], document_count: Int
) -> Int:
    if document_count <= 0:
        return 0

    var minimum = (
        doc_profile_offsets[1]
        - doc_profile_offsets[0]
    )
    for document_index in range(1, document_count):
        var profile_limit = (
            doc_profile_offsets[document_index + 1]
            - doc_profile_offsets[document_index]
        )
        if profile_limit < minimum:
            minimum = profile_limit
    return minimum


def max_document_profile_limit(
    read doc_profile_offsets: List[Int], document_count: Int
) -> Int:
    if document_count <= 0:
        return 0

    var maximum = 0
    for document_index in range(document_count):
        var profile_limit = (
            doc_profile_offsets[document_index + 1]
            - doc_profile_offsets[document_index]
        )
        if profile_limit > maximum:
            maximum = profile_limit
    return maximum


def find_doc_index_by_id(read doc_ids: List[String], doc_id: String) raises -> Int:
    for document_index in range(len(doc_ids)):
        if doc_ids[document_index] == doc_id:
            return document_index
    raise Error("held-out GEM summary doc_id was not found in the stored graph")


def mean_profile_limit_for_doc_ids(
    read doc_ids: List[String],
    read doc_profile_offsets: List[Int],
    read selected_doc_ids: List[String],
) raises -> Float64:
    if len(selected_doc_ids) == 0:
        return 0.0

    var total = 0
    for doc_id in selected_doc_ids:
        total += document_profile_limit_for_index(
            doc_profile_offsets,
            find_doc_index_by_id(doc_ids, doc_id),
        )
    return Float64(total) / Float64(len(selected_doc_ids))


def document_profile_contains_any_cluster(
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    document_index: Int,
    read relevant_cluster_ids: List[Int],
) raises -> Bool:
    if len(relevant_cluster_ids) == 0:
        return False

    for profile_index in range(
        doc_profile_offsets[document_index],
        doc_profile_offsets[document_index + 1],
    ):
        for relevant_cluster_id in relevant_cluster_ids:
            if doc_profile_cluster_ids[profile_index] == relevant_cluster_id:
                return True
    return False


def training_positive_profile_hit_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> Float64:
    if len(split.training_pairs) == 0:
        return 0.0

    var hit_count = 0
    for training_pair in split.training_pairs:
        var document_index = find_doc_index_by_id(
            doc_ids,
            training_pair.positive_doc_id,
        )
        if document_profile_contains_any_cluster(
            doc_profile_offsets,
            doc_profile_cluster_ids,
            document_index,
            query_relevant_cluster_ids(
                training_pair.query,
                index,
                cluster_top_k_per_query_token,
            ),
        ):
            hit_count += 1
    return Float64(hit_count) / Float64(len(split.training_pairs))


def evaluation_positive_profile_hit_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read doc_profile_offsets: List[Int],
    read doc_profile_cluster_ids: List[Int],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> Float64:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var hit_count = 0
    for judged_query in queries:
        var document_index = find_doc_index_by_id(
            doc_ids,
            first_relevant_doc_id_for_query(judged_query),
        )
        if document_profile_contains_any_cluster(
            doc_profile_offsets,
            doc_profile_cluster_ids,
            document_index,
            query_relevant_cluster_ids(
                judged_query.query,
                index,
                cluster_top_k_per_query_token,
            ),
        ):
            hit_count += 1
    return Float64(hit_count) / Float64(len(queries))


def query_positive_doc_indices(
    read judged_query: JudgedQuery,
    read doc_ids: List[String],
) raises -> List[Int]:
    var indices = List[Int]()
    for doc_id in judged_query.relevant_doc_ids:
        indices.append(find_doc_index_by_id(doc_ids, doc_id))
    return indices^


def any_doc_index_matches(
    read left_indices: List[Int], read right_indices: List[Int]
) -> Bool:
    for left_index in left_indices:
        if list_contains_int(right_indices, left_index):
            return True
    return False


def gated_reachable_doc_indices(
    read index: GemGraphIndex,
    read entry_doc_indices: List[Int],
    read relevant_cluster_ids: List[Int],
) raises -> List[Int]:
    var visited = List[Int]()
    var queue = List[Int]()
    for entry_doc in entry_doc_indices:
        if entry_doc < 0:
            continue
        if not document_profile_contains_any_cluster(
            index.doc_profile_offsets,
            index.doc_profile_cluster_ids,
            entry_doc,
            relevant_cluster_ids,
        ):
            continue
        if list_contains_int(visited, entry_doc):
            continue
        visited.append(entry_doc)
        queue.append(entry_doc)

    var queue_index = 0
    while queue_index < len(queue):
        var current_doc = queue[queue_index]
        queue_index += 1
        for neighbor_index in range(
            index.neighbor_offsets[current_doc],
            index.neighbor_offsets[current_doc + 1],
        ):
            var neighbor_doc = index.neighbor_doc_indices[neighbor_index]
            if list_contains_int(visited, neighbor_doc):
                continue
            if not document_profile_contains_any_cluster(
                index.doc_profile_offsets,
                index.doc_profile_cluster_ids,
                neighbor_doc,
                relevant_cluster_ids,
            ):
                continue
            visited.append(neighbor_doc)
            queue.append(neighbor_doc)
    return visited^


def evaluation_positive_entry_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> Float64:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var hit_count = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            index,
            cluster_top_k_per_query_token,
        )
        if any_doc_index_matches(
            query_positive_doc_indices(judged_query, doc_ids),
            query_entry_doc_indices(index, relevant_clusters),
        ):
            hit_count += 1
    return Float64(hit_count) / Float64(len(queries))


def evaluation_positive_representative_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
    representative_depth: Int,
) raises -> Float64:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var hit_count = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            index,
            cluster_top_k_per_query_token,
        )
        if any_doc_index_matches(
            query_positive_doc_indices(judged_query, doc_ids),
            query_representative_doc_indices(
                index,
                relevant_clusters,
                representative_depth,
            ),
        ):
            hit_count += 1
    return Float64(hit_count) / Float64(len(queries))


def evaluation_positive_reachable_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> Float64:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var hit_count = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            index,
            cluster_top_k_per_query_token,
        )
        if any_doc_index_matches(
            query_positive_doc_indices(judged_query, doc_ids),
            gated_reachable_doc_indices(
                index,
                query_entry_doc_indices(index, relevant_clusters),
                relevant_clusters,
            ),
        ):
            hit_count += 1
    return Float64(hit_count) / Float64(len(queries))


def query_positive_min_gated_hop_count(
    read positive_doc_indices: List[Int],
    read index: GemGraphIndex,
    read entry_doc_indices: List[Int],
    read relevant_cluster_ids: List[Int],
) raises -> Int:
    for positive_doc_index in positive_doc_indices:
        if list_contains_int(entry_doc_indices, positive_doc_index):
            return 0

    var visited = List[Int]()
    var queue = List[Int]()
    var queue_hops = List[Int]()
    for entry_doc in entry_doc_indices:
        if entry_doc < 0:
            continue
        if not document_profile_contains_any_cluster(
            index.doc_profile_offsets,
            index.doc_profile_cluster_ids,
            entry_doc,
            relevant_cluster_ids,
        ):
            continue
        if list_contains_int(visited, entry_doc):
            continue
        visited.append(entry_doc)
        queue.append(entry_doc)
        queue_hops.append(0)

    var queue_index = 0
    while queue_index < len(queue):
        var current_doc = queue[queue_index]
        var current_hops = queue_hops[queue_index]
        queue_index += 1
        for neighbor_index in range(
            index.neighbor_offsets[current_doc],
            index.neighbor_offsets[current_doc + 1],
        ):
            var neighbor_doc = index.neighbor_doc_indices[neighbor_index]
            if list_contains_int(visited, neighbor_doc):
                continue
            if not document_profile_contains_any_cluster(
                index.doc_profile_offsets,
                index.doc_profile_cluster_ids,
                neighbor_doc,
                relevant_cluster_ids,
            ):
                continue
            var neighbor_hops = current_hops + 1
            if list_contains_int(positive_doc_indices, neighbor_doc):
                return neighbor_hops
            visited.append(neighbor_doc)
            queue.append(neighbor_doc)
            queue_hops.append(neighbor_hops)
    return -1


def evaluation_positive_mean_reachable_hop_count(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
) raises -> Float64:
    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var reachable_query_count = 0
    var hop_total = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            index,
            cluster_top_k_per_query_token,
        )
        var min_hops = query_positive_min_gated_hop_count(
            query_positive_doc_indices(judged_query, doc_ids),
            index,
            query_entry_doc_indices(index, relevant_clusters),
            relevant_clusters,
        )
        if min_hops < 0:
            continue
        reachable_query_count += 1
        hop_total += min_hops
    if reachable_query_count == 0:
        return 0.0
    return Float64(hop_total) / Float64(reachable_query_count)


def evaluation_positive_within_hop_rate(
    read split: GemHeldoutQuerySplit,
    read doc_ids: List[String],
    read index: GemGraphIndex,
    cluster_top_k_per_query_token: Int,
    max_hops: Int,
) raises -> Float64:
    if max_hops < 0:
        raise Error("held-out GEM max_hops must be non-negative")

    var queries = split.evaluation_task.task.queries.copy()
    if len(queries) == 0:
        return 0.0

    var hit_count = 0
    for judged_query in queries:
        var relevant_clusters = query_relevant_cluster_ids(
            judged_query.query,
            index,
            cluster_top_k_per_query_token,
        )
        var min_hops = query_positive_min_gated_hop_count(
            query_positive_doc_indices(judged_query, doc_ids),
            index,
            query_entry_doc_indices(index, relevant_clusters),
            relevant_clusters,
        )
        if min_hops >= 0 and min_hops <= max_hops:
            hit_count += 1
    return Float64(hit_count) / Float64(len(queries))


def training_positive_doc_ids(
    read split: GemHeldoutQuerySplit
) -> List[String]:
    var doc_ids = List[String]()
    for training_pair in split.training_pairs:
        doc_ids.append(training_pair.positive_doc_id.copy())
    return doc_ids^


def evaluation_positive_doc_ids(
    read split: GemHeldoutQuerySplit
) -> List[String]:
    var doc_ids = List[String]()
    for judged_query in split.evaluation_task.task.queries:
        for doc_id in judged_query.relevant_doc_ids:
            doc_ids.append(doc_id.copy())
    return doc_ids^


def gem_heldout_ablation_variant_spec(
    variant_kind: String,
    read base_config: GemGraphBuildConfig,
    read training_pairs: List[GemGraphTrainingPair],
) raises -> GemHeldoutAblationVariantSpec:
    var adaptive_enabled = False
    var shortcuts_enabled = False
    var adaptive_cluster_cutoff_max = resolved_adaptive_cluster_cutoff_max(base_config)
    var selected_training_pairs = List[GemGraphTrainingPair]()
    var adaptive_label_policy = base_config.adaptive_label_policy.copy()

    if variant_kind == GEM_HELDOUT_VARIANT_BASELINE:
        pass
    elif variant_kind == GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF:
        adaptive_enabled = True
        selected_training_pairs = training_pairs.copy()
    elif variant_kind == GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_RELEVANT_COVERAGE:
        adaptive_enabled = True
        selected_training_pairs = training_pairs.copy()
        adaptive_label_policy = (
            GEM_GRAPH_ADAPTIVE_LABEL_POLICY_RELEVANT_CLUSTER_COVERAGE
        )
    elif variant_kind == GEM_HELDOUT_VARIANT_SHORTCUTS:
        shortcuts_enabled = True
        selected_training_pairs = training_pairs.copy()
    elif variant_kind == GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS:
        adaptive_enabled = True
        shortcuts_enabled = True
        selected_training_pairs = training_pairs.copy()
    else:
        raise Error("unknown held-out GEM ablation variant_kind: " + variant_kind)

    return GemHeldoutAblationVariantSpec(
        variant_kind,
        GemGraphBuildConfig(
            base_config.fine_cluster_count,
            base_config.coarse_cluster_count,
            base_config.cluster_cutoff,
            base_config.construction_neighbor_count,
            base_config.degree_limit,
            adaptive_enabled,
            adaptive_cluster_cutoff_max,
            base_config.adaptive_tree_max_depth,
            base_config.cluster_top_k_per_query_token,
            shortcuts_enabled,
            base_config.shortcut_candidate_k,
            base_config.shortcut_beam_width,
            selected_training_pairs^,
            adaptive_label_policy^,
        ),
    )


def standard_gem_heldout_ablation_variant_specs(
    read base_config: GemGraphBuildConfig,
    read split: GemHeldoutQuerySplit,
) raises -> List[GemHeldoutAblationVariantSpec]:
    return [
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_BASELINE,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_SHORTCUTS,
            base_config,
            split.training_pairs,
        ),
        gem_heldout_ablation_variant_spec(
            GEM_HELDOUT_VARIANT_ADAPTIVE_CUTOFF_SHORTCUTS,
            base_config,
            split.training_pairs,
        ),
    ]


def standard_gem_heldout_ablation_variant_specs(
    read stored_index: StoredPackedIndex,
    query_vector_budget: Int,
    read split: GemHeldoutQuerySplit,
) raises -> List[GemHeldoutAblationVariantSpec]:
    return standard_gem_heldout_ablation_variant_specs(
        frontier_gem_graph_build_config(stored_index, query_vector_budget),
        split,
    )


def build_gem_heldout_ablation_summary(
    read backend: ExactCpuBackend,
    read stored_index: StoredPackedIndex,
    read split: GemHeldoutQuerySplit,
    collection_root: Path,
    collection_id: String,
    read variant: GemHeldoutAblationVariantSpec,
    candidate_k: Int,
    search_cluster_top_k_per_query_token: Int = DEFAULT_GEM_GRAPH_QUERY_CLUSTER_TOP_K,
    search_beam_width: Int = DEFAULT_GEM_GRAPH_QUERY_BEAM_WIDTH,
) raises -> GemHeldoutAblationSummary:
    var snapshot_id = SnapshotId("snapshot-0001")
    var heldout_task = split.evaluation_task.task.copy()
    var adaptive_diagnostics = build_gem_heldout_adaptive_diagnostics(
        stored_index.index,
        variant.build_config,
    )
    var collection_path = ensure_one_segment_collection_mirror(
        collection_root,
        CollectionId(collection_id),
        TenantId("public"),
        NamespaceId("benchmark"),
        snapshot_id,
        1,
        stored_index,
        variant.build_config,
    )
    var snapshot = load_resolved_collection_snapshot(collection_path, snapshot_id)
    var stored_gem = loaded_segment_stored_gem_graph_index(snapshot.segments[0])
    var profile_offsets = stored_gem.index.doc_profile_offsets.copy()
    var frontier = build_faithfulness_frontier_summary_for_plan(
        backend,
        split.evaluation_task,
        snapshot,
        gem_graph_search_plan(
            heldout_task.k,
            candidate_k,
            best_effort_faithfulness_policy(),
            search_cluster_top_k_per_query_token,
            search_beam_width,
        ),
        heldout_task.nominal_query_vector_count,
        variant.build_config.fine_cluster_count,
        0,
    )

    return GemHeldoutAblationSummary(
        frontier.dataset_id.copy(),
        frontier.model_name.copy(),
        frontier.family.copy(),
        frontier.slice_name.copy(),
        frontier.collection_id.copy(),
        frontier.snapshot_id.copy(),
        variant.variant_kind.copy(),
        split.training_selection_policy.copy(),
        split.positive_selection_policy.copy(),
        len(split.training_pairs),
        len(heldout_task.queries),
        frontier.final_k,
        frontier.candidate_k,
        frontier.query_vector_budget,
        search_cluster_top_k_per_query_token,
        search_beam_width,
        variant.build_config.fine_cluster_count,
        variant.build_config.coarse_cluster_count,
        variant.build_config.cluster_cutoff,
        variant.build_config.construction_neighbor_count,
        variant.build_config.degree_limit,
        variant.build_config.cluster_top_k_per_query_token,
        stored_gem.adaptive_cluster_cutoff_enabled,
        stored_gem.adaptive_cluster_cutoff_max,
        stored_gem.adaptive_label_policy.copy(),
        stored_gem.shortcuts_enabled,
        variant.build_config.shortcut_candidate_k,
        variant.build_config.shortcut_beam_width,
        stored_gem.document_count,
        stored_gem.cluster_count,
        stored_gem.graph_edge_count,
        stored_gem.shortcut_edge_count,
        stored_gem.entry_point_count,
        stored_gem.quantization_centroid_count,
        mean_document_profile_limit(profile_offsets, stored_gem.document_count),
        min_document_profile_limit(profile_offsets, stored_gem.document_count),
        max_document_profile_limit(profile_offsets, stored_gem.document_count),
        mean_profile_limit_for_doc_ids(
            stored_gem.index.doc_ids,
            profile_offsets,
            training_positive_doc_ids(split),
        ),
        mean_profile_limit_for_doc_ids(
            stored_gem.index.doc_ids,
            profile_offsets,
            evaluation_positive_doc_ids(split),
        ),
        training_positive_profile_hit_rate(
            split,
            stored_gem.index.doc_ids,
            profile_offsets,
            stored_gem.index.doc_profile_cluster_ids,
            stored_gem.index,
            variant.build_config.cluster_top_k_per_query_token,
        ),
        evaluation_positive_profile_hit_rate(
            split,
            stored_gem.index.doc_ids,
            profile_offsets,
            stored_gem.index.doc_profile_cluster_ids,
            stored_gem.index,
            variant.build_config.cluster_top_k_per_query_token,
        ),
        evaluation_positive_entry_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            variant.build_config.cluster_top_k_per_query_token,
        ),
        evaluation_positive_reachable_rate(
            split,
            stored_gem.index.doc_ids,
            stored_gem.index,
            variant.build_config.cluster_top_k_per_query_token,
        ),
        adaptive_diagnostics.training_label_count,
        adaptive_diagnostics.training_label_mean,
        adaptive_diagnostics.training_label_min,
        adaptive_diagnostics.training_label_max,
        adaptive_diagnostics.training_label_one_share,
        adaptive_diagnostics.predicted_profile_limit_count,
        adaptive_diagnostics.predicted_profile_limit_mean,
        adaptive_diagnostics.predicted_profile_limit_min,
        adaptive_diagnostics.predicted_profile_limit_max,
        adaptive_diagnostics.predicted_profile_limit_one_share,
        frontier.mean_candidate_generation_seconds,
        frontier.mean_search_seconds,
        frontier.mean_candidate_recall_at_final_k,
        frontier.mean_ndcg_at_k,
        frontier.mean_reciprocal_rank,
        frontier.mean_recall_at_k,
        frontier.success_rate_at_k,
        frontier.stage1_vector_count,
        frontier.stage1_token_count,
        frontier.stage1_byte_size,
        frontier.stage1_bytes_per_document,
        frontier.stage1_bytes_per_vector,
        frontier.mean_stage1_graph_visited_vertex_count,
        frontier.mean_stage1_graph_expanded_edge_count,
        frontier.mean_stage1_graph_visited_cluster_count,
        frontier.mean_stage1_graph_entry_point_count,
        frontier.mean_stage1_graph_max_frontier_size,
    )


def append_gem_heldout_ablation_summary_json(
    mut buffer: String, read summary: GemHeldoutAblationSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"family\":\"" + json_escape(summary.family) + "\","
    buffer += "\"slice_name\":\"" + json_escape(summary.slice_name) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"variant_kind\":\"" + json_escape(summary.variant_kind) + "\","
    buffer += "\"training_selection_policy\":\""
    buffer += json_escape(summary.training_selection_policy) + "\","
    buffer += "\"positive_selection_policy\":\""
    buffer += json_escape(summary.positive_selection_policy) + "\","
    buffer += "\"training_query_count\":" + String(summary.training_query_count) + ","
    buffer += "\"evaluation_query_count\":"
    buffer += String(summary.evaluation_query_count) + ","
    buffer += "\"final_k\":" + String(summary.final_k) + ","
    buffer += "\"candidate_k\":" + String(summary.candidate_k) + ","
    buffer += "\"query_vector_budget\":" + String(summary.query_vector_budget) + ","
    buffer += "\"search_cluster_top_k_per_query_token\":"
    buffer += String(summary.search_cluster_top_k_per_query_token) + ","
    buffer += "\"search_beam_width\":" + String(summary.search_beam_width) + ","
    buffer += "\"fine_cluster_count\":" + String(summary.fine_cluster_count) + ","
    buffer += "\"coarse_cluster_count\":" + String(summary.coarse_cluster_count) + ","
    buffer += "\"cluster_cutoff\":" + String(summary.cluster_cutoff) + ","
    buffer += "\"construction_neighbor_count\":"
    buffer += String(summary.construction_neighbor_count) + ","
    buffer += "\"degree_limit\":" + String(summary.degree_limit) + ","
    buffer += "\"build_cluster_top_k_per_query_token\":"
    buffer += String(summary.build_cluster_top_k_per_query_token) + ","
    buffer += "\"adaptive_cluster_cutoff_enabled\":"
    if summary.adaptive_cluster_cutoff_enabled:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"adaptive_cluster_cutoff_max\":"
    buffer += String(summary.adaptive_cluster_cutoff_max) + ","
    buffer += "\"adaptive_label_policy\":\""
    buffer += json_escape(summary.adaptive_label_policy) + "\","
    buffer += "\"shortcuts_enabled\":"
    if summary.shortcuts_enabled:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"shortcut_candidate_k\":"
    buffer += String(summary.shortcut_candidate_k) + ","
    buffer += "\"shortcut_beam_width\":"
    buffer += String(summary.shortcut_beam_width) + ","
    buffer += "\"artifact_document_count\":"
    buffer += String(summary.artifact_document_count) + ","
    buffer += "\"artifact_cluster_count\":"
    buffer += String(summary.artifact_cluster_count) + ","
    buffer += "\"artifact_graph_edge_count\":"
    buffer += String(summary.artifact_graph_edge_count) + ","
    buffer += "\"artifact_shortcut_edge_count\":"
    buffer += String(summary.artifact_shortcut_edge_count) + ","
    buffer += "\"artifact_entry_point_count\":"
    buffer += String(summary.artifact_entry_point_count) + ","
    buffer += "\"artifact_quantization_centroid_count\":"
    buffer += String(summary.artifact_quantization_centroid_count) + ","
    buffer += "\"mean_document_profile_limit\":"
    buffer += String(summary.mean_document_profile_limit) + ","
    buffer += "\"min_document_profile_limit\":"
    buffer += String(summary.min_document_profile_limit) + ","
    buffer += "\"max_document_profile_limit\":"
    buffer += String(summary.max_document_profile_limit) + ","
    buffer += "\"mean_training_positive_profile_limit\":"
    buffer += String(summary.mean_training_positive_profile_limit) + ","
    buffer += "\"mean_evaluation_positive_profile_limit\":"
    buffer += String(summary.mean_evaluation_positive_profile_limit) + ","
    buffer += "\"training_positive_profile_hit_rate\":"
    buffer += String(summary.training_positive_profile_hit_rate) + ","
    buffer += "\"evaluation_positive_profile_hit_rate\":"
    buffer += String(summary.evaluation_positive_profile_hit_rate) + ","
    buffer += "\"evaluation_positive_entry_rate\":"
    buffer += String(summary.evaluation_positive_entry_rate) + ","
    buffer += "\"evaluation_positive_reachable_rate\":"
    buffer += String(summary.evaluation_positive_reachable_rate) + ","
    buffer += "\"adaptive_training_label_count\":"
    buffer += String(summary.adaptive_training_label_count) + ","
    buffer += "\"adaptive_training_label_mean\":"
    buffer += String(summary.adaptive_training_label_mean) + ","
    buffer += "\"adaptive_training_label_min\":"
    buffer += String(summary.adaptive_training_label_min) + ","
    buffer += "\"adaptive_training_label_max\":"
    buffer += String(summary.adaptive_training_label_max) + ","
    buffer += "\"adaptive_training_label_one_share\":"
    buffer += String(summary.adaptive_training_label_one_share) + ","
    buffer += "\"adaptive_predicted_profile_limit_count\":"
    buffer += String(summary.adaptive_predicted_profile_limit_count) + ","
    buffer += "\"adaptive_predicted_profile_limit_mean\":"
    buffer += String(summary.adaptive_predicted_profile_limit_mean) + ","
    buffer += "\"adaptive_predicted_profile_limit_min\":"
    buffer += String(summary.adaptive_predicted_profile_limit_min) + ","
    buffer += "\"adaptive_predicted_profile_limit_max\":"
    buffer += String(summary.adaptive_predicted_profile_limit_max) + ","
    buffer += "\"adaptive_predicted_profile_limit_one_share\":"
    buffer += String(summary.adaptive_predicted_profile_limit_one_share) + ","
    buffer += "\"mean_candidate_generation_seconds\":"
    buffer += String(summary.mean_candidate_generation_seconds) + ","
    buffer += "\"mean_search_seconds\":" + String(summary.mean_search_seconds) + ","
    buffer += "\"mean_candidate_recall_at_final_k\":"
    buffer += String(summary.mean_candidate_recall_at_final_k) + ","
    buffer += "\"mean_ndcg_at_k\":" + String(summary.mean_ndcg_at_k) + ","
    buffer += "\"mean_reciprocal_rank\":"
    buffer += String(summary.mean_reciprocal_rank) + ","
    buffer += "\"mean_recall_at_k\":" + String(summary.mean_recall_at_k) + ","
    buffer += "\"success_rate_at_k\":" + String(summary.success_rate_at_k) + ","
    buffer += "\"stage1_vector_count\":" + String(summary.stage1_vector_count) + ","
    buffer += "\"stage1_token_count\":" + String(summary.stage1_token_count) + ","
    buffer += "\"stage1_byte_size\":" + String(summary.stage1_byte_size) + ","
    buffer += "\"stage1_bytes_per_document\":"
    buffer += String(summary.stage1_bytes_per_document) + ","
    buffer += "\"stage1_bytes_per_vector\":"
    buffer += String(summary.stage1_bytes_per_vector) + ","
    buffer += "\"mean_stage1_graph_visited_vertex_count\":"
    buffer += String(summary.mean_stage1_graph_visited_vertex_count) + ","
    buffer += "\"mean_stage1_graph_expanded_edge_count\":"
    buffer += String(summary.mean_stage1_graph_expanded_edge_count) + ","
    buffer += "\"mean_stage1_graph_visited_cluster_count\":"
    buffer += String(summary.mean_stage1_graph_visited_cluster_count) + ","
    buffer += "\"mean_stage1_graph_entry_point_count\":"
    buffer += String(summary.mean_stage1_graph_entry_point_count) + ","
    buffer += "\"mean_stage1_graph_max_frontier_size\":"
    buffer += String(summary.mean_stage1_graph_max_frontier_size)
    buffer += "}"


def gem_heldout_ablation_summary_json(
    read summary: GemHeldoutAblationSummary
) -> String:
    var buffer = String()
    append_gem_heldout_ablation_summary_json(buffer, summary)
    return buffer^


def gem_heldout_ablation_summaries_json(
    read summaries: List[GemHeldoutAblationSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","
        append_gem_heldout_ablation_summary_json(buffer, summaries[index])

    buffer += "]"
    return buffer^
