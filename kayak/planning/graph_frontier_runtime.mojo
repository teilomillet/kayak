# Graph-family frontier scheduling runtime.
#
# Owns:
# - frontier-policy-specific GEM graph traversal
# - graph-native counters for per-segment graph search
#
# Does not own:
# - collection-level candidate aggregation
# - search-plan construction or planner defaults

from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import (
    build_quantization_distance_matrix,
    document_profile_intersects_clusters,
    quantize_query_codes,
    quantized_chamfer_distance_for_document,
    query_entry_doc_indices,
    query_relevant_cluster_ids,
)
from kayak.numeric import MetricScalar, ScoreScalar
from kayak.storage import StoredGemGraphIndex

from .collection_hit import CollectionHit
from .graph_frontier_policy import (
    DEFAULT_GRAPH_FRONTIER_POLICY_KIND,
    GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY,
    GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND,
    GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY,
    require_graph_frontier_policy_kind,
)
from .graph_search_counters import GraphSearchCounters


def list_contains_int(read values: List[Int], target: Int) -> Bool:
    for value in values:
        if value == target:
            return True
    return False


def max_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0e300)


def insert_ascending_metric(
    mut ids: List[Int],
    mut distances: List[MetricScalar],
    item_id: Int,
    distance: MetricScalar,
    k: Int,
):
    if k == 0:
        return

    for index in range(len(ids)):
        if ids[index] == item_id:
            if distance >= distances[index]:
                return
            distances[index] = distance
            var current = index
            while current > 0 and distances[current] < distances[current - 1]:
                var swap_id = ids[current - 1]
                ids[current - 1] = ids[current]
                ids[current] = swap_id
                var swap_distance = distances[current - 1]
                distances[current - 1] = distances[current]
                distances[current] = swap_distance
                current -= 1
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


def remove_front(
    mut ids: List[Int], mut distances: List[MetricScalar]
) raises -> Int:
    if len(ids) == 0:
        raise Error("cannot remove from an empty frontier")

    var head_id = ids[0]
    for index in range(1, len(ids)):
        ids[index - 1] = ids[index]
        distances[index - 1] = distances[index]
    _ = ids.pop()
    _ = distances.pop()
    return head_id


def metric_score(distance: MetricScalar) -> ScoreScalar:
    return ScoreScalar(0.0) - ScoreScalar(distance)


struct GemGraphSegmentSearchResult(Copyable):
    var hits: List[CollectionHit]
    var graph_search_counters: GraphSearchCounters

    def __init__(
        out self,
        hits: List[CollectionHit],
        graph_search_counters: GraphSearchCounters,
    ):
        self.hits = hits.copy()
        self.graph_search_counters = graph_search_counters.copy()


def empty_gem_graph_segment_search_result(
    visited_cluster_count: Int = 0,
    entry_point_count: Int = 0,
) raises -> GemGraphSegmentSearchResult:
    return GemGraphSegmentSearchResult(
        List[CollectionHit](),
        GraphSearchCounters(
            0,
            0,
            visited_cluster_count,
            entry_point_count,
            0,
        ),
    )


def prune_tau_exhausted_frontiers(
    mut queue_doc_indices: List[List[Int]],
    mut queue_distances: List[List[MetricScalar]],
    read result_distances: List[MetricScalar],
    effective_beam_width: Int,
    mut current_frontier_size: Int,
):
    if len(result_distances) < effective_beam_width:
        return

    var tau = result_distances[len(result_distances) - 1]
    for queue_index in range(len(queue_doc_indices)):
        if len(queue_doc_indices[queue_index]) == 0:
            continue
        if queue_distances[queue_index][0] <= tau:
            continue
        current_frontier_size -= len(queue_doc_indices[queue_index])
        queue_doc_indices[queue_index] = List[Int]()
        queue_distances[queue_index] = List[MetricScalar]()


def best_frontier_queue_index(
    read queue_doc_indices: List[List[Int]],
    read queue_distances: List[List[MetricScalar]],
) -> Int:
    var best_queue_index = -1
    var best_distance = max_metric_scalar()
    for queue_index in range(len(queue_doc_indices)):
        if len(queue_doc_indices[queue_index]) == 0:
            continue
        var candidate_distance = queue_distances[queue_index][0]
        if best_queue_index < 0 or candidate_distance < best_distance:
            best_queue_index = queue_index
            best_distance = candidate_distance
    return best_queue_index


def best_quota_frontier_queue_index(
    read queue_doc_indices: List[List[Int]],
    read queue_distances: List[List[MetricScalar]],
    read expansions_in_round: List[Int],
    per_round_expansion_quota: Int,
) -> Int:
    var best_queue_index = -1
    var best_distance = max_metric_scalar()
    for queue_index in range(len(queue_doc_indices)):
        if len(queue_doc_indices[queue_index]) == 0:
            continue
        if expansions_in_round[queue_index] >= per_round_expansion_quota:
            continue
        var candidate_distance = queue_distances[queue_index][0]
        if best_queue_index < 0 or candidate_distance < best_distance:
            best_queue_index = queue_index
            best_distance = candidate_distance
    return best_queue_index


def reset_frontier_round_expansion_counts(mut expansions_in_round: List[Int]):
    for queue_index in range(len(expansions_in_round)):
        expansions_in_round[queue_index] = 0


def segment_hits_for_gem_graph_local_per_entry(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read stored_gem_graph: StoredGemGraphIndex,
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
) raises -> GemGraphSegmentSearchResult:
    if candidate_k <= 0:
        return GemGraphSegmentSearchResult(
            List[CollectionHit](),
            GraphSearchCounters(),
        )

    var index = stored_gem_graph.index.copy()
    var relevant_clusters = query_relevant_cluster_ids(
        query,
        index,
        cluster_top_k_per_query_token,
    )
    var visited_cluster_count = len(relevant_clusters)
    if len(relevant_clusters) == 0:
        return empty_gem_graph_segment_search_result(visited_cluster_count, 0)

    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    var entry_point_count = len(entry_docs)
    if len(entry_docs) == 0:
        return empty_gem_graph_segment_search_result(
            visited_cluster_count,
            entry_point_count,
        )

    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var query_codes = quantize_query_codes(query, index)

    var result_doc_indices = List[Int]()
    var result_distances = List[MetricScalar]()
    var visited = List[Int]()
    var queue_doc_indices = List[List[Int]]()
    var queue_distances = List[List[MetricScalar]]()
    var visited_vertex_count = 0
    var expanded_edge_count = 0
    var current_frontier_size = 0
    var max_frontier_size = 0

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
        if not list_contains_int(visited, entry_doc):
            visited_vertex_count += 1
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
        current_frontier_size += len(local_ids)
        if current_frontier_size > max_frontier_size:
            max_frontier_size = current_frontier_size
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

            if len(result_distances) >= effective_beam_width and current_distance > tau:
                current_frontier_size -= len(queue_doc_indices[queue_index])
                queue_doc_indices[queue_index] = List[Int]()
                queue_distances[queue_index] = List[MetricScalar]()
                continue

            var current_doc = remove_front(
                queue_doc_indices[queue_index],
                queue_distances[queue_index],
            )
            current_frontier_size -= 1
            var neighbor_start = index.neighbor_offsets[current_doc]
            var neighbor_stop = index.neighbor_offsets[current_doc + 1]
            for neighbor_index in range(neighbor_start, neighbor_stop):
                expanded_edge_count += 1
                var neighbor_doc = index.neighbor_doc_indices[neighbor_index]
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
                var previous_queue_len = len(queue_doc_indices[queue_index])
                insert_ascending_metric(
                    queue_doc_indices[queue_index],
                    queue_distances[queue_index],
                    neighbor_doc,
                    neighbor_distance,
                    effective_beam_width,
                )
                if len(queue_doc_indices[queue_index]) > previous_queue_len:
                    current_frontier_size += (
                        len(queue_doc_indices[queue_index]) - previous_queue_len
                    )
                    if current_frontier_size > max_frontier_size:
                        max_frontier_size = current_frontier_size
                visited_vertex_count += 1
                insert_ascending_metric(
                    result_doc_indices,
                    result_distances,
                    neighbor_doc,
                    neighbor_distance,
                    effective_beam_width,
                )
        if not any_non_empty:
            break

    var hits = List[CollectionHit]()
    var limit = candidate_k
    if len(result_doc_indices) < limit:
        limit = len(result_doc_indices)
    for hit_index in range(limit):
        hits.append(
            CollectionHit(
                segment_id.copy(),
                index.doc_ids[result_doc_indices[hit_index]].copy(),
                metric_score(result_distances[hit_index]),
                segment_index,
                result_doc_indices[hit_index],
            )
        )
    return GemGraphSegmentSearchResult(
        hits^,
        GraphSearchCounters(
            visited_vertex_count,
            expanded_edge_count,
            visited_cluster_count,
            entry_point_count,
            max_frontier_size,
        ),
    )


def segment_hits_for_gem_graph_global_best_first(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read stored_gem_graph: StoredGemGraphIndex,
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
) raises -> GemGraphSegmentSearchResult:
    if candidate_k <= 0:
        return GemGraphSegmentSearchResult(
            List[CollectionHit](),
            GraphSearchCounters(),
        )

    var index = stored_gem_graph.index.copy()
    var relevant_clusters = query_relevant_cluster_ids(
        query,
        index,
        cluster_top_k_per_query_token,
    )
    var visited_cluster_count = len(relevant_clusters)
    if len(relevant_clusters) == 0:
        return empty_gem_graph_segment_search_result(visited_cluster_count, 0)

    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    var entry_point_count = len(entry_docs)
    if len(entry_docs) == 0:
        return empty_gem_graph_segment_search_result(
            visited_cluster_count,
            entry_point_count,
        )

    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var query_codes = quantize_query_codes(query, index)
    var effective_beam_width = beam_width
    if effective_beam_width < candidate_k:
        effective_beam_width = candidate_k

    var result_doc_indices = List[Int]()
    var result_distances = List[MetricScalar]()
    var frontier_doc_indices = List[Int]()
    var frontier_distances = List[MetricScalar]()
    var visited = List[Int]()
    var visited_vertex_count = 0
    var expanded_edge_count = 0
    var max_frontier_size = 0

    for entry_doc in entry_docs:
        var entry_distance = quantized_chamfer_distance_for_document(
            query_codes,
            index,
            entry_doc,
            distance_matrix,
        )
        if not list_contains_int(visited, entry_doc):
            visited.append(entry_doc)
            visited_vertex_count += 1
        insert_ascending_metric(
            frontier_doc_indices,
            frontier_distances,
            entry_doc,
            entry_distance,
            effective_beam_width,
        )
        insert_ascending_metric(
            result_doc_indices,
            result_distances,
            entry_doc,
            entry_distance,
            effective_beam_width,
        )
        if len(frontier_doc_indices) > max_frontier_size:
            max_frontier_size = len(frontier_doc_indices)
    while len(frontier_doc_indices) > 0:
        var current_distance = frontier_distances[0]
        var tau = max_metric_scalar()
        if len(result_distances) >= effective_beam_width:
            tau = result_distances[len(result_distances) - 1]
        if len(result_distances) >= effective_beam_width and current_distance > tau:
            break

        var current_doc = remove_front(frontier_doc_indices, frontier_distances)
        var neighbor_start = index.neighbor_offsets[current_doc]
        var neighbor_stop = index.neighbor_offsets[current_doc + 1]
        for neighbor_index in range(neighbor_start, neighbor_stop):
            expanded_edge_count += 1
            var neighbor_doc = index.neighbor_doc_indices[neighbor_index]
            if list_contains_int(visited, neighbor_doc):
                continue
            if not document_profile_intersects_clusters(
                index,
                neighbor_doc,
                relevant_clusters,
            ):
                continue
            var neighbor_distance = quantized_chamfer_distance_for_document(
                query_codes,
                index,
                neighbor_doc,
                distance_matrix,
            )
            visited.append(neighbor_doc)
            visited_vertex_count += 1
            insert_ascending_metric(
                frontier_doc_indices,
                frontier_distances,
                neighbor_doc,
                neighbor_distance,
                effective_beam_width,
            )
            if len(frontier_doc_indices) > max_frontier_size:
                max_frontier_size = len(frontier_doc_indices)
            insert_ascending_metric(
                result_doc_indices,
                result_distances,
                neighbor_doc,
                neighbor_distance,
                effective_beam_width,
            )
    var hits = List[CollectionHit]()
    var limit = candidate_k
    if len(result_doc_indices) < limit:
        limit = len(result_doc_indices)
    for hit_index in range(limit):
        hits.append(
            CollectionHit(
                segment_id.copy(),
                index.doc_ids[result_doc_indices[hit_index]].copy(),
                metric_score(result_distances[hit_index]),
                segment_index,
                result_doc_indices[hit_index],
            )
        )
    return GemGraphSegmentSearchResult(
        hits^,
        GraphSearchCounters(
            visited_vertex_count,
            expanded_edge_count,
            visited_cluster_count,
            entry_point_count,
            max_frontier_size,
        ),
    )


def segment_hits_for_gem_graph_best_head_per_entry(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read stored_gem_graph: StoredGemGraphIndex,
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
    per_round_expansion_quota: Int,
) raises -> GemGraphSegmentSearchResult:
    if candidate_k <= 0:
        return GemGraphSegmentSearchResult(
            List[CollectionHit](),
            GraphSearchCounters(),
        )

    var index = stored_gem_graph.index.copy()
    var relevant_clusters = query_relevant_cluster_ids(
        query,
        index,
        cluster_top_k_per_query_token,
    )
    var visited_cluster_count = len(relevant_clusters)
    if len(relevant_clusters) == 0:
        return empty_gem_graph_segment_search_result(visited_cluster_count, 0)

    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    var entry_point_count = len(entry_docs)
    if len(entry_docs) == 0:
        return empty_gem_graph_segment_search_result(
            visited_cluster_count,
            entry_point_count,
        )

    var distance_matrix = build_quantization_distance_matrix(
        index.quantization_centroids
    )
    var query_codes = quantize_query_codes(query, index)

    var result_doc_indices = List[Int]()
    var result_distances = List[MetricScalar]()
    var visited = List[Int]()
    var queue_doc_indices = List[List[Int]]()
    var queue_distances = List[List[MetricScalar]]()
    var queue_expansions_in_round = List[Int]()
    var visited_vertex_count = 0
    var expanded_edge_count = 0
    var current_frontier_size = 0
    var max_frontier_size = 0

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
        if not list_contains_int(visited, entry_doc):
            visited_vertex_count += 1
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
        current_frontier_size += len(local_ids)
        if current_frontier_size > max_frontier_size:
            max_frontier_size = current_frontier_size
        queue_doc_indices.append(local_ids^)
        queue_distances.append(local_distances^)
        queue_expansions_in_round.append(0)

    while True:
        prune_tau_exhausted_frontiers(
            queue_doc_indices,
            queue_distances,
            result_distances,
            effective_beam_width,
            current_frontier_size,
        )
        var queue_index = best_frontier_queue_index(
            queue_doc_indices,
            queue_distances,
        )
        if per_round_expansion_quota > 0:
            queue_index = best_quota_frontier_queue_index(
                queue_doc_indices,
                queue_distances,
                queue_expansions_in_round,
                per_round_expansion_quota,
            )
            if queue_index < 0:
                reset_frontier_round_expansion_counts(queue_expansions_in_round)
                queue_index = best_quota_frontier_queue_index(
                    queue_doc_indices,
                    queue_distances,
                    queue_expansions_in_round,
                    per_round_expansion_quota,
                )
        if queue_index < 0:
            break
        if per_round_expansion_quota > 0:
            queue_expansions_in_round[queue_index] += 1

        var current_doc = remove_front(
            queue_doc_indices[queue_index],
            queue_distances[queue_index],
        )
        current_frontier_size -= 1
        var neighbor_start = index.neighbor_offsets[current_doc]
        var neighbor_stop = index.neighbor_offsets[current_doc + 1]
        for neighbor_index in range(neighbor_start, neighbor_stop):
            expanded_edge_count += 1
            var neighbor_doc = index.neighbor_doc_indices[neighbor_index]
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
            var previous_queue_len = len(queue_doc_indices[queue_index])
            insert_ascending_metric(
                queue_doc_indices[queue_index],
                queue_distances[queue_index],
                neighbor_doc,
                neighbor_distance,
                effective_beam_width,
            )
            if len(queue_doc_indices[queue_index]) > previous_queue_len:
                current_frontier_size += (
                    len(queue_doc_indices[queue_index]) - previous_queue_len
                )
                if current_frontier_size > max_frontier_size:
                    max_frontier_size = current_frontier_size
            visited_vertex_count += 1
            insert_ascending_metric(
                result_doc_indices,
                result_distances,
                neighbor_doc,
                neighbor_distance,
                effective_beam_width,
            )

    var hits = List[CollectionHit]()
    var limit = candidate_k
    if len(result_doc_indices) < limit:
        limit = len(result_doc_indices)
    for hit_index in range(limit):
        hits.append(
            CollectionHit(
                segment_id.copy(),
                index.doc_ids[result_doc_indices[hit_index]].copy(),
                metric_score(result_distances[hit_index]),
                segment_index,
                result_doc_indices[hit_index],
            )
        )
    return GemGraphSegmentSearchResult(
        hits^,
        GraphSearchCounters(
            visited_vertex_count,
            expanded_edge_count,
            visited_cluster_count,
            entry_point_count,
            max_frontier_size,
        ),
    )


def segment_hits_for_gem_graph(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read stored_gem_graph: StoredGemGraphIndex,
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
) raises -> GemGraphSegmentSearchResult:
    return segment_hits_for_gem_graph(
        query,
        segment_id,
        segment_index,
        stored_gem_graph,
        candidate_k,
        cluster_top_k_per_query_token,
        beam_width,
        DEFAULT_GRAPH_FRONTIER_POLICY_KIND,
    )


def segment_hits_for_gem_graph(
    read query: EncodedQuery,
    segment_id: String,
    segment_index: Int,
    read stored_gem_graph: StoredGemGraphIndex,
    candidate_k: Int,
    cluster_top_k_per_query_token: Int,
    beam_width: Int,
    graph_frontier_policy_kind: String,
) raises -> GemGraphSegmentSearchResult:
    var policy_kind = require_graph_frontier_policy_kind(
        graph_frontier_policy_kind
    )
    if policy_kind == GRAPH_FRONTIER_POLICY_KIND_LOCAL_PER_ENTRY:
        return segment_hits_for_gem_graph_local_per_entry(
            query,
            segment_id,
            segment_index,
            stored_gem_graph,
            candidate_k,
            cluster_top_k_per_query_token,
            beam_width,
        )
    if policy_kind == GRAPH_FRONTIER_POLICY_KIND_GLOBAL_BEST_FIRST:
        return segment_hits_for_gem_graph_global_best_first(
            query,
            segment_id,
            segment_index,
            stored_gem_graph,
            candidate_k,
            cluster_top_k_per_query_token,
            beam_width,
        )
    if policy_kind == GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_PER_ENTRY:
        return segment_hits_for_gem_graph_best_head_per_entry(
            query,
            segment_id,
            segment_index,
            stored_gem_graph,
            candidate_k,
            cluster_top_k_per_query_token,
            beam_width,
            0,
        )
    if policy_kind == GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_FAIR_ROUND:
        return segment_hits_for_gem_graph_best_head_per_entry(
            query,
            segment_id,
            segment_index,
            stored_gem_graph,
            candidate_k,
            cluster_top_k_per_query_token,
            beam_width,
            1,
        )
    if policy_kind == GRAPH_FRONTIER_POLICY_KIND_HYBRID_BEST_HEAD_QUOTA2_ROUND:
        return segment_hits_for_gem_graph_best_head_per_entry(
            query,
            segment_id,
            segment_index,
            stored_gem_graph,
            candidate_k,
            cluster_top_k_per_query_token,
            beam_width,
            2,
        )

    raise Error("unsupported graph_frontier_policy_kind: " + policy_kind)
