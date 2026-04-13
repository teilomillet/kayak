from std.collections import List

from kayak.collections import (
    SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
    ResolvedCollectionSnapshot,
    loaded_search_artifact_stored_gem_graph_index,
    loaded_segment_has_search_artifact,
    loaded_segment_search_artifact,
)
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, match_all_filter
from kayak.index import (
    build_quantization_distance_matrix,
    document_profile_intersects_clusters,
    quantize_query_codes,
    quantized_chamfer_distance_for_document,
    query_entry_doc_indices,
    query_relevant_cluster_ids,
)
from kayak.numeric import MetricScalar, ScoreScalar
from kayak.runtime import ExactScoringBackend
from kayak.storage import StoredGemGraphIndex

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .graph_search_counters import GraphSearchCounters
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def list_contains_int(read values: List[Int], target: Int) -> Bool:
    for value in values:
        if value == target:
            return True
    return False


def max_metric_scalar() -> MetricScalar:
    return MetricScalar(1.0e300)


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
                var swap_id = ids[index - 1]
                ids[index - 1] = ids[index]
                ids[index] = swap_id
                var swap_distance = distances[index - 1]
                distances[index - 1] = distances[index]
                distances[index] = swap_distance
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
        raise Error("cannot remove from an empty queue")

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


def segment_hits_for_gem_graph(
    read query: EncodedQuery,
    segment_id: String,
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
        return GemGraphSegmentSearchResult(
            List[CollectionHit](),
            GraphSearchCounters(0, 0, visited_cluster_count, 0, 0),
        )

    var entry_docs = query_entry_doc_indices(index, relevant_clusters)
    var entry_point_count = len(entry_docs)
    if len(entry_docs) == 0:
        return GemGraphSegmentSearchResult(
            List[CollectionHit](),
            GraphSearchCounters(0, 0, visited_cluster_count, entry_point_count, 0),
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


def candidate_generation_for_graph_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    _ = backend
    _ = filter_expression

    var hits = List[CollectionHit]()
    var token_count = 0
    var vector_count = 0
    var byte_size = 0
    var visited_vertex_count = 0
    var expanded_edge_count = 0
    var visited_cluster_count = 0
    var entry_point_count = 0
    var max_frontier_size = 0

    for segment in snapshot.segments:
        if not loaded_segment_has_search_artifact(
            segment, SEARCH_ARTIFACT_FAMILY_GEM_GRAPH
        ):
            raise Error(
                "gem_graph stage-1 requires a gem graph sidecar for every segment"
            )

        var stored_gem_graph = loaded_search_artifact_stored_gem_graph_index(
            loaded_segment_search_artifact(
                segment,
                SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
            )
        )
        token_count += stored_gem_graph.index.total_token_count
        vector_count += (
            stored_gem_graph.index.total_token_count
            + stored_gem_graph.quantization_centroid_count
            + stored_gem_graph.cluster_count
        )
        byte_size += stored_gem_graph.artifact_byte_size

        var segment_result = segment_hits_for_gem_graph(
            query,
            segment.manifest.segment_id.value,
            stored_gem_graph,
            plan.candidate_budget.candidate_k,
            plan.candidate_generator.cluster_top_k_per_query_token,
            plan.candidate_generator.beam_width,
        )
        visited_vertex_count += (
            segment_result.graph_search_counters.visited_vertex_count
        )
        expanded_edge_count += (
            segment_result.graph_search_counters.expanded_edge_count
        )
        visited_cluster_count += (
            segment_result.graph_search_counters.visited_cluster_count
        )
        entry_point_count += segment_result.graph_search_counters.entry_point_count
        if segment_result.graph_search_counters.max_frontier_size > max_frontier_size:
            max_frontier_size = (
                segment_result.graph_search_counters.max_frontier_size
            )

        for hit in segment_result.hits:
            insert_descending_collection_hit(
                hits,
                hit.copy(),
                plan.candidate_budget.candidate_k,
            )

    return CandidateSet(
        plan.candidate_generator.kind.copy(),
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        token_count,
        vector_count,
        byte_size,
        GraphSearchCounters(
            visited_vertex_count,
            expanded_edge_count,
            visited_cluster_count,
            entry_point_count,
            max_frontier_size,
        ),
    )
