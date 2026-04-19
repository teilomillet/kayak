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
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .graph_frontier_runtime import (
    segment_hits_for_gem_graph,
)
from .graph_search_counters import (
    GraphSearchCounters,
    accumulate_graph_search_counters,
)
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_graph_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    _ = backend
    if not filter_expression.is_match_all():
        raise Error("gem_graph stage-1 currently supports only match_all filters")

    var hits = List[CollectionHit]()
    var token_count = 0
    var vector_count = 0
    var byte_size = 0
    var graph_search_counters = GraphSearchCounters()

    for segment_index in range(len(snapshot.segments)):
        if not loaded_segment_has_search_artifact(
            snapshot.segments[segment_index],
            SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
        ):
            raise Error(
                "gem_graph stage-1 requires a gem graph sidecar for every segment"
            )

        var stored_gem_graph = loaded_search_artifact_stored_gem_graph_index(
            loaded_segment_search_artifact(
                snapshot.segments[segment_index],
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
            snapshot.segments[segment_index].manifest.segment_id.value,
            segment_index,
            stored_gem_graph,
            plan.candidate_budget.candidate_k,
            plan.candidate_generator.cluster_top_k_per_query_token,
            plan.candidate_generator.beam_width,
            plan.candidate_generator.graph_frontier_policy_kind,
        )
        graph_search_counters = accumulate_graph_search_counters(
            graph_search_counters,
            segment_result.graph_search_counters,
        )

        for hit in segment_result.hits:
            insert_descending_collection_hit(
                hits,
                hit.copy(),
                plan.candidate_budget.candidate_k,
            )

    return CandidateSet(
        plan.candidate_generator,
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        token_count,
        vector_count,
        byte_size,
        graph_search_counters,
    )
