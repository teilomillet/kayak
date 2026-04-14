from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, match_all_filter
from kayak.numeric import MetricScalar
from kayak.runtime import ExactScoringBackend
from kayak.search import SearchHit

from .candidate_generator import (
    CANDIDATE_GENERATOR_FAMILY_CENTROID,
    CANDIDATE_GENERATOR_FAMILY_EXACT,
    CANDIDATE_GENERATOR_FAMILY_GRAPH,
    CANDIDATE_GENERATOR_FAMILY_PROXY,
)
from .candidate_set import CandidateSet
from .collection_hit import CollectionHit, to_search_hit
from .centroid_candidate_generation_workspace import (
    MutableCentroidCandidateGenerationWorkspace,
)
from .execution_centroid_family import (
    candidate_generation_for_centroid_family,
    candidate_generation_for_centroid_family_with_workspace,
)
from .execution_exact_family import candidate_generation_for_exact_family
from .execution_graph_family import candidate_generation_for_graph_family
from .execution_proxy_family import candidate_generation_for_proxy_family
from .execution_stage2 import stage2_result_for_plan
from .execution_stage3 import stage3_result_for_plan
from .search_plan import SearchPlan
from .stage2_result import Stage2Result


def candidate_generation_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    var workspace = MutableCentroidCandidateGenerationWorkspace()
    return candidate_generation_for_plan_with_workspace(
        backend,
        query,
        snapshot,
        plan,
        workspace,
        filter_expression,
    )


def candidate_generation_for_plan_with_workspace[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    mut workspace: MutableCentroidCandidateGenerationWorkspace,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    if plan.candidate_generator.family == CANDIDATE_GENERATOR_FAMILY_EXACT:
        return candidate_generation_for_exact_family(
            backend,
            query,
            snapshot,
            plan,
            filter_expression,
        )

    if plan.candidate_generator.family == CANDIDATE_GENERATOR_FAMILY_PROXY:
        return candidate_generation_for_proxy_family(
            backend,
            query,
            snapshot,
            plan,
            filter_expression,
        )

    if plan.candidate_generator.family == CANDIDATE_GENERATOR_FAMILY_CENTROID:
        return candidate_generation_for_centroid_family_with_workspace(
            backend,
            query,
            snapshot,
            plan,
            workspace,
            filter_expression,
        )

    if plan.candidate_generator.family == CANDIDATE_GENERATOR_FAMILY_GRAPH:
        return candidate_generation_for_graph_family(
            backend,
            query,
            snapshot,
            plan,
            filter_expression,
        )

    raise Error(
        "unsupported candidate generator family: "
        + plan.candidate_generator.family
    )


def collection_hit_matches(
    read lhs: CollectionHit, read rhs: CollectionHit
) -> Bool:
    return lhs.segment_id == rhs.segment_id and lhs.doc_id == rhs.doc_id


def candidate_recall_at_final_k(
    read candidate_set: CandidateSet, final_hits: List[CollectionHit]
) -> MetricScalar:
    if len(final_hits) == 0:
        return MetricScalar(0.0)

    var found_count = 0

    for final_hit in final_hits:
        for candidate_hit in candidate_set.hits:
            if collection_hit_matches(final_hit, candidate_hit):
                found_count += 1
                break

    return MetricScalar(found_count) / MetricScalar(len(final_hits))


def final_hits_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) raises -> Stage2Result:
    var reference_result = stage2_result_for_plan(
        backend,
        query,
        query_text,
        snapshot,
        candidate_set,
        plan,
    )
    return stage3_result_for_plan(
        query_text,
        snapshot,
        reference_result,
        plan,
    )


def search_collection_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
    query_text: String = "",
) raises -> List[CollectionHit]:
    var workspace = MutableCentroidCandidateGenerationWorkspace()
    return search_collection_for_plan_with_workspace(
        backend,
        query,
        snapshot,
        plan,
        workspace,
        filter_expression,
        query_text,
    )


def search_collection_for_plan_with_workspace[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    mut workspace: MutableCentroidCandidateGenerationWorkspace,
    read filter_expression: FilterExpression = match_all_filter(),
    query_text: String = "",
) raises -> List[CollectionHit]:
    var candidate_set = candidate_generation_for_plan_with_workspace(
        backend,
        query,
        snapshot,
        plan,
        workspace,
        filter_expression,
    )
    return final_hits_for_plan(
        backend,
        query,
        query_text,
        snapshot,
        candidate_set,
        plan,
    ).final_hits.copy()


def final_hits_to_search_hits(
    read hits: List[CollectionHit]
) -> List[SearchHit]:
    var search_hits = List[SearchHit]()

    for hit in hits:
        search_hits.append(to_search_hit(hit))

    return search_hits^
