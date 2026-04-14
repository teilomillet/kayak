# Stage-2 reference scoring dispatch over one candidate window.

from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.runtime import ExactCpuBackend
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .exact_stage import exact_rerank_candidates_for_plan
from .search_plan import SearchPlan
from .stage2_result import Stage2Result


def count_unique_segment_ids(read hits: List[CollectionHit]) -> Int:
    var unique_segment_ids = List[String]()

    for hit in hits:
        var already_seen = False
        for unique_segment_id in unique_segment_ids:
            if unique_segment_id == hit.segment_id:
                already_seen = True
                break

        if not already_seen:
            unique_segment_ids.append(hit.segment_id.copy())

    return len(unique_segment_ids)


def noop_topk_stage2_result(
    read hits: List[CollectionHit], output_k: Int
) raises -> Stage2Result:
    var final_hits = List[CollectionHit]()
    var limit = output_k
    if limit > len(hits):
        limit = len(hits)

    for index in range(limit):
        final_hits.append(hits[index].copy())

    return Stage2Result(
        final_hits^,
        count_unique_segment_ids(hits),
        len(hits),
        0,
        0,
        0,
    )


def reference_stage_output_k(
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) -> Int:
    if plan.stage3_verifier.kind == "none":
        return plan.candidate_budget.final_k

    var limit = plan.candidate_budget.candidate_k
    if limit > len(candidate_set.hits):
        limit = len(candidate_set.hits)
    return limit


def stage2_result_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) raises -> Stage2Result:
    _ = query_text
    var output_k = reference_stage_output_k(candidate_set, plan)

    if plan.stage2_reference_operator.kind == "noop_topk":
        return noop_topk_stage2_result(
            candidate_set.hits,
            output_k,
        )

    if plan.stage2_reference_operator.kind == "exact_late_interaction":
        return exact_rerank_candidates_for_plan(
            backend,
            query,
            snapshot,
            candidate_set.hits,
            output_k,
        )

    raise Error(
        "unsupported stage2 reference operator kind: "
        + plan.stage2_reference_operator.kind
    )


def stage2_result_for_plan(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read candidate_set: CandidateSet,
    read plan: SearchPlan,
) raises -> Stage2Result:
    _ = query_text
    var output_k = reference_stage_output_k(candidate_set, plan)

    if plan.stage2_reference_operator.kind == "noop_topk":
        return noop_topk_stage2_result(
            candidate_set.hits,
            output_k,
        )

    if plan.stage2_reference_operator.kind == "exact_late_interaction":
        return exact_rerank_candidates_for_plan(
            backend,
            query,
            snapshot,
            candidate_set.hits,
            output_k,
        )

    raise Error(
        "unsupported stage2 reference operator kind: "
        + plan.stage2_reference_operator.kind
    )
