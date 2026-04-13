# Stage-3 verifier dispatch over one reference-scored window.

from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot

from .clause_text_stage import clause_text_rerank_candidates_for_plan
from .collection_hit import CollectionHit
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


def identity_stage3_result(read hits: List[CollectionHit]) raises -> Stage2Result:
    return Stage2Result(
        hits.copy(),
        count_unique_segment_ids(hits),
        len(hits),
        0,
        0,
        0,
    )


def stage3_result_for_plan(
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read reference_result: Stage2Result,
    read plan: SearchPlan,
) raises -> Stage2Result:
    if plan.stage3_verifier.kind == "none":
        return identity_stage3_result(reference_result.final_hits)

    if plan.stage3_verifier.kind == "clause_text":
        return clause_text_rerank_candidates_for_plan(
            query_text,
            snapshot,
            reference_result.final_hits,
            plan.candidate_budget.final_k,
        )

    raise Error(
        "unsupported stage3 verifier kind: "
        + plan.stage3_verifier.kind
    )
