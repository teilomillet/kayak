from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.filters import FilterExpression, filter_expression_matches_doc_id, match_all_filter
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_exact_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> CandidateSet:
    var hits = List[CollectionHit]()

    for segment in snapshot.segments:
        var scores = backend.score_all(query, segment.stored_index.index)

        for index in range(len(scores)):
            if not filter_expression_matches_doc_id(
                filter_expression,
                segment.stored_index.index.doc_ids[index],
            ):
                continue
            insert_descending_collection_hit(
                hits,
                CollectionHit(
                    segment.manifest.segment_id.value.copy(),
                    segment.stored_index.index.doc_ids[index].copy(),
                    scores[index],
                ),
                plan.candidate_budget.candidate_k,
            )

    return CandidateSet(
        plan.candidate_generator.kind.copy(),
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        snapshot.snapshot.stats.token_count,
        snapshot.snapshot.stats.total_vector_count,
        snapshot.snapshot.stats.byte_size,
    )
