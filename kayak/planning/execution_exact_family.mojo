from std.collections import List

from kayak.collections import (
    ResolvedCollectionSnapshot,
    loaded_segment_has_document_filter_index,
)
from kayak.contracts import EncodedQuery
from kayak.collections.resolved_snapshot import (
    loaded_segment_document_metadata_for_doc_index,
)
from kayak.filters import (
    FilterExpression,
    filter_expression_matches_document,
    filter_expression_requires_document_metadata,
    match_all_filter,
)
from kayak.runtime import ExactScoringBackend

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .filter_allowlist import document_filter_allowlist_for_segment
from .filter_scope import effective_filter_expression_for_segment
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
        var allowed_flags = List[Int]()
        var use_allowlist = False
        if not filter_expression.is_match_all():
            var effective_filter = effective_filter_expression_for_segment(
                snapshot.collection,
                segment,
                filter_expression,
            )
            if (
                not filter_expression_requires_document_metadata(effective_filter)
                or loaded_segment_has_document_filter_index(segment)
            ):
                var allowlist = document_filter_allowlist_for_segment(
                    segment,
                    effective_filter,
                )
                allowed_flags = allowlist.flags.copy()
                use_allowlist = True

        for index in range(len(scores)):
            if use_allowlist:
                if allowed_flags[index] == 0:
                    continue
            elif not filter_expression.is_match_all():
                if not filter_expression_matches_document(
                    filter_expression,
                    segment.stored_index.index.doc_ids[index],
                    loaded_segment_document_metadata_for_doc_index(
                        segment, index
                    ),
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
        plan.candidate_generator,
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        snapshot.snapshot.stats.token_count,
        snapshot.snapshot.stats.total_vector_count,
        snapshot.snapshot.stats.byte_size,
    )
