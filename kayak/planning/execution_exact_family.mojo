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
from .filter_allowlist import (
    document_filter_allowlist_artifact_byte_size_for_segment,
    document_filter_allowlist_for_segment,
)
from .filter_application_profile import (
    filter_application_profile_for_effective_filter,
)
from .filter_scope import (
    effective_filter_expression_for_collection,
    effective_filter_expression_for_segment,
)
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
    var effective_collection_filter = effective_filter_expression_for_collection(
        snapshot.collection,
        filter_expression,
    )
    var filter_input_document_count = 0
    var filter_matching_document_count = 0
    var filter_artifact_byte_size = 0
    var uses_document_filter_index = False

    for segment_index in range(len(snapshot.segments)):
        var scores = backend.score_all(
            query,
            snapshot.segments[segment_index].stored_index.index,
        )
        var allowed_flags = List[Int]()
        var use_allowlist = False
        var segment_matching_document_count = len(scores)
        filter_input_document_count += len(scores)
        var effective_filter = effective_filter_expression_for_segment(
            snapshot.collection,
            snapshot.segments[segment_index],
            filter_expression,
        )
        if not effective_filter.is_match_all():
            segment_matching_document_count = 0
            if (
                not filter_expression_requires_document_metadata(effective_filter)
                or loaded_segment_has_document_filter_index(
                    snapshot.segments[segment_index]
                )
            ):
                var allowlist = document_filter_allowlist_for_segment(
                    snapshot.segments[segment_index],
                    effective_filter,
                )
                allowed_flags = allowlist.flags.copy()
                segment_matching_document_count = (
                    allowlist.matching_document_count
                )
                use_allowlist = True
                if filter_expression_requires_document_metadata(effective_filter):
                    filter_artifact_byte_size += (
                        document_filter_allowlist_artifact_byte_size_for_segment(
                            snapshot.segments[segment_index],
                            effective_filter,
                        )
                    )
                    uses_document_filter_index = True

        for index in range(len(scores)):
            if use_allowlist:
                if allowed_flags[index] == 0:
                    continue
            elif not effective_filter.is_match_all():
                if not filter_expression_matches_document(
                    effective_filter,
                    snapshot.segments[segment_index].stored_index.index.doc_ids[index],
                    loaded_segment_document_metadata_for_doc_index(
                        snapshot.segments[segment_index], index
                    ),
                ):
                    continue
                segment_matching_document_count += 1
            insert_descending_collection_hit(
                hits,
                CollectionHit(
                    snapshot.segments[segment_index].manifest.segment_id.value.copy(),
                    snapshot.segments[segment_index].stored_index.index.doc_ids[
                        index
                    ].copy(),
                    scores[index],
                    segment_index,
                    index,
                ),
                plan.candidate_budget.candidate_k,
            )
        filter_matching_document_count += segment_matching_document_count

    var candidate_set = CandidateSet(
        plan.candidate_generator,
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        snapshot.snapshot.stats.token_count,
        snapshot.snapshot.stats.total_vector_count,
        snapshot.snapshot.stats.byte_size,
    )
    candidate_set.filter_application_profile = (
        filter_application_profile_for_effective_filter(
            filter_expression,
            effective_collection_filter,
            filter_input_document_count,
            filter_matching_document_count,
            filter_artifact_byte_size,
            uses_document_filter_index,
        )
    )
    return candidate_set^
