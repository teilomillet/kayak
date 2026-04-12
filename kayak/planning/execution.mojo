from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.contracts import EncodedQuery
from kayak.numeric import MetricScalar
from kayak.runtime import ExactScoringBackend
from kayak.search import SearchHit

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit, to_search_hit
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CandidateSet:
    if plan.candidate_generator.kind != "exact_full_scan":
        raise Error(
            "unsupported candidate generator kind: "
            + plan.candidate_generator.kind
        )

    var hits = List[CollectionHit]()

    for segment in snapshot.segments:
        var scores = backend.score_all(query, segment.stored_index.index)

        for index in range(len(scores)):
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


def final_hits_for_plan(
    read candidate_set: CandidateSet, read plan: SearchPlan
) -> List[CollectionHit]:
    var hits = List[CollectionHit]()
    var limit = plan.candidate_budget.final_k
    if limit > len(candidate_set.hits):
        limit = len(candidate_set.hits)

    for index in range(limit):
        hits.append(candidate_set.hits[index].copy())

    return hits^


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


def final_hits_to_search_hits(
    read hits: List[CollectionHit]
) -> List[SearchHit]:
    var search_hits = List[SearchHit]()

    for hit in hits:
        search_hits.append(to_search_hit(hit))

    return search_hits^
