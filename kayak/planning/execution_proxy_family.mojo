from std.collections import List

from kayak.collections import (
    ResolvedCollectionSnapshot,
    loaded_segment_has_document_proxy_index,
    loaded_segment_stored_document_proxy_index,
)
from kayak.contracts import EncodedQuery
from kayak.index import build_query_proxy_vector
from kayak.runtime import ExactScoringBackend
from kayak.scoring.dot import dot_product

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .search_plan import SearchPlan
from .topk import insert_descending_collection_hit


def candidate_generation_for_proxy_family[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read plan: SearchPlan,
) raises -> CandidateSet:
    _ = backend

    var hits = List[CollectionHit]()
    var query_proxy = build_query_proxy_vector(query, 0)
    var vector_count = 0
    var byte_size = 0

    for segment in snapshot.segments:
        if not loaded_segment_has_document_proxy_index(segment):
            raise Error(
                "document_proxy stage-1 requires a document proxy sidecar for every segment"
            )

        var stored_proxy = loaded_segment_stored_document_proxy_index(segment)
        vector_count += (
            stored_proxy.index.document_count
            * stored_proxy.proxy_vector_count_per_document
        )
        byte_size += stored_proxy.artifact_byte_size

        for document_index in range(stored_proxy.index.document_count):
            insert_descending_collection_hit(
                hits,
                CollectionHit(
                    segment.manifest.segment_id.value.copy(),
                    stored_proxy.index.doc_ids[document_index].copy(),
                    dot_product(
                        query_proxy,
                        stored_proxy.index.proxy_vectors[document_index],
                    ),
                ),
                plan.candidate_budget.candidate_k,
            )

    return CandidateSet(
        plan.candidate_generator.kind.copy(),
        hits^,
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        0,
        vector_count,
        byte_size,
    )
