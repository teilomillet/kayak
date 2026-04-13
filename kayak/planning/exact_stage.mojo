from std.collections import List

from kayak.collections import (
    LoadedSealedSegment,
    ResolvedCollectionSnapshot,
    loaded_segment_has_document_filter_index,
)
from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.collections.resolved_snapshot import (
    loaded_segment_document_metadata_for_doc_index,
)
from kayak.filters import (
    FilterExpression,
    filter_expression_matches_document,
    filter_expression_requires_document_metadata,
    match_all_filter,
)
from kayak.index import PackedIndex, pack_documents
from kayak.numeric import VectorScalar
from kayak.runtime import ExactScoringBackend
from kayak.storage.binary_vector_codec import native_vector_scalar_byte_width

from .collection_hit import CollectionHit
from .filter_allowlist import document_filter_allowlist_for_segment
from .filter_scope import effective_filter_expression_for_segment
from .graph_search_counters import GraphSearchCounters
from .score_histogram import build_score_histogram
from .stage_profile import SearchStageProfile
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage2_reference_operator import STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION
from .stage2_result import Stage2Result
from .topk import insert_descending_collection_hit


struct MaterializedCandidateIndex(Copyable):
    var index: PackedIndex
    var segment_ids: List[String]
    var segment_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        index: PackedIndex,
        var segment_ids: List[String],
        segment_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ):
        self.index = index.copy()
        self.segment_ids = segment_ids^
        self.segment_count = segment_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size


def count_unique_segment_ids(read segment_ids: List[String]) -> Int:
    var unique_segment_ids = List[String]()

    for segment_id in segment_ids:
        var already_seen = False
        for unique_segment_id in unique_segment_ids:
            if unique_segment_id == segment_id:
                already_seen = True
                break

        if not already_seen:
            unique_segment_ids.append(segment_id.copy())

    return len(unique_segment_ids)


def find_document_index_in_segment(
    read segment: LoadedSealedSegment, doc_id: String
) raises -> Int:
    for document_index in range(segment.stored_index.index.document_count):
        if segment.stored_index.index.doc_ids[document_index] == doc_id:
            return document_index

    raise Error("candidate doc_id not found in loaded segment: " + doc_id)


def build_encoded_document_from_segment(
    read segment: LoadedSealedSegment, document_index: Int
) raises -> EncodedDocument:
    var start = segment.stored_index.index.doc_offsets[document_index]
    var stop = segment.stored_index.index.doc_offsets[document_index + 1]
    var token_vectors = List[List[VectorScalar]]()

    for token_index in range(start, stop):
        token_vectors.append(segment.stored_index.index.token_vectors[token_index].copy())

    return EncodedDocument(
        segment.stored_index.index.doc_ids[document_index].copy(),
        token_vectors^,
    )


def materialize_candidate_index(
    read snapshot: ResolvedCollectionSnapshot, read hits: List[CollectionHit]
) raises -> MaterializedCandidateIndex:
    if len(hits) == 0:
        raise Error("cannot materialize an empty candidate index")

    var documents = List[EncodedDocument]()
    var segment_ids = List[String]()

    for hit in hits:
        var matched = False

        for segment in snapshot.segments:
            if segment.manifest.segment_id.value != hit.segment_id:
                continue

            documents.append(
                build_encoded_document_from_segment(
                    segment,
                    find_document_index_in_segment(segment, hit.doc_id),
                )
            )
            segment_ids.append(segment.manifest.segment_id.value.copy())
            matched = True
            break

        if not matched:
            raise Error(
                "candidate segment_id not found in resolved snapshot: " + hit.segment_id
            )

    var index = pack_documents(documents)
    var scalar_width = native_vector_scalar_byte_width()
    return MaterializedCandidateIndex(
        index,
        segment_ids^,
        count_unique_segment_ids(segment_ids),
        index.total_vector_count,
        index.total_vector_count,
        index.total_vector_count * index.vector_dim * scalar_width,
    )


def exact_rerank_candidates_for_plan[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read hits: List[CollectionHit],
    final_k: Int,
) raises -> Stage2Result:
    if len(hits) == 0:
        return Stage2Result([], 0, 0, 0, 0, 0)

    var materialized = materialize_candidate_index(snapshot, hits)
    var scores = backend.score_all(query, materialized.index)
    var final_hits = List[CollectionHit]()

    for document_index in range(len(scores)):
        insert_descending_collection_hit(
            final_hits,
            CollectionHit(
                materialized.segment_ids[document_index].copy(),
                materialized.index.doc_ids[document_index].copy(),
                scores[document_index],
            ),
            final_k,
        )

    return Stage2Result(
        final_hits^,
        [
            StageArtifactMaterialization(
                STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
                materialized.segment_count,
                materialized.index.document_count,
                materialized.token_count,
                materialized.vector_count,
                materialized.byte_size,
            )
        ],
        materialized.segment_count,
        materialized.index.document_count,
        materialized.token_count,
        materialized.vector_count,
        materialized.byte_size,
    )


def exact_oracle_hits_for_snapshot[Backend: ExactScoringBackend](
    read backend: Backend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    final_k: Int,
    read filter_expression: FilterExpression = match_all_filter(),
) raises -> List[CollectionHit]:
    var oracle_hits = List[CollectionHit]()

    for segment in snapshot.segments:
        var scores = backend.score_all(query, segment.stored_index.index)
        var allowed_flags = List[Int]()
        var use_allowlist = False
        var effective_filter = effective_filter_expression_for_segment(
            snapshot.collection,
            segment,
            filter_expression,
        )
        if not effective_filter.is_match_all():
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

        for document_index in range(len(scores)):
            if use_allowlist:
                if allowed_flags[document_index] == 0:
                    continue
            elif not effective_filter.is_match_all():
                if not filter_expression_matches_document(
                    effective_filter,
                    segment.stored_index.index.doc_ids[document_index],
                    loaded_segment_document_metadata_for_doc_index(
                        segment, document_index
                    ),
                ):
                    continue
            insert_descending_collection_hit(
                oracle_hits,
                CollectionHit(
                    segment.manifest.segment_id.value.copy(),
                    segment.stored_index.index.doc_ids[document_index].copy(),
                    scores[document_index],
                ),
                final_k,
            )

    return oracle_hits^


def exact_oracle_stage_profile_for_snapshot(
    read snapshot: ResolvedCollectionSnapshot,
    read oracle_hits: List[CollectionHit],
) raises -> SearchStageProfile:
    var scalar_width = native_vector_scalar_byte_width()
    return SearchStageProfile(
        "exact_oracle",
        snapshot.snapshot.stats.document_count,
        len(oracle_hits),
        snapshot.snapshot.stats.segment_count,
        snapshot.snapshot.stats.document_count,
        snapshot.snapshot.stats.token_count,
        snapshot.snapshot.stats.total_vector_count,
        snapshot.snapshot.stats.total_vector_count
            * snapshot.collection.vector_dim
            * scalar_width,
        GraphSearchCounters(),
        build_score_histogram(oracle_hits, 8),
    )
