from std.algorithm.backend.cpu.parallelize import sync_parallelize
from std.collections import List

from kayak.collections import (
    LoadedSealedSegment,
    ResolvedCollectionSnapshot,
    loaded_segment_has_document_filter_index,
)
from kayak.contracts import (
    EncodedDocument,
    EncodedQuery,
    FlatQueryDim128,
    build_flat_query_dim128,
)
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
from kayak.numeric import (
    VECTOR_SCALAR_NAME,
    ScoreScalar,
    VectorScalar,
    zero_score_scalar,
)
from kayak.runtime import ExactScoringBackend
from kayak.runtime import ExactCpuBackend
from kayak.scoring.dot128 import COLBERT_VECTOR_DIM
from kayak.scoring.maxsim import (
    choose_parallel_work_item_count_for_shape,
    exact_score_for_document_dim128_flat_query_tiled4,
    exact_score_for_document_with_config,
)
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


struct ResolvedCandidateDocument(Copyable):
    var segment_index: Int
    var document_index: Int
    var segment_id: String
    var doc_id: String
    var vector_count: Int

    def __init__(
        out self,
        segment_index: Int,
        document_index: Int,
        var segment_id: String,
        var doc_id: String,
        vector_count: Int,
    ):
        self.segment_index = segment_index
        self.document_index = document_index
        self.segment_id = segment_id^
        self.doc_id = doc_id^
        self.vector_count = vector_count


struct ResolvedCandidateWindow(Copyable):
    var documents: List[ResolvedCandidateDocument]
    var segment_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int

    def __init__(
        out self,
        var documents: List[ResolvedCandidateDocument],
        segment_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
    ):
        self.documents = documents^
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


def append_unique_segment_index(
    mut unique_segment_indices: List[Int], segment_index: Int
):
    for existing_segment_index in unique_segment_indices:
        if existing_segment_index == segment_index:
            return

    unique_segment_indices.append(segment_index)


def find_document_index_in_segment(
    read segment: LoadedSealedSegment, doc_id: String
) raises -> Int:
    for document_index in range(segment.stored_index.index.document_count):
        if segment.stored_index.index.doc_ids[document_index] == doc_id:
            return document_index

    raise Error("candidate doc_id not found in loaded segment: " + doc_id)


def has_resolved_indices(
    read hit: CollectionHit, read snapshot: ResolvedCollectionSnapshot
) -> Bool:
    if hit.segment_index < 0 or hit.document_index < 0:
        return False
    if hit.segment_index >= len(snapshot.segments):
        return False

    if (
        hit.document_index
        >= snapshot.segments[hit.segment_index].stored_index.index.document_count
    ):
        return False

    return (
        snapshot.segments[hit.segment_index].manifest.segment_id.value == hit.segment_id
        and snapshot.segments[hit.segment_index].stored_index.index.doc_ids[
            hit.document_index
        ] == hit.doc_id
    )


def resolve_document_index_for_hit(
    read snapshot: ResolvedCollectionSnapshot, read hit: CollectionHit
) raises -> ResolvedCandidateDocument:
    if has_resolved_indices(hit, snapshot):
        return ResolvedCandidateDocument(
            hit.segment_index,
            hit.document_index,
            hit.segment_id.copy(),
            hit.doc_id.copy(),
            vector_count_for_document_index(
                snapshot.segments[hit.segment_index],
                hit.document_index,
            ),
        )

    for segment_index in range(len(snapshot.segments)):
        if snapshot.segments[segment_index].manifest.segment_id.value != hit.segment_id:
            continue

        var document_index = find_document_index_in_segment(
            snapshot.segments[segment_index],
            hit.doc_id,
        )
        return ResolvedCandidateDocument(
            segment_index,
            document_index,
            hit.segment_id.copy(),
            hit.doc_id.copy(),
            vector_count_for_document_index(
                snapshot.segments[segment_index],
                document_index,
            ),
        )

    raise Error(
        "candidate segment_id not found in resolved snapshot: " + hit.segment_id
    )


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


def vector_count_for_document_index(
    read segment: LoadedSealedSegment, document_index: Int
) -> Int:
    return (
        segment.stored_index.index.doc_offsets[document_index + 1]
        - segment.stored_index.index.doc_offsets[document_index]
    )


def resolve_candidate_window(
    read snapshot: ResolvedCollectionSnapshot, read hits: List[CollectionHit]
) raises -> ResolvedCandidateWindow:
    if len(hits) == 0:
        raise Error("cannot resolve an empty candidate window")

    var documents = List[ResolvedCandidateDocument]()
    var unique_segment_indices = List[Int]()
    var total_vector_count = 0

    for hit in hits:
        var resolved_document = resolve_document_index_for_hit(snapshot, hit)
        append_unique_segment_index(
            unique_segment_indices, resolved_document.segment_index
        )
        total_vector_count += resolved_document.vector_count
        documents.append(resolved_document.copy())

    var scalar_width = native_vector_scalar_byte_width()
    return ResolvedCandidateWindow(
        documents^,
        len(unique_segment_indices),
        total_vector_count,
        total_vector_count,
        total_vector_count * snapshot.collection.vector_dim * scalar_width,
    )


def build_resolved_candidate_vector_offsets(
    read resolved: ResolvedCandidateWindow
) -> List[Int]:
    var vector_offsets = List[Int]()
    var running_total = 0
    vector_offsets.append(0)

    for resolved_document in resolved.documents:
        running_total += resolved_document.vector_count
        vector_offsets.append(running_total)

    return vector_offsets^


def build_resolved_candidate_boundaries(
    read resolved: ResolvedCandidateWindow, work_item_count: Int
) -> List[Int]:
    var boundaries = List[Int]()
    var document_count = len(resolved.documents)
    boundaries.append(0)

    if work_item_count <= 1:
        boundaries.append(document_count)
        return boundaries^

    var vector_offsets = build_resolved_candidate_vector_offsets(resolved)
    var start_doc = 0

    for work_item in range(work_item_count - 1):
        var remaining_work_items = work_item_count - work_item
        var remaining_vectors = resolved.vector_count - vector_offsets[start_doc]
        var target_vectors = (
            remaining_vectors + remaining_work_items - 1
        ) // remaining_work_items
        var max_stop_doc = document_count - (remaining_work_items - 1)
        var stop_doc = start_doc + 1

        while (
            stop_doc < max_stop_doc
            and (
                vector_offsets[stop_doc] - vector_offsets[start_doc]
            ) < target_vectors
        ):
            stop_doc += 1

        boundaries.append(stop_doc)
        start_doc = stop_doc

    boundaries.append(document_count)
    return boundaries^


def exact_score_for_resolved_document(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_document: ResolvedCandidateDocument,
) -> ScoreScalar:
    return exact_score_for_document_with_config(
        query,
        snapshot.segments[resolved_document.segment_index].stored_index.index,
        resolved_document.document_index,
        backend.scoring_config,
    )


def should_use_dim128_tiled4_exact_stage(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
) -> Bool:
    return (
        backend.scoring_config.enable_dim128_fast_path
        and VECTOR_SCALAR_NAME == "Float32"
        and query.vector_dim == COLBERT_VECTOR_DIM
        and query.vector_count >= 4
        and query.vector_count <= 64
    )


def exact_score_for_resolved_document_dim128_tiled4(
    read query: FlatQueryDim128,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved_document: ResolvedCandidateDocument,
) -> ScoreScalar:
    return exact_score_for_document_dim128_flat_query_tiled4(
        query,
        snapshot.segments[resolved_document.segment_index].stored_index.index,
        resolved_document.document_index,
    )


def score_resolved_candidate_window_for_cpu_dim128_tiled4(
    read backend: ExactCpuBackend,
    read query: FlatQueryDim128,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            scores_ptr[document_index] = (
                exact_score_for_resolved_document_dim128_tiled4(
                    query,
                    snapshot,
                    resolved.documents[document_index],
                )
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved, work_item_count
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = (
                exact_score_for_resolved_document_dim128_tiled4(
                    query,
                    snapshot,
                    resolved.documents[document_index],
                )
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def score_resolved_candidate_window_for_cpu(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read resolved: ResolvedCandidateWindow,
) raises -> List[ScoreScalar]:
    # This path is only enabled for the measured dim128 exact-stage shapes:
    # q in [4, 64]. Other shapes continue to use the existing scorer until they
    # are benchmarked.
    if should_use_dim128_tiled4_exact_stage(backend, query):
        return score_resolved_candidate_window_for_cpu_dim128_tiled4(
            backend,
            build_flat_query_dim128(query),
            snapshot,
            resolved,
        )

    var scores = List[ScoreScalar]()
    for _ in range(len(resolved.documents)):
        scores.append(zero_score_scalar())

    var work_item_count = choose_parallel_work_item_count_for_shape(
        query.vector_count,
        len(resolved.documents),
        resolved.vector_count,
        backend.scoring_config,
    )
    var scores_ptr = scores.unsafe_ptr()

    if work_item_count <= 1:
        for document_index in range(len(resolved.documents)):
            scores_ptr[document_index] = exact_score_for_resolved_document(
                backend,
                query,
                snapshot,
                resolved.documents[document_index],
            )
        return scores^

    var boundaries = build_resolved_candidate_boundaries(
        resolved, work_item_count
    )

    @parameter
    def score_partition(work_item: Int):
        var start_doc = boundaries[work_item]
        var stop_doc = boundaries[work_item + 1]

        for document_index in range(start_doc, stop_doc):
            scores_ptr[document_index] = exact_score_for_resolved_document(
                backend,
                query,
                snapshot,
                resolved.documents[document_index],
            )

    sync_parallelize[score_partition](work_item_count)
    return scores^


def materialize_candidate_index(
    read snapshot: ResolvedCollectionSnapshot, read hits: List[CollectionHit]
) raises -> MaterializedCandidateIndex:
    if len(hits) == 0:
        raise Error("cannot materialize an empty candidate index")

    var documents = List[EncodedDocument]()
    var segment_ids = List[String]()
    var unique_segment_indices = List[Int]()

    for hit in hits:
        var resolved_document = resolve_document_index_for_hit(snapshot, hit)
        documents.append(
            build_encoded_document_from_segment(
                snapshot.segments[resolved_document.segment_index],
                resolved_document.document_index,
            )
        )
        segment_ids.append(
            snapshot.segments[resolved_document.segment_index].manifest.segment_id.value.copy()
        )
        append_unique_segment_index(
            unique_segment_indices,
            resolved_document.segment_index,
        )

    var index = pack_documents(documents)
    var scalar_width = native_vector_scalar_byte_width()
    return MaterializedCandidateIndex(
        index,
        segment_ids^,
        len(unique_segment_indices),
        index.total_vector_count,
        index.total_vector_count,
        index.total_vector_count * index.vector_dim * scalar_width,
    )


def exact_rerank_candidates_for_plan(
    read backend: ExactCpuBackend,
    read query: EncodedQuery,
    read snapshot: ResolvedCollectionSnapshot,
    read hits: List[CollectionHit],
    final_k: Int,
) raises -> Stage2Result:
    if len(hits) == 0:
        return Stage2Result([], 0, 0, 0, 0, 0)

    var resolved = resolve_candidate_window(snapshot, hits)
    var scores = score_resolved_candidate_window_for_cpu(
        backend,
        query,
        snapshot,
        resolved,
    )
    var final_hits = List[CollectionHit]()

    for document_index in range(len(scores)):
        insert_descending_collection_hit(
            final_hits,
            CollectionHit(
                resolved.documents[document_index].segment_id.copy(),
                resolved.documents[document_index].doc_id.copy(),
                scores[document_index],
            ),
            final_k,
        )

    return Stage2Result(
        final_hits^,
        [
            StageArtifactMaterialization(
                STAGE2_REFERENCE_REQUIRED_ARTIFACT_LATE_INTERACTION,
                resolved.segment_count,
                len(resolved.documents),
                resolved.token_count,
                resolved.vector_count,
                resolved.byte_size,
            )
        ],
        resolved.segment_count,
        len(resolved.documents),
        resolved.token_count,
        resolved.vector_count,
        resolved.byte_size,
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
