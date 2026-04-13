from std.collections import List
from std.pathlib import Path

from kayak.contracts import EncodedDocument
from kayak.index import unpack_documents

from .collection_store import load_collection_manifest
from .compaction import CompactionPlan
from .document_representation_transform import (
    DocumentRepresentationTransformManifest,
    same_document_representation_transforms,
)
from .document_metadata import DocumentMetadataMap
from .ids import SegmentId, SnapshotId
from .publish import publish_collection_snapshot
from .resolution_requirements import load_all_snapshot_requirements
from .resolved_snapshot import (
    LoadedSealedSegment,
    ResolvedCollectionSnapshot,
    loaded_segment_document_metadata_for_doc_index,
)
from .resolver import load_resolved_collection_snapshot
from .segment import SealedSegmentManifest
from .segment_builder import seal_single_segment_from_stored_documents
from .segment_store import sealed_segment_manifest_exists
from .snapshot import SnapshotManifest
from .snapshot_store import load_snapshot_manifest
from .stats import CollectionStats, SegmentStats


def find_loaded_segment_index(
    read snapshot: ResolvedCollectionSnapshot, read segment_id: SegmentId
) -> Int:
    for index in range(len(snapshot.segments)):
        if snapshot.segments[index].manifest.segment_id.value == segment_id.value:
            return index

    return -1


def require_live_snapshot_compaction_target(
    collection_root: Path, snapshot_id: SnapshotId
) raises -> SnapshotManifest:
    var collection = load_collection_manifest(collection_root)
    var snapshot = load_snapshot_manifest(collection_root / "snapshots" / snapshot_id.value)
    if collection.active_snapshot_id.byte_length() != 0:
        if snapshot.snapshot_id.value != collection.active_snapshot_id:
            raise Error(
                "compaction currently supports only the active published snapshot"
            )

    if snapshot.generation != collection.latest_generation:
        raise Error(
            "compaction currently supports only the live published snapshot generation"
        )

    return snapshot^


def build_compaction_plan_for_snapshot(
    collection_root: Path,
    snapshot_id: SnapshotId,
    read source_segment_ids: List[SegmentId],
    target_segment_id: SegmentId,
    reason: String,
) raises -> CompactionPlan:
    var resolved = load_resolved_collection_snapshot(
        collection_root,
        snapshot_id,
        load_all_snapshot_requirements(),
    )
    _ = require_live_snapshot_compaction_target(collection_root, snapshot_id)

    var document_count = 0
    var token_count = 0
    var total_vector_count = 0

    for segment_id in source_segment_ids:
        var source_index = find_loaded_segment_index(resolved, segment_id)
        if source_index == -1:
            raise Error(
                "compaction source segment is not present in the snapshot: "
                + segment_id.value
            )

        document_count += resolved.segments[source_index].manifest.stats.document_count
        token_count += resolved.segments[source_index].manifest.stats.token_count
        total_vector_count += resolved.segments[source_index].manifest.stats.total_vector_count

    return CompactionPlan(
        resolved.collection.collection_id,
        resolved.collection.tenant_id,
        resolved.collection.namespace_id,
        source_segment_ids.copy(),
        target_segment_id,
        reason,
        SegmentStats(
            document_count,
            token_count,
            total_vector_count,
            0,
        ),
    )


def append_segment_documents(
    mut documents: List[EncodedDocument],
    mut texts: List[String],
    mut metadata_maps: List[DocumentMetadataMap],
    read segment: LoadedSealedSegment,
) raises:
    var unpacked = unpack_documents(segment.stored_index.index)
    for index in range(len(unpacked)):
        documents.append(unpacked[index].copy())
        if segment.has_text_corpus:
            texts.append(segment.stored_text_corpus.corpus.texts[index].copy())
        else:
            texts.append(String())
        metadata_maps.append(
            loaded_segment_document_metadata_for_doc_index(segment, index)
        )


def aggregate_snapshot_stats(
    read kept_segments: List[LoadedSealedSegment],
    read compacted_segment: SealedSegmentManifest,
) raises -> CollectionStats:
    var segment_count = 1
    var document_count = compacted_segment.stats.document_count
    var token_count = compacted_segment.stats.token_count
    var total_vector_count = compacted_segment.stats.total_vector_count
    var byte_size = compacted_segment.stats.byte_size

    for segment in kept_segments:
        segment_count += 1
        document_count += segment.manifest.stats.document_count
        token_count += segment.manifest.stats.token_count
        total_vector_count += segment.manifest.stats.total_vector_count
        byte_size += segment.manifest.stats.byte_size

    return CollectionStats(
        segment_count,
        document_count,
        token_count,
        total_vector_count,
        byte_size,
    )


def require_consistent_source_segment_transforms(
    read source_segments: List[LoadedSealedSegment]
) raises -> List[DocumentRepresentationTransformManifest]:
    if len(source_segments) == 0:
        return List[DocumentRepresentationTransformManifest]()

    var expected = (
        source_segments[0].manifest.document_representation_transforms.copy()
    )
    for segment_index in range(1, len(source_segments)):
        if not same_document_representation_transforms(
            expected,
            source_segments[segment_index]
                .manifest
                .document_representation_transforms,
        ):
            raise Error(
                "compaction currently requires source segments to share the same document representation transforms"
            )

    return expected.copy()


def execute_compaction_plan(
    collection_root: Path,
    source_snapshot_id: SnapshotId,
    target_snapshot_id: SnapshotId,
    read plan: CompactionPlan,
) raises -> SnapshotManifest:
    _ = require_live_snapshot_compaction_target(collection_root, source_snapshot_id)
    var resolved = load_resolved_collection_snapshot(
        collection_root,
        source_snapshot_id,
        load_all_snapshot_requirements(),
    )
    if resolved.collection.collection_id.value != plan.collection_id.value:
        raise Error("compaction plan collection_id does not match resolved snapshot")
    if resolved.collection.tenant_id.value != plan.tenant_id.value:
        raise Error("compaction plan tenant_id does not match resolved snapshot")
    if resolved.collection.namespace_id.value != plan.namespace_id.value:
        raise Error("compaction plan namespace_id does not match resolved snapshot")

    if sealed_segment_manifest_exists(
        collection_root / "segments" / plan.target_segment_id.value
    ):
        raise Error("compaction target segment already exists")

    var documents = List[EncodedDocument]()
    var texts = List[String]()
    var metadata_maps = List[DocumentMetadataMap]()
    var kept_segments = List[LoadedSealedSegment]()
    var kept_segment_ids = List[SegmentId]()
    var source_segments = List[LoadedSealedSegment]()
    var source_segment_count = 0

    for segment in resolved.segments:
        var is_source = False
        for source_segment_id in plan.source_segment_ids:
            if segment.manifest.segment_id.value == source_segment_id.value:
                is_source = True
                break

        if is_source:
            source_segment_count += 1
            source_segments.append(segment.copy())
            append_segment_documents(documents, texts, metadata_maps, segment)
        else:
            kept_segments.append(segment.copy())
            kept_segment_ids.append(segment.manifest.segment_id.copy())

    if source_segment_count != len(plan.source_segment_ids):
        raise Error("not all compaction source segments were found in the snapshot")
    if len(documents) == 0:
        raise Error("compaction plan did not select any documents")
    var document_representation_transforms = (
        require_consistent_source_segment_transforms(source_segments)
    )

    var next_generation = resolved.collection.latest_generation + 1
    var compacted_segment = seal_single_segment_from_stored_documents(
        collection_root,
        resolved.collection,
        plan.target_segment_id,
        next_generation,
        documents,
        document_representation_transforms,
        texts,
        metadata_maps,
    )
    if compacted_segment.stats.document_count != plan.expected_output_stats.document_count:
        raise Error("compacted document_count does not match compaction plan")
    if compacted_segment.stats.token_count != plan.expected_output_stats.token_count:
        raise Error("compacted token_count does not match compaction plan")
    if (
        compacted_segment.stats.total_vector_count
        != plan.expected_output_stats.total_vector_count
    ):
        raise Error("compacted total_vector_count does not match compaction plan")

    var next_segment_ids = List[SegmentId]()
    for segment_id in kept_segment_ids:
        next_segment_ids.append(segment_id.copy())
    next_segment_ids.append(plan.target_segment_id.copy())

    var replacement_snapshot = SnapshotManifest(
        target_snapshot_id,
        resolved.collection.collection_id,
        resolved.collection.tenant_id,
        resolved.collection.namespace_id,
        next_generation,
        next_segment_ids^,
        aggregate_snapshot_stats(kept_segments, compacted_segment),
    )

    _ = publish_collection_snapshot(
        collection_root,
        resolved.collection,
        replacement_snapshot,
    )
    return replacement_snapshot^
