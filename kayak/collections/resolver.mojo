from std.collections import List
from std.pathlib import Path

from kayak.index import CentroidPostingIndex, DocumentProxyIndex
from kayak.storage import (
    StoredCentroidPostingIndex,
    StoredDocumentProxyIndex,
    load_stored_centroid_posting_index,
    load_stored_document_proxy_index,
    load_stored_packed_index,
)
from kayak.text import DocumentTextCorpus

from .collection import CollectionManifest
from .collection_store import load_collection_manifest
from .ids import CollectionId, SegmentId, SnapshotId
from .paths import (
    collection_segment_root,
    collection_snapshot_root,
    resolve_segment_artifact_root,
)
from .resolved_snapshot import LoadedSealedSegment, ResolvedCollectionSnapshot
from .segment import (
    SealedSegmentManifest,
    sealed_segment_has_centroid_postings_index,
    sealed_segment_has_document_proxy_index,
    sealed_segment_has_text_corpus,
)
from .segment_store import load_sealed_segment_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import load_snapshot_manifest
from .stats import CollectionStats
from .text_corpus import StoredDocumentTextCorpus
from .text_corpus_store import load_stored_document_text_corpus


def empty_stored_document_text_corpus(
    read collection_id: CollectionId, read segment_id: SegmentId
) raises -> StoredDocumentTextCorpus:
    return StoredDocumentTextCorpus(
        collection_id.copy(),
        segment_id.copy(),
        DocumentTextCorpus([], []),
    )


def empty_stored_document_proxy_index(
    model_name: String, vector_scalar_name: String, vector_dim: Int
) raises -> StoredDocumentProxyIndex:
    return StoredDocumentProxyIndex(
        "",
        model_name.copy(),
        vector_scalar_name.copy(),
        0,
        1,
        0,
        DocumentProxyIndex([], [], vector_dim),
    )


def empty_stored_centroid_posting_index(
    model_name: String, vector_scalar_name: String, vector_dim: Int
) raises -> StoredCentroidPostingIndex:
    return StoredCentroidPostingIndex(
        "",
        model_name.copy(),
        vector_scalar_name.copy(),
        0,
        0,
        CentroidPostingIndex([], [], [0], [], [], vector_dim, 0),
    )


def require_collection_and_snapshot_match(
    read collection: CollectionManifest, read snapshot: SnapshotManifest
) raises:
    if snapshot.collection_id.value != collection.collection_id.value:
        raise Error("snapshot collection_id does not match collection manifest")

    if snapshot.tenant_id.value != collection.tenant_id.value:
        raise Error("snapshot tenant_id does not match collection manifest")

    if snapshot.namespace_id.value != collection.namespace_id.value:
        raise Error("snapshot namespace_id does not match collection manifest")

    if snapshot.generation > collection.latest_generation:
        raise Error("snapshot generation exceeds collection latest_generation")


def require_segment_matches_collection(
    read collection: CollectionManifest,
    read snapshot: SnapshotManifest,
    read segment: SealedSegmentManifest,
) raises:
    if segment.collection_id.value != collection.collection_id.value:
        raise Error("segment collection_id does not match collection manifest")

    if segment.tenant_id.value != collection.tenant_id.value:
        raise Error("segment tenant_id does not match collection manifest")

    if segment.namespace_id.value != collection.namespace_id.value:
        raise Error("segment namespace_id does not match collection manifest")

    if segment.model_name != collection.model_name:
        raise Error("segment model_name does not match collection manifest")

    if segment.vector_scalar_name != collection.vector_scalar_name:
        raise Error("segment vector_scalar_name does not match collection manifest")

    if segment.vector_dim != collection.vector_dim:
        raise Error("segment vector_dim does not match collection manifest")

    if segment.generation > snapshot.generation:
        raise Error("segment generation exceeds snapshot generation")


def require_loaded_index_matches_segment(
    read segment: SealedSegmentManifest, loaded_index_document_count: Int,
    loaded_index_total_vector_count: Int, loaded_index_vector_dim: Int,
    loaded_index_model_name: String, loaded_index_vector_scalar_name: String,
) raises:
    if loaded_index_model_name != segment.model_name:
        raise Error("packed index model_name does not match segment manifest")

    if loaded_index_vector_scalar_name != segment.vector_scalar_name:
        raise Error(
            "packed index vector_scalar_name does not match segment manifest"
        )

    if loaded_index_vector_dim != segment.vector_dim:
        raise Error("packed index vector_dim does not match segment manifest")

    if loaded_index_document_count != segment.stats.document_count:
        raise Error("packed index document_count does not match segment stats")

    if loaded_index_total_vector_count != segment.stats.total_vector_count:
        raise Error("packed index total_vector_count does not match segment stats")


def require_loaded_text_corpus_matches_segment(
    read segment: SealedSegmentManifest, read stored_text_corpus: StoredDocumentTextCorpus
) raises:
    if stored_text_corpus.collection_id.value != segment.collection_id.value:
        raise Error("text corpus collection_id does not match segment manifest")

    if stored_text_corpus.segment_id.value != segment.segment_id.value:
        raise Error("text corpus segment_id does not match segment manifest")

    if len(stored_text_corpus.corpus.doc_ids) != segment.stats.document_count:
        raise Error("text corpus document_count does not match segment stats")


def require_loaded_document_proxy_matches_segment(
    read segment: SealedSegmentManifest,
    read stored_document_proxy_index: StoredDocumentProxyIndex,
) raises:
    if stored_document_proxy_index.model_name != segment.model_name:
        raise Error("document proxy model_name does not match segment manifest")

    if stored_document_proxy_index.vector_scalar_name != segment.vector_scalar_name:
        raise Error(
            "document proxy vector_scalar_name does not match segment manifest"
        )

    if stored_document_proxy_index.index.vector_dim != segment.vector_dim:
        raise Error("document proxy vector_dim does not match segment manifest")

    if (
        stored_document_proxy_index.index.document_count
        != segment.stats.document_count
    ):
        raise Error("document proxy document_count does not match segment stats")


def require_loaded_centroid_postings_matches_segment(
    read segment: SealedSegmentManifest,
    read stored_centroid_postings_index: StoredCentroidPostingIndex,
) raises:
    if stored_centroid_postings_index.model_name != segment.model_name:
        raise Error("centroid postings model_name does not match segment manifest")

    if (
        stored_centroid_postings_index.vector_scalar_name
        != segment.vector_scalar_name
    ):
        raise Error(
            "centroid postings vector_scalar_name does not match segment manifest"
        )

    if stored_centroid_postings_index.index.vector_dim != segment.vector_dim:
        raise Error("centroid postings vector_dim does not match segment manifest")

    if (
        stored_centroid_postings_index.index.document_count
        != segment.stats.document_count
    ):
        raise Error("centroid postings document_count does not match segment stats")


def aggregate_segment_stats(
    read segments: List[LoadedSealedSegment]
) raises -> CollectionStats:
    var document_count = 0
    var token_count = 0
    var total_vector_count = 0
    var byte_size = 0

    for segment in segments:
        document_count += segment.manifest.stats.document_count
        token_count += segment.manifest.stats.token_count
        total_vector_count += segment.manifest.stats.total_vector_count
        byte_size += segment.manifest.stats.byte_size

    return CollectionStats(
        len(segments),
        document_count,
        token_count,
        total_vector_count,
        byte_size,
    )


def require_snapshot_stats_match_loaded_segments(
    read snapshot: SnapshotManifest, read segments: List[LoadedSealedSegment]
) raises:
    var aggregated_stats = aggregate_segment_stats(segments)

    if aggregated_stats.segment_count != snapshot.stats.segment_count:
        raise Error("snapshot segment_count does not match loaded segments")

    if aggregated_stats.document_count != snapshot.stats.document_count:
        raise Error("snapshot document_count does not match loaded segments")

    if aggregated_stats.token_count != snapshot.stats.token_count:
        raise Error("snapshot token_count does not match loaded segments")

    if aggregated_stats.total_vector_count != snapshot.stats.total_vector_count:
        raise Error("snapshot total_vector_count does not match loaded segments")

    if aggregated_stats.byte_size != snapshot.stats.byte_size:
        raise Error("snapshot byte_size does not match loaded segments")


def load_resolved_collection_snapshot(
    collection_root: Path, snapshot_id: SnapshotId
) raises -> ResolvedCollectionSnapshot:
    var collection = load_collection_manifest(collection_root)
    var snapshot = load_snapshot_manifest(
        collection_snapshot_root(collection_root, snapshot_id)
    )
    require_collection_and_snapshot_match(collection, snapshot)

    var loaded_segments = List[LoadedSealedSegment]()

    for segment_id in snapshot.segment_ids:
        var segment_root = collection_segment_root(collection_root, segment_id)
        var segment = load_sealed_segment_manifest(segment_root)
        require_segment_matches_collection(collection, snapshot, segment)

        var stored_index = load_stored_packed_index(
            resolve_segment_artifact_root(
                segment_root,
                segment.packed_index_root,
                "packed_index_root",
            )
        )
        require_loaded_index_matches_segment(
            segment,
            stored_index.index.document_count,
            stored_index.index.total_vector_count,
            stored_index.index.vector_dim,
            stored_index.model_name,
            stored_index.vector_scalar_name,
        )

        var has_centroid_postings_index = sealed_segment_has_centroid_postings_index(
            segment
        )
        var stored_centroid_postings_index = empty_stored_centroid_posting_index(
            segment.model_name,
            segment.vector_scalar_name,
            segment.vector_dim,
        )
        if has_centroid_postings_index:
            stored_centroid_postings_index = load_stored_centroid_posting_index(
                resolve_segment_artifact_root(
                    segment_root,
                    segment.centroid_postings_root,
                    "centroid_postings_root",
                )
            )
            require_loaded_centroid_postings_matches_segment(
                segment, stored_centroid_postings_index
            )

        var has_document_proxy_index = sealed_segment_has_document_proxy_index(segment)
        var stored_document_proxy_index = empty_stored_document_proxy_index(
            segment.model_name,
            segment.vector_scalar_name,
            segment.vector_dim,
        )
        if has_document_proxy_index:
            stored_document_proxy_index = load_stored_document_proxy_index(
                resolve_segment_artifact_root(
                    segment_root,
                    segment.document_proxy_root,
                    "document_proxy_root",
                )
            )
            require_loaded_document_proxy_matches_segment(
                segment, stored_document_proxy_index
            )

        var has_text_corpus = sealed_segment_has_text_corpus(segment)
        var stored_text_corpus = empty_stored_document_text_corpus(
            segment.collection_id,
            segment.segment_id,
        )
        if has_text_corpus:
            stored_text_corpus = load_stored_document_text_corpus(
                resolve_segment_artifact_root(
                    segment_root,
                    segment.text_corpus_root,
                    "text_corpus_root",
                )
            )
            require_loaded_text_corpus_matches_segment(segment, stored_text_corpus)

        loaded_segments.append(
            LoadedSealedSegment(
                segment,
                stored_index,
                has_centroid_postings_index,
                stored_centroid_postings_index,
                has_document_proxy_index,
                stored_document_proxy_index,
                has_text_corpus,
                stored_text_corpus,
            )
        )

    require_snapshot_stats_match_loaded_segments(snapshot, loaded_segments)
    return ResolvedCollectionSnapshot(collection, snapshot, loaded_segments)
