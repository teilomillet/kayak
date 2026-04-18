from std.pathlib import Path

from kayak.storage import (
    save_stored_centroid_heads_index,
    save_stored_centroid_posting_index,
    save_stored_document_proxy_index,
    save_stored_gem_graph_index,
    save_stored_latent_proxy_index,
    save_stored_packed_index,
)

from .collection import CollectionManifest
from .collection_store import (
    collection_manifest_exists,
    load_collection_manifest,
    save_collection_manifest,
)
from .document_filter_index_store import save_stored_document_filter_index
from .document_metadata_store import save_stored_document_metadata_corpus
from .paths import collection_segment_root, collection_snapshot_root
from .resolved_snapshot import (
    LoadedSearchArtifact,
    LoadedSealedSegment,
    ResolvedCollectionSnapshot,
    loaded_search_artifact_stored_centroid_postings_index,
    loaded_search_artifact_stored_document_filter_index,
    loaded_search_artifact_stored_document_metadata,
    loaded_search_artifact_stored_document_proxy_index,
    loaded_search_artifact_stored_gem_graph_index,
    loaded_search_artifact_stored_latent_proxy_index,
)
from .resolver import load_resolved_collection_snapshot
from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
    SEARCH_ARTIFACT_FAMILY_LATENT_PROXY,
    same_search_artifacts,
)
from .search_artifact_policy import same_search_artifact_build_policy
from .segment import SealedSegmentManifest
from .segment_store import (
    load_sealed_segment_manifest,
    save_sealed_segment_manifest,
    sealed_segment_manifest_exists,
)
from .snapshot import SnapshotManifest
from .snapshot_bundle import SnapshotExportBundleManifest
from .snapshot_bundle_store import (
    load_snapshot_export_bundle_manifest,
    save_snapshot_export_bundle_manifest,
    snapshot_export_bundle_manifest_exists,
)
from .snapshot_store import (
    load_snapshot_manifest,
    save_snapshot_manifest,
    snapshot_manifest_exists,
)
from .stats import CollectionStats, SegmentStats
from .text_corpus_store import save_stored_document_text_corpus


def same_segment_stats(
    read left: SegmentStats, read right: SegmentStats
) -> Bool:
    return (
        left.document_count == right.document_count
        and left.token_count == right.token_count
        and left.total_vector_count == right.total_vector_count
        and left.byte_size == right.byte_size
    )


def same_collection_stats(
    read left: CollectionStats, read right: CollectionStats
) -> Bool:
    return (
        left.segment_count == right.segment_count
        and left.document_count == right.document_count
        and left.token_count == right.token_count
        and left.total_vector_count == right.total_vector_count
        and left.byte_size == right.byte_size
    )


def require_collection_manifest_compatible(
    read existing: CollectionManifest, read imported: CollectionManifest
) raises:
    if existing.collection_id.value != imported.collection_id.value:
        raise Error("collection manifest collection_id mismatch during import")

    if existing.tenant_id.value != imported.tenant_id.value:
        raise Error("collection manifest tenant_id mismatch during import")

    if existing.namespace_id.value != imported.namespace_id.value:
        raise Error("collection manifest namespace_id mismatch during import")

    if existing.model_name != imported.model_name:
        raise Error("collection manifest model_name mismatch during import")

    if existing.vector_scalar_name != imported.vector_scalar_name:
        raise Error(
            "collection manifest vector_scalar_name mismatch during import"
        )

    if existing.vector_dim != imported.vector_dim:
        raise Error("collection manifest vector_dim mismatch during import")

    if existing.collection_layout_family != imported.collection_layout_family:
        raise Error(
            "collection manifest collection_layout_family mismatch during import"
        )

    if not same_search_artifact_build_policy(
        existing.search_artifact_build_policy,
        imported.search_artifact_build_policy,
    ):
        raise Error(
            "collection manifest search_artifact_build_policy mismatch during import"
        )


def merged_collection_manifest_for_import(
    read existing: CollectionManifest, read imported: CollectionManifest
) raises -> CollectionManifest:
    require_collection_manifest_compatible(existing, imported)

    var latest_generation = existing.latest_generation
    var active_snapshot_id = existing.active_snapshot_id.copy()
    if imported.latest_generation > latest_generation:
        latest_generation = imported.latest_generation
        active_snapshot_id = imported.active_snapshot_id.copy()
    elif (
        imported.latest_generation == latest_generation
        and active_snapshot_id.byte_length() == 0
    ):
        active_snapshot_id = imported.active_snapshot_id.copy()

    return CollectionManifest(
        existing.collection_id,
        existing.tenant_id,
        existing.namespace_id,
        existing.model_name,
        existing.vector_scalar_name,
        existing.vector_dim,
        latest_generation,
        active_snapshot_id,
        existing.default_keep_latest_inactive_count,
        existing.search_artifact_build_policy,
        existing.collection_layout_family,
    )


def require_snapshot_manifest_compatible(
    read existing: SnapshotManifest, read imported: SnapshotManifest
) raises:
    if existing.snapshot_id.value != imported.snapshot_id.value:
        raise Error("snapshot manifest snapshot_id mismatch during import")

    if existing.collection_id.value != imported.collection_id.value:
        raise Error("snapshot manifest collection_id mismatch during import")

    if existing.tenant_id.value != imported.tenant_id.value:
        raise Error("snapshot manifest tenant_id mismatch during import")

    if existing.namespace_id.value != imported.namespace_id.value:
        raise Error("snapshot manifest namespace_id mismatch during import")

    if existing.generation != imported.generation:
        raise Error("snapshot manifest generation mismatch during import")

    if len(existing.segment_ids) != len(imported.segment_ids):
        raise Error("snapshot manifest segment_ids mismatch during import")

    for index in range(len(existing.segment_ids)):
        if existing.segment_ids[index].value != imported.segment_ids[index].value:
            raise Error("snapshot manifest segment_ids mismatch during import")

    if not same_collection_stats(existing.stats, imported.stats):
        raise Error("snapshot manifest stats mismatch during import")


def require_segment_manifest_compatible(
    read existing: SealedSegmentManifest, read imported: SealedSegmentManifest
) raises:
    if existing.segment_id.value != imported.segment_id.value:
        raise Error("segment manifest segment_id mismatch during import")

    if existing.collection_id.value != imported.collection_id.value:
        raise Error("segment manifest collection_id mismatch during import")

    if existing.tenant_id.value != imported.tenant_id.value:
        raise Error("segment manifest tenant_id mismatch during import")

    if existing.namespace_id.value != imported.namespace_id.value:
        raise Error("segment manifest namespace_id mismatch during import")

    if existing.generation != imported.generation:
        raise Error("segment manifest generation mismatch during import")

    if existing.model_name != imported.model_name:
        raise Error("segment manifest model_name mismatch during import")

    if existing.vector_scalar_name != imported.vector_scalar_name:
        raise Error("segment manifest vector_scalar_name mismatch during import")

    if existing.vector_dim != imported.vector_dim:
        raise Error("segment manifest vector_dim mismatch during import")

    if existing.packed_index_root != imported.packed_index_root:
        raise Error("segment manifest packed_index_root mismatch during import")

    if not same_search_artifacts(
        existing.search_artifacts, imported.search_artifacts
    ):
        raise Error("segment manifest search_artifacts mismatch during import")

    if existing.text_corpus_root != imported.text_corpus_root:
        raise Error("segment manifest text_corpus_root mismatch during import")

    if not same_segment_stats(existing.stats, imported.stats):
        raise Error("segment manifest stats mismatch during import")


def collection_manifest_for_snapshot_bundle(
    read resolved: ResolvedCollectionSnapshot
) raises -> CollectionManifest:
    return CollectionManifest(
        resolved.collection.collection_id,
        resolved.collection.tenant_id,
        resolved.collection.namespace_id,
        resolved.collection.model_name,
        resolved.collection.vector_scalar_name,
        resolved.collection.vector_dim,
        resolved.snapshot.generation,
        resolved.snapshot.snapshot_id.value.copy(),
        resolved.collection.default_keep_latest_inactive_count,
        resolved.collection.search_artifact_build_policy,
        resolved.collection.collection_layout_family,
    )


def bundle_manifest_for_snapshot(
    read resolved: ResolvedCollectionSnapshot
) raises -> SnapshotExportBundleManifest:
    return SnapshotExportBundleManifest(
        resolved.snapshot.snapshot_id,
        resolved.collection.collection_id,
        resolved.collection.tenant_id,
        resolved.collection.namespace_id,
        resolved.snapshot.generation,
        len(resolved.snapshot.segment_ids),
    )


def require_bundle_matches_resolved_snapshot(
    read bundle: SnapshotExportBundleManifest,
    read resolved: ResolvedCollectionSnapshot,
) raises:
    if bundle.snapshot_id.value != resolved.snapshot.snapshot_id.value:
        raise Error("bundle snapshot_id does not match resolved snapshot")

    if bundle.collection_id.value != resolved.collection.collection_id.value:
        raise Error("bundle collection_id does not match resolved snapshot")

    if bundle.tenant_id.value != resolved.collection.tenant_id.value:
        raise Error("bundle tenant_id does not match resolved snapshot")

    if bundle.namespace_id.value != resolved.collection.namespace_id.value:
        raise Error("bundle namespace_id does not match resolved snapshot")

    if bundle.generation != resolved.snapshot.generation:
        raise Error("bundle generation does not match resolved snapshot")

    if bundle.segment_count != len(resolved.snapshot.segment_ids):
        raise Error("bundle segment_count does not match resolved snapshot")


def write_loaded_segment_into_collection_root(
    collection_root: Path, read segment: LoadedSealedSegment
) raises:
    var segment_root = collection_segment_root(
        collection_root, segment.manifest.segment_id
    )

    if sealed_segment_manifest_exists(segment_root):
        require_segment_manifest_compatible(
            load_sealed_segment_manifest(segment_root), segment.manifest
        )

    save_sealed_segment_manifest(segment_root, segment.manifest)
    save_stored_packed_index(
        segment_root / segment.manifest.packed_index_root,
        segment.stored_index,
    )

    for artifact in segment.search_artifacts:
        write_loaded_search_artifact(segment_root, artifact)

    if segment.has_text_corpus:
        save_stored_document_text_corpus(
            segment_root / segment.manifest.text_corpus_root,
            segment.stored_text_corpus,
        )


def write_loaded_search_artifact(
    segment_root: Path, read artifact: LoadedSearchArtifact
) raises:
    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS:
        save_stored_centroid_posting_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_centroid_postings_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS:
        save_stored_centroid_heads_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_centroid_postings_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY:
        save_stored_document_proxy_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_document_proxy_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX:
        save_stored_document_filter_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_document_filter_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_GEM_GRAPH:
        save_stored_gem_graph_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_gem_graph_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_LATENT_PROXY:
        save_stored_latent_proxy_index(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_latent_proxy_index(artifact),
        )
        return

    if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA:
        save_stored_document_metadata_corpus(
            segment_root / artifact.manifest.root,
            loaded_search_artifact_stored_document_metadata(artifact),
        )
        return

    raise Error(
        "unsupported loaded search artifact family while writing segment: "
        + artifact.manifest.family
    )


def export_snapshot_bundle(
    collection_root: Path, snapshot_id: SnapshotId, bundle_root: Path
) raises -> SnapshotExportBundleManifest:
    if collection_manifest_exists(bundle_root):
        raise Error("bundle_root already contains a collection manifest")

    if snapshot_export_bundle_manifest_exists(bundle_root):
        raise Error("bundle_root already contains a snapshot export bundle")

    var resolved = load_resolved_collection_snapshot(collection_root, snapshot_id)
    save_collection_manifest(bundle_root, collection_manifest_for_snapshot_bundle(resolved))
    save_snapshot_manifest(
        collection_snapshot_root(bundle_root, snapshot_id), resolved.snapshot
    )

    for segment in resolved.segments:
        write_loaded_segment_into_collection_root(bundle_root, segment)

    var bundle_manifest = bundle_manifest_for_snapshot(resolved)
    save_snapshot_export_bundle_manifest(bundle_root, bundle_manifest)

    _ = load_resolved_collection_snapshot(bundle_root, snapshot_id)
    return bundle_manifest^


def import_snapshot_bundle(
    bundle_root: Path, collection_root: Path
) raises -> SnapshotExportBundleManifest:
    var bundle_manifest = load_snapshot_export_bundle_manifest(bundle_root)
    var resolved = load_resolved_collection_snapshot(
        bundle_root, bundle_manifest.snapshot_id
    )
    require_bundle_matches_resolved_snapshot(bundle_manifest, resolved)

    var imported_collection = collection_manifest_for_snapshot_bundle(resolved)
    if collection_manifest_exists(collection_root):
        save_collection_manifest(
            collection_root,
            merged_collection_manifest_for_import(
                load_collection_manifest(collection_root), imported_collection
            ),
        )
    else:
        save_collection_manifest(collection_root, imported_collection)

    var target_snapshot_root = collection_snapshot_root(
        collection_root, bundle_manifest.snapshot_id
    )
    if snapshot_manifest_exists(target_snapshot_root):
        require_snapshot_manifest_compatible(
            load_snapshot_manifest(target_snapshot_root), resolved.snapshot
        )

    save_snapshot_manifest(target_snapshot_root, resolved.snapshot)

    for segment in resolved.segments:
        write_loaded_segment_into_collection_root(collection_root, segment)

    _ = load_resolved_collection_snapshot(collection_root, bundle_manifest.snapshot_id)
    return bundle_manifest^
