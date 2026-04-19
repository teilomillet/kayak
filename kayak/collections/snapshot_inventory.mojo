from std.collections import List
from std.pathlib import Path

from .collection_store import load_collection_manifest
from .document_encoder_compression import (
    DocumentEncoderCompressionManifest,
    same_document_encoder_compression_manifest,
)
from .ids import SnapshotId
from .paths import collection_segment_root, collection_snapshot_root
from .search_artifact import SearchArtifactManifest
from .segment import SealedSegmentManifest
from .segment_store import load_sealed_segment_manifest
from .snapshot_store import load_snapshot_manifest
from .validation import require_non_empty_string, require_non_negative_int


def append_unique_string(mut values: List[String], value: String) raises:
    var normalized = require_non_empty_string(value, "snapshot inventory family")
    for existing in values:
        if existing == normalized:
            return

    values.append(normalized)


def require_snapshot_matches_collection(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    snapshot_collection_id: String,
    snapshot_tenant_id: String,
    snapshot_namespace_id: String,
) raises:
    if collection_id != snapshot_collection_id:
        raise Error("snapshot collection_id does not match collection manifest")

    if tenant_id != snapshot_tenant_id:
        raise Error("snapshot tenant_id does not match collection manifest")

    if namespace_id != snapshot_namespace_id:
        raise Error("snapshot namespace_id does not match collection manifest")


def require_segment_matches_collection(
    collection_id: String,
    tenant_id: String,
    namespace_id: String,
    model_name: String,
    vector_scalar_name: String,
    vector_dim: Int,
    read document_encoder_compression: DocumentEncoderCompressionManifest,
    read segment: SealedSegmentManifest,
) raises:
    if collection_id != segment.collection_id.value:
        raise Error("segment collection_id does not match collection manifest")

    if tenant_id != segment.tenant_id.value:
        raise Error("segment tenant_id does not match collection manifest")

    if namespace_id != segment.namespace_id.value:
        raise Error("segment namespace_id does not match collection manifest")

    if model_name != segment.model_name:
        raise Error("segment model_name does not match collection manifest")

    if vector_scalar_name != segment.vector_scalar_name:
        raise Error("segment vector_scalar_name does not match collection manifest")

    if vector_dim != segment.vector_dim:
        raise Error("segment vector_dim does not match collection manifest")
    if not same_document_encoder_compression_manifest(
        segment.document_encoder_compression,
        document_encoder_compression,
    ):
        raise Error(
            "segment document_encoder_compression does not match collection manifest"
        )


def search_artifact_family_on_segment(
    read segment: SealedSegmentManifest, family: String
) -> Bool:
    for artifact in segment.search_artifacts:
        if artifact.family == family:
            return True

    return False


def search_artifact_family_on_all_segments(
    read segments: List[SealedSegmentManifest], family: String
) -> Bool:
    if len(segments) == 0:
        return False

    for segment in segments:
        if not search_artifact_family_on_segment(segment, family):
            return False

    return True


struct SnapshotSearchArtifactAvailability(Copyable):
    var segment_count: Int
    var search_artifact_families_available_on_any_segment: List[String]
    var search_artifact_families_available_on_all_segments: List[String]

    def __init__(
        out self,
        segment_count: Int,
        read search_artifact_families_available_on_any_segment: List[String],
        read search_artifact_families_available_on_all_segments: List[String],
    ) raises:
        self.segment_count = require_non_negative_int(
            segment_count, "snapshot inventory segment_count"
        )
        var any_families = List[String]()
        var all_families = List[String]()

        for family in search_artifact_families_available_on_any_segment:
            append_unique_string(any_families, family)

        for family in search_artifact_families_available_on_all_segments:
            append_unique_string(all_families, family)

        for family in all_families:
            var present_in_any = False
            for any_family in any_families:
                if family == any_family:
                    present_in_any = True
                    break

            if not present_in_any:
                raise Error(
                    "snapshot inventory all-segment family is missing from any-segment family set: "
                    + family
                )

        self.search_artifact_families_available_on_any_segment = any_families^
        self.search_artifact_families_available_on_all_segments = all_families^

    def has_search_artifact_family_on_all_segments(self, family: String) -> Bool:
        for available_family in self.search_artifact_families_available_on_all_segments:
            if available_family == family:
                return True

        return False


def load_snapshot_search_artifact_availability(
    collection_root: Path, snapshot_id: SnapshotId
) raises -> SnapshotSearchArtifactAvailability:
    var collection = load_collection_manifest(collection_root)
    var snapshot = load_snapshot_manifest(
        collection_snapshot_root(collection_root, snapshot_id)
    )
    require_snapshot_matches_collection(
        collection.collection_id.value,
        collection.tenant_id.value,
        collection.namespace_id.value,
        snapshot.collection_id.value,
        snapshot.tenant_id.value,
        snapshot.namespace_id.value,
    )

    var segments = List[SealedSegmentManifest]()
    var families_available_on_any_segment = List[String]()

    for segment_id in snapshot.segment_ids:
        var segment = load_sealed_segment_manifest(
            collection_segment_root(collection_root, segment_id)
        )
        require_segment_matches_collection(
            collection.collection_id.value,
            collection.tenant_id.value,
            collection.namespace_id.value,
            collection.model_name,
            collection.vector_scalar_name,
            collection.vector_dim,
            collection.document_encoder_compression,
            segment,
        )
        for artifact in segment.search_artifacts:
            append_unique_string(
                families_available_on_any_segment, artifact.family
            )

        segments.append(segment^)

    var families_available_on_all_segments = List[String]()
    for family in families_available_on_any_segment:
        if search_artifact_family_on_all_segments(segments, family):
            families_available_on_all_segments.append(family.copy())

    return SnapshotSearchArtifactAvailability(
        len(segments),
        families_available_on_any_segment,
        families_available_on_all_segments,
    )
