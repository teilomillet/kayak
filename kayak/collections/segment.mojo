# Search-visible segment contract for hosted late-interaction collections.

from std.collections import List

from .document_representation_transform import (
    DocumentRepresentationTransformManifest,
    copy_document_representation_transforms,
    document_representation_transforms_have_kind,
)
from .ids import CollectionId, NamespaceId, SegmentId, TenantId
from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SearchArtifactManifest,
    centroid_heads_search_artifact,
    centroid_postings_search_artifact,
    copy_search_artifacts,
    document_proxy_search_artifact,
    has_search_artifact,
    search_artifact_root,
)
from .stats import SegmentStats
from .validation import require_non_empty_string, require_non_negative_int, require_positive_int


struct SealedSegmentManifest(Copyable):
    var segment_id: SegmentId
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var generation: Int
    var model_name: String
    var vector_scalar_name: String
    var vector_dim: Int
    var packed_index_root: String
    var document_representation_transforms: List[
        DocumentRepresentationTransformManifest
    ]
    var search_artifacts: List[SearchArtifactManifest]
    var text_corpus_root: String
    var stats: SegmentStats

    def __init__(
        out self,
        segment_id: SegmentId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        packed_index_root: String,
        read document_representation_transforms: List[
            DocumentRepresentationTransformManifest
        ],
        search_artifacts: List[SearchArtifactManifest],
        text_corpus_root: String,
        stats: SegmentStats,
    ) raises:
        self.segment_id = segment_id.copy()
        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.generation = require_non_negative_int(generation, "segment generation")
        self.model_name = require_non_empty_string(model_name, "model_name")
        self.vector_scalar_name = require_non_empty_string(
            vector_scalar_name, "vector_scalar_name"
        )
        self.vector_dim = require_positive_int(vector_dim, "vector_dim")
        self.packed_index_root = require_non_empty_string(
            packed_index_root, "packed_index_root"
        )
        self.document_representation_transforms = (
            copy_document_representation_transforms(
                document_representation_transforms
            )
        )
        self.search_artifacts = copy_search_artifacts(search_artifacts)
        self.text_corpus_root = text_corpus_root.copy()
        self.stats = stats.copy()

    def __init__(
        out self,
        segment_id: SegmentId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        packed_index_root: String,
        search_artifacts: List[SearchArtifactManifest],
        text_corpus_root: String,
        stats: SegmentStats,
        read document_representation_transforms: List[
            DocumentRepresentationTransformManifest
        ],
    ) raises:
        self = SealedSegmentManifest(
            segment_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            model_name,
            vector_scalar_name,
            vector_dim,
            packed_index_root,
            document_representation_transforms,
            search_artifacts,
            text_corpus_root,
            stats,
        )

    def __init__(
        out self,
        segment_id: SegmentId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        packed_index_root: String,
        search_artifacts: List[SearchArtifactManifest],
        text_corpus_root: String,
        stats: SegmentStats,
    ) raises:
        self = SealedSegmentManifest(
            segment_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            model_name,
            vector_scalar_name,
            vector_dim,
            packed_index_root,
            [],
            search_artifacts,
            text_corpus_root,
            stats,
        )

    def __init__(
        out self,
        segment_id: SegmentId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        packed_index_root: String,
        centroid_postings_root: String,
        centroid_heads_root: String,
        document_proxy_root: String,
        text_corpus_root: String,
        stats: SegmentStats,
    ) raises:
        var search_artifacts = List[SearchArtifactManifest]()
        if centroid_postings_root.byte_length() != 0:
            search_artifacts.append(
                centroid_postings_search_artifact(centroid_postings_root)
            )
        if centroid_heads_root.byte_length() != 0:
            search_artifacts.append(centroid_heads_search_artifact(centroid_heads_root))
        if document_proxy_root.byte_length() != 0:
            search_artifacts.append(document_proxy_search_artifact(document_proxy_root))

        self = SealedSegmentManifest(
            segment_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            model_name,
            vector_scalar_name,
            vector_dim,
            packed_index_root,
            [],
            search_artifacts^,
            text_corpus_root,
            stats,
        )

    def __init__(
        out self,
        segment_id: SegmentId,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        generation: Int,
        model_name: String,
        vector_scalar_name: String,
        vector_dim: Int,
        packed_index_root: String,
        document_proxy_root: String,
        text_corpus_root: String,
        stats: SegmentStats,
    ) raises:
        self = SealedSegmentManifest(
            segment_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            model_name,
            vector_scalar_name,
            vector_dim,
            packed_index_root,
            "",
            "",
            document_proxy_root,
            text_corpus_root,
            stats,
        )


def sealed_segment_search_artifact_root(
    read segment: SealedSegmentManifest, family: String
) -> String:
    return search_artifact_root(segment.search_artifacts, family)


def sealed_segment_has_search_artifact(
    read segment: SealedSegmentManifest, family: String
) -> Bool:
    return has_search_artifact(segment.search_artifacts, family)


def sealed_segment_centroid_postings_root(
    read segment: SealedSegmentManifest
) -> String:
    return sealed_segment_search_artifact_root(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
    )


def sealed_segment_centroid_heads_root(
    read segment: SealedSegmentManifest
) -> String:
    return sealed_segment_search_artifact_root(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
    )


def sealed_segment_document_proxy_root(
    read segment: SealedSegmentManifest
) -> String:
    return sealed_segment_search_artifact_root(
        segment, SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
    )


def sealed_segment_has_centroid_postings_index(
    read segment: SealedSegmentManifest
) -> Bool:
    return sealed_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
    )


def sealed_segment_has_centroid_heads_index(
    read segment: SealedSegmentManifest
) -> Bool:
    return sealed_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
    )


def sealed_segment_has_document_proxy_index(
    read segment: SealedSegmentManifest
) -> Bool:
    return sealed_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
    )


def sealed_segment_has_text_corpus(read segment: SealedSegmentManifest) -> Bool:
    return segment.text_corpus_root.byte_length() != 0


def sealed_segment_has_document_representation_transforms(
    read segment: SealedSegmentManifest
) -> Bool:
    return len(segment.document_representation_transforms) != 0


def sealed_segment_has_document_representation_transform_kind(
    read segment: SealedSegmentManifest, kind: String
) -> Bool:
    return document_representation_transforms_have_kind(
        segment.document_representation_transforms, kind
    )
