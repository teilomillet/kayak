# Search-ready loaded view of one collection snapshot.

from std.collections import List

from kayak.storage import (
    StoredCentroidPostingIndex,
    StoredDocumentProxyIndex,
    StoredGemGraphIndex,
    StoredPackedIndex,
)

from .collection import CollectionManifest
from .document_filter_index import StoredDocumentFilterIndex
from .document_metadata import (
    DocumentMetadataMap,
    StoredDocumentMetadataCorpus,
    empty_document_metadata_map,
)
from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
    SearchArtifactManifest,
)
from .segment import SealedSegmentManifest
from .snapshot import SnapshotManifest
from .text_corpus import StoredDocumentTextCorpus


struct LoadedSearchArtifact(Copyable):
    var manifest: SearchArtifactManifest
    var stored_centroid_postings_index: StoredCentroidPostingIndex
    var stored_document_filter_index: StoredDocumentFilterIndex
    var stored_document_metadata_corpus: StoredDocumentMetadataCorpus
    var stored_document_proxy_index: StoredDocumentProxyIndex
    var stored_gem_graph_index: StoredGemGraphIndex

    def __init__(
        out self,
        manifest: SearchArtifactManifest,
        stored_centroid_postings_index: StoredCentroidPostingIndex,
        stored_document_filter_index: StoredDocumentFilterIndex,
        stored_document_metadata_corpus: StoredDocumentMetadataCorpus,
        stored_document_proxy_index: StoredDocumentProxyIndex,
        stored_gem_graph_index: StoredGemGraphIndex,
    ):
        self.manifest = manifest.copy()
        self.stored_centroid_postings_index = stored_centroid_postings_index.copy()
        self.stored_document_filter_index = stored_document_filter_index.copy()
        self.stored_document_metadata_corpus = (
            stored_document_metadata_corpus.copy()
        )
        self.stored_document_proxy_index = stored_document_proxy_index.copy()
        self.stored_gem_graph_index = stored_gem_graph_index.copy()


def loaded_search_artifact_family(read artifact: LoadedSearchArtifact) -> String:
    return artifact.manifest.family.copy()


def loaded_search_artifact_root(read artifact: LoadedSearchArtifact) -> String:
    return artifact.manifest.root.copy()


def loaded_search_artifact_has_family(
    read artifact: LoadedSearchArtifact, family: String
) -> Bool:
    return artifact.manifest.family == family


def loaded_search_artifact_is_centroid_family(
    read artifact: LoadedSearchArtifact
) -> Bool:
    return (
        artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
        or artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
    )


def loaded_search_artifact_is_document_proxy(
    read artifact: LoadedSearchArtifact
) -> Bool:
    return artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY


def loaded_search_artifact_is_document_metadata(
    read artifact: LoadedSearchArtifact
) -> Bool:
    return artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA


def loaded_search_artifact_is_document_filter_index(
    read artifact: LoadedSearchArtifact
) -> Bool:
    return artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX


def loaded_search_artifact_is_gem_graph(read artifact: LoadedSearchArtifact) -> Bool:
    return artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_GEM_GRAPH


def loaded_search_artifact_stored_centroid_postings_index(
    read artifact: LoadedSearchArtifact
) raises -> StoredCentroidPostingIndex:
    if not loaded_search_artifact_is_centroid_family(artifact):
        raise Error(
            "loaded search artifact family is not a centroid artifact: "
            + artifact.manifest.family
        )

    return artifact.stored_centroid_postings_index.copy()


def loaded_search_artifact_stored_document_filter_index(
    read artifact: LoadedSearchArtifact
) raises -> StoredDocumentFilterIndex:
    if not loaded_search_artifact_is_document_filter_index(artifact):
        raise Error(
            "loaded search artifact family is not a document filter index artifact: "
            + artifact.manifest.family
        )

    return artifact.stored_document_filter_index.copy()


def loaded_search_artifact_stored_document_proxy_index(
    read artifact: LoadedSearchArtifact
) raises -> StoredDocumentProxyIndex:
    if not loaded_search_artifact_is_document_proxy(artifact):
        raise Error(
            "loaded search artifact family is not a document proxy artifact: "
            + artifact.manifest.family
        )

    return artifact.stored_document_proxy_index.copy()


def loaded_search_artifact_stored_document_metadata(
    read artifact: LoadedSearchArtifact
) raises -> StoredDocumentMetadataCorpus:
    if not loaded_search_artifact_is_document_metadata(artifact):
        raise Error(
            "loaded search artifact family is not a document metadata artifact: "
            + artifact.manifest.family
        )

    return artifact.stored_document_metadata_corpus.copy()


def loaded_search_artifact_stored_gem_graph_index(
    read artifact: LoadedSearchArtifact
) raises -> StoredGemGraphIndex:
    if not loaded_search_artifact_is_gem_graph(artifact):
        raise Error(
            "loaded search artifact family is not a gem graph artifact: "
            + artifact.manifest.family
        )

    return artifact.stored_gem_graph_index.copy()


struct LoadedSealedSegment(Copyable):
    var manifest: SealedSegmentManifest
    var stored_index: StoredPackedIndex
    var search_artifacts: List[LoadedSearchArtifact]
    var has_text_corpus: Bool
    var stored_text_corpus: StoredDocumentTextCorpus

    def __init__(
        out self,
        manifest: SealedSegmentManifest,
        stored_index: StoredPackedIndex,
        search_artifacts: List[LoadedSearchArtifact],
        has_text_corpus: Bool,
        stored_text_corpus: StoredDocumentTextCorpus,
    ):
        self.manifest = manifest.copy()
        self.stored_index = stored_index.copy()
        self.search_artifacts = search_artifacts.copy()
        self.has_text_corpus = has_text_corpus
        self.stored_text_corpus = stored_text_corpus.copy()


def loaded_segment_search_artifact(
    read segment: LoadedSealedSegment, family: String
) raises -> LoadedSearchArtifact:
    for artifact in segment.search_artifacts:
        if loaded_search_artifact_has_family(artifact, family):
            return artifact.copy()

    raise Error("loaded segment is missing search artifact family: " + family)


def loaded_segment_search_artifact_families(
    read segment: LoadedSealedSegment
) -> List[String]:
    var families = List[String]()
    for artifact in segment.search_artifacts:
        families.append(loaded_search_artifact_family(artifact))
    return families^


def loaded_segment_has_search_artifact(
    read segment: LoadedSealedSegment, family: String
) -> Bool:
    for artifact in segment.search_artifacts:
        if loaded_search_artifact_has_family(artifact, family):
            return True

    return False


def loaded_segment_has_centroid_postings_index(
    read segment: LoadedSealedSegment
) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS
    )


def loaded_segment_has_centroid_heads_index(
    read segment: LoadedSealedSegment
) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS
    )


def loaded_segment_has_document_proxy_index(
    read segment: LoadedSealedSegment
) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
    )


def loaded_segment_has_document_metadata(
    read segment: LoadedSealedSegment
) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA
    )


def loaded_segment_has_document_filter_index(
    read segment: LoadedSealedSegment
) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX
    )


def loaded_segment_has_gem_graph_index(read segment: LoadedSealedSegment) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_GEM_GRAPH
    )


def loaded_segment_stored_centroid_postings_index(
    read segment: LoadedSealedSegment,
    family: String = SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
) raises -> StoredCentroidPostingIndex:
    return loaded_search_artifact_stored_centroid_postings_index(
        loaded_segment_search_artifact(segment, family)
    )


def loaded_segment_stored_document_proxy_index(
    read segment: LoadedSealedSegment
) raises -> StoredDocumentProxyIndex:
    return loaded_search_artifact_stored_document_proxy_index(
        loaded_segment_search_artifact(
            segment,
            SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
        )
    )


def loaded_segment_stored_document_metadata(
    read segment: LoadedSealedSegment
) raises -> StoredDocumentMetadataCorpus:
    return loaded_search_artifact_stored_document_metadata(
        loaded_segment_search_artifact(
            segment,
            SEARCH_ARTIFACT_FAMILY_DOCUMENT_METADATA,
        )
    )


def loaded_segment_stored_document_filter_index(
    read segment: LoadedSealedSegment
) raises -> StoredDocumentFilterIndex:
    return loaded_search_artifact_stored_document_filter_index(
        loaded_segment_search_artifact(
            segment,
            SEARCH_ARTIFACT_FAMILY_DOCUMENT_FILTER_INDEX,
        )
    )


def loaded_segment_document_metadata_for_doc_index(
    read segment: LoadedSealedSegment,
    document_index: Int,
) raises -> DocumentMetadataMap:
    if document_index < 0:
        raise Error("document_index must be non-negative")

    if not loaded_segment_has_document_metadata(segment):
        return empty_document_metadata_map()

    var stored_document_metadata = loaded_segment_stored_document_metadata(segment)
    if document_index >= len(stored_document_metadata.metadata_maps):
        raise Error("document_index exceeds loaded document metadata corpus")

    return stored_document_metadata.metadata_maps[document_index].copy()


def loaded_segment_stored_gem_graph_index(
    read segment: LoadedSealedSegment
) raises -> StoredGemGraphIndex:
    return loaded_search_artifact_stored_gem_graph_index(
        loaded_segment_search_artifact(
            segment,
            SEARCH_ARTIFACT_FAMILY_GEM_GRAPH,
        )
    )


struct ResolvedCollectionSnapshot(Copyable):
    var collection: CollectionManifest
    var snapshot: SnapshotManifest
    var segments: List[LoadedSealedSegment]

    def __init__(
        out self,
        collection: CollectionManifest,
        snapshot: SnapshotManifest,
        segments: List[LoadedSealedSegment],
    ):
        self.collection = collection.copy()
        self.snapshot = snapshot.copy()
        self.segments = segments.copy()
