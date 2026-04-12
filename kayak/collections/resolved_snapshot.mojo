# Search-ready loaded view of one collection snapshot.

from std.collections import List

from kayak.storage import (
    StoredCentroidPostingIndex,
    StoredDocumentProxyIndex,
    StoredGemGraphIndex,
    StoredPackedIndex,
)

from .collection import CollectionManifest
from .search_artifact import (
    SEARCH_ARTIFACT_FAMILY_CENTROID_HEADS,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
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
    var stored_document_proxy_index: StoredDocumentProxyIndex
    var stored_gem_graph_index: StoredGemGraphIndex

    def __init__(
        out self,
        manifest: SearchArtifactManifest,
        stored_centroid_postings_index: StoredCentroidPostingIndex,
        stored_document_proxy_index: StoredDocumentProxyIndex,
        stored_gem_graph_index: StoredGemGraphIndex,
    ):
        self.manifest = manifest.copy()
        self.stored_centroid_postings_index = stored_centroid_postings_index.copy()
        self.stored_document_proxy_index = stored_document_proxy_index.copy()
        self.stored_gem_graph_index = stored_gem_graph_index.copy()


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


def loaded_search_artifact_is_gem_graph(read artifact: LoadedSearchArtifact) -> Bool:
    return artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_GEM_GRAPH


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


def loaded_segment_has_search_artifact(
    read segment: LoadedSealedSegment, family: String
) -> Bool:
    for artifact in segment.search_artifacts:
        if artifact.manifest.family == family:
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


def loaded_segment_has_gem_graph_index(read segment: LoadedSealedSegment) -> Bool:
    return loaded_segment_has_search_artifact(
        segment, SEARCH_ARTIFACT_FAMILY_GEM_GRAPH
    )


def loaded_segment_stored_centroid_postings_index(
    read segment: LoadedSealedSegment,
    family: String = SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
) raises -> StoredCentroidPostingIndex:
    for artifact in segment.search_artifacts:
        if artifact.manifest.family == family:
            if not loaded_search_artifact_is_centroid_family(artifact):
                raise Error(
                    "loaded search artifact family is not a centroid artifact: "
                    + family
                )
            return artifact.stored_centroid_postings_index.copy()

    raise Error("loaded segment is missing search artifact family: " + family)


def loaded_segment_stored_document_proxy_index(
    read segment: LoadedSealedSegment
) raises -> StoredDocumentProxyIndex:
    for artifact in segment.search_artifacts:
        if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY:
            if not loaded_search_artifact_is_document_proxy(artifact):
                raise Error(
                    "loaded search artifact family is not a document proxy artifact"
                )
            return artifact.stored_document_proxy_index.copy()

    raise Error(
        "loaded segment is missing search artifact family: "
        + SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY
    )


def loaded_segment_stored_gem_graph_index(
    read segment: LoadedSealedSegment
) raises -> StoredGemGraphIndex:
    for artifact in segment.search_artifacts:
        if artifact.manifest.family == SEARCH_ARTIFACT_FAMILY_GEM_GRAPH:
            if not loaded_search_artifact_is_gem_graph(artifact):
                raise Error(
                    "loaded search artifact family is not a gem graph artifact"
                )
            return artifact.stored_gem_graph_index.copy()

    raise Error(
        "loaded segment is missing search artifact family: "
        + SEARCH_ARTIFACT_FAMILY_GEM_GRAPH
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
