# Search-ready loaded view of one collection snapshot.

from std.collections import List

from kayak.storage import (
    StoredCentroidPostingIndex,
    StoredDocumentProxyIndex,
    StoredPackedIndex,
)

from .collection import CollectionManifest
from .segment import SealedSegmentManifest
from .snapshot import SnapshotManifest
from .text_corpus import StoredDocumentTextCorpus


struct LoadedSealedSegment(Copyable):
    var manifest: SealedSegmentManifest
    var stored_index: StoredPackedIndex
    var has_centroid_postings_index: Bool
    var stored_centroid_postings_index: StoredCentroidPostingIndex
    var has_document_proxy_index: Bool
    var stored_document_proxy_index: StoredDocumentProxyIndex
    var has_text_corpus: Bool
    var stored_text_corpus: StoredDocumentTextCorpus

    def __init__(
        out self,
        manifest: SealedSegmentManifest,
        stored_index: StoredPackedIndex,
        has_centroid_postings_index: Bool,
        stored_centroid_postings_index: StoredCentroidPostingIndex,
        has_document_proxy_index: Bool,
        stored_document_proxy_index: StoredDocumentProxyIndex,
        has_text_corpus: Bool,
        stored_text_corpus: StoredDocumentTextCorpus,
    ):
        self.manifest = manifest.copy()
        self.stored_index = stored_index.copy()
        self.has_centroid_postings_index = has_centroid_postings_index
        self.stored_centroid_postings_index = stored_centroid_postings_index.copy()
        self.has_document_proxy_index = has_document_proxy_index
        self.stored_document_proxy_index = stored_document_proxy_index.copy()
        self.has_text_corpus = has_text_corpus
        self.stored_text_corpus = stored_text_corpus.copy()


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
