# Search-ready loaded view of one collection snapshot.

from std.collections import List

from kayak.storage import StoredPackedIndex

from .collection import CollectionManifest
from .segment import SealedSegmentManifest
from .snapshot import SnapshotManifest
from .text_corpus import StoredDocumentTextCorpus


struct LoadedSealedSegment(Copyable):
    var manifest: SealedSegmentManifest
    var stored_index: StoredPackedIndex
    var has_text_corpus: Bool
    var stored_text_corpus: StoredDocumentTextCorpus

    def __init__(
        out self,
        manifest: SealedSegmentManifest,
        stored_index: StoredPackedIndex,
        has_text_corpus: Bool,
        stored_text_corpus: StoredDocumentTextCorpus,
    ):
        self.manifest = manifest.copy()
        self.stored_index = stored_index.copy()
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
