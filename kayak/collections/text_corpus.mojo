# Optional raw-text sidecar for a sealed collection segment.

from kayak.text import DocumentTextCorpus

from .ids import CollectionId, SegmentId


struct StoredDocumentTextCorpus(Copyable):
    var collection_id: CollectionId
    var segment_id: SegmentId
    var corpus: DocumentTextCorpus

    def __init__(
        out self,
        collection_id: CollectionId,
        segment_id: SegmentId,
        corpus: DocumentTextCorpus,
    ):
        self.collection_id = collection_id.copy()
        self.segment_id = segment_id.copy()
        self.corpus = corpus.copy()
