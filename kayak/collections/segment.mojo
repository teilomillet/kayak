# Search-visible segment contract for hosted late-interaction collections.

from .ids import CollectionId, NamespaceId, SegmentId, TenantId
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
    var document_proxy_root: String
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
        document_proxy_root: String,
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
        self.document_proxy_root = document_proxy_root.copy()
        self.text_corpus_root = text_corpus_root.copy()
        self.stats = stats.copy()


def sealed_segment_has_document_proxy_index(
    read segment: SealedSegmentManifest
) -> Bool:
    return segment.document_proxy_root.byte_length() != 0


def sealed_segment_has_text_corpus(read segment: SealedSegmentManifest) -> Bool:
    return segment.text_corpus_root.byte_length() != 0
