from std.collections import List
from std.pathlib import Path

from kayak.storage import (
    StoredLatentProxyIndex,
    StoredPackedIndex,
    latent_proxy_storage_byte_size,
    save_stored_latent_proxy_index,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus

from .collection import CollectionManifest
from .collection_store import save_collection_manifest
from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .mirror import (
    mirror_has_any_text,
    packed_index_storage_byte_size,
    require_document_text_corpus_matches_packed_index,
)
from .search_artifact import SearchArtifactManifest, latent_proxy_search_artifact
from .segment import SealedSegmentManifest
from .segment_builder import text_corpus_storage_byte_size
from .segment_store import save_sealed_segment_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import save_snapshot_manifest
from .stats import CollectionStats, SegmentStats
from .text_corpus import StoredDocumentTextCorpus
from .text_corpus_store import save_stored_document_text_corpus


# Owns the narrow collection-mirror path for one packed index plus one latent
# proxy sidecar. It does not build the latent proxy artifact itself.


def require_latent_proxy_matches_packed_index(
    read stored_index: StoredPackedIndex,
    read stored_latent_proxy_index: StoredLatentProxyIndex,
) raises:
    if stored_latent_proxy_index.model_name != stored_index.model_name:
        raise Error(
            "latent proxy model_name must match packed index model_name"
        )

    if stored_latent_proxy_index.vector_scalar_name != stored_index.vector_scalar_name:
        raise Error(
            "latent proxy vector_scalar_name must match packed index vector_scalar_name"
        )

    if stored_latent_proxy_index.input_vector_dim != stored_index.index.vector_dim:
        raise Error(
            "latent proxy input_vector_dim must match packed index vector_dim"
        )

    if (
        stored_latent_proxy_index.index.document_count
        != stored_index.index.document_count
    ):
        raise Error(
            "latent proxy document_count must match packed index document_count"
        )

    for doc_index in range(stored_index.index.document_count):
        if (
            stored_latent_proxy_index.index.doc_ids[doc_index]
            != stored_index.index.doc_ids[doc_index]
        ):
            raise Error(
                "latent proxy doc_id order must match packed index doc_ids"
            )


def ensure_one_segment_collection_mirror_with_latent_proxy(
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    read stored_index: StoredPackedIndex,
    read stored_latent_proxy_index: StoredLatentProxyIndex,
    read document_text_corpus: DocumentTextCorpus,
) raises -> Path:
    require_document_text_corpus_matches_packed_index(
        stored_index,
        document_text_corpus,
    )
    require_latent_proxy_matches_packed_index(
        stored_index,
        stored_latent_proxy_index,
    )

    var segment_id = SegmentId("segment-0001")
    var segment_root = collection_root / "segments" / segment_id.value
    save_collection_manifest(
        collection_root,
        CollectionManifest(
            collection_id,
            tenant_id,
            namespace_id,
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            generation,
        ),
    )
    save_stored_packed_index(segment_root / "packed_index", stored_index.copy())
    save_stored_latent_proxy_index(
        segment_root / "latent_proxy",
        stored_latent_proxy_index.copy(),
    )

    var text_corpus_root_name = String()
    var text_corpus_byte_size = 0
    if (
        len(document_text_corpus.doc_ids) != 0
        and mirror_has_any_text(document_text_corpus)
    ):
        text_corpus_root_name = "text_corpus"
        save_stored_document_text_corpus(
            segment_root / text_corpus_root_name,
            StoredDocumentTextCorpus(
                collection_id,
                segment_id.copy(),
                document_text_corpus,
            ),
        )
        text_corpus_byte_size = text_corpus_storage_byte_size(
            segment_root / text_corpus_root_name,
            stored_index.index.document_count,
        )

    var search_artifacts = List[SearchArtifactManifest]()
    search_artifacts.append(latent_proxy_search_artifact("latent_proxy"))
    var segment_stats = SegmentStats(
        stored_index.index.document_count,
        stored_index.index.total_vector_count,
        stored_index.index.total_vector_count,
        packed_index_storage_byte_size(segment_root / "packed_index")
            + latent_proxy_storage_byte_size(segment_root / "latent_proxy")
            + text_corpus_byte_size,
    )
    save_sealed_segment_manifest(
        segment_root,
        SealedSegmentManifest(
            segment_id.copy(),
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            stored_index.model_name.copy(),
            stored_index.vector_scalar_name.copy(),
            stored_index.index.vector_dim,
            "packed_index",
            search_artifacts^,
            text_corpus_root_name,
            segment_stats.copy(),
        ),
    )
    save_snapshot_manifest(
        collection_root / "snapshots" / snapshot_id.value,
        SnapshotManifest(
            snapshot_id,
            collection_id,
            tenant_id,
            namespace_id,
            generation,
            [segment_id.copy()],
            CollectionStats(
                1,
                segment_stats.document_count,
                segment_stats.token_count,
                segment_stats.total_vector_count,
                segment_stats.byte_size,
            ),
        ),
    )
    return collection_root


def ensure_one_segment_collection_mirror_with_latent_proxy(
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    read stored_index: StoredPackedIndex,
    read stored_latent_proxy_index: StoredLatentProxyIndex,
) raises -> Path:
    return ensure_one_segment_collection_mirror_with_latent_proxy(
        collection_root,
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        generation,
        stored_index,
        stored_latent_proxy_index,
        DocumentTextCorpus([], []),
    )
