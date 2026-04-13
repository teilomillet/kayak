from std.collections import List
from std.pathlib import Path

from kayak.storage import (
    StoredPackedIndex,
    build_stored_gem_graph_index,
    centroid_heads_storage_byte_size,
    centroid_postings_storage_byte_size,
    ensure_stored_centroid_heads_index,
    ensure_stored_centroid_posting_index,
    document_proxy_storage_byte_size,
    ensure_stored_document_proxy_index,
    gem_graph_storage_byte_size,
    save_stored_gem_graph_index,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus

from .collection import CollectionManifest
from .collection_store import save_collection_manifest
from .ids import CollectionId, NamespaceId, SegmentId, SnapshotId, TenantId
from .search_artifact import (
    SearchArtifactManifest,
    centroid_heads_search_artifact,
    centroid_postings_search_artifact,
    document_proxy_search_artifact,
    gem_graph_search_artifact,
)
from .segment_builder import text_corpus_storage_byte_size
from .segment import SealedSegmentManifest
from .segment_store import save_sealed_segment_manifest
from .snapshot import SnapshotManifest
from .snapshot_store import save_snapshot_manifest
from .stats import CollectionStats, SegmentStats
from .text_corpus import StoredDocumentTextCorpus
from .text_corpus_store import save_stored_document_text_corpus


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "doc_offsets.tsv")

    if (root / "token_vectors.bin").exists():
        total += file_size_bytes(root / "token_vectors.bin")
    else:
        total += file_size_bytes(root / "token_vectors.tsv")

    return total


def require_document_text_corpus_matches_packed_index(
    read stored_index: StoredPackedIndex,
    read document_text_corpus: DocumentTextCorpus,
) raises:
    if len(document_text_corpus.doc_ids) == 0:
        return

    if len(document_text_corpus.doc_ids) != stored_index.index.document_count:
        raise Error(
            "document text corpus document_count must match packed index document_count"
        )

    for index in range(stored_index.index.document_count):
        if document_text_corpus.doc_ids[index] != stored_index.index.doc_ids[index]:
            raise Error(
                "document text corpus doc_id order must match packed index doc_ids"
            )


def mirror_has_any_text(read document_text_corpus: DocumentTextCorpus) -> Bool:
    for text in document_text_corpus.texts:
        if text.byte_length() != 0:
            return True

    return False


def ensure_one_segment_collection_mirror(
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    read stored_index: StoredPackedIndex,
    read document_text_corpus: DocumentTextCorpus,
    document_proxy_vector_budget: Int = 0,
    centroid_postings_vector_budget: Int = 0,
    centroid_head_posting_cap: Int = 0,
    gem_graph_fine_cluster_count: Int = 0,
    gem_graph_coarse_cluster_count: Int = 0,
    gem_graph_cluster_cutoff: Int = 0,
) raises -> Path:
    require_document_text_corpus_matches_packed_index(
        stored_index,
        document_text_corpus,
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
    _ = ensure_stored_document_proxy_index(
        segment_root / "document_proxy",
        stored_index,
        document_proxy_vector_budget,
    )
    _ = ensure_stored_centroid_posting_index(
        segment_root / "centroid_postings",
        stored_index,
        centroid_postings_vector_budget,
    )
    if centroid_head_posting_cap > 0:
        _ = ensure_stored_centroid_heads_index(
            segment_root / "centroid_heads",
            stored_index,
            centroid_postings_vector_budget,
            centroid_head_posting_cap,
        )
    var gem_graph_byte_size = 0
    if (
        gem_graph_fine_cluster_count > 0
        and gem_graph_coarse_cluster_count > 0
        and gem_graph_cluster_cutoff > 0
    ):
        save_stored_gem_graph_index(
            segment_root / "gem_graph",
            build_stored_gem_graph_index(
                stored_index,
                gem_graph_fine_cluster_count,
                gem_graph_coarse_cluster_count,
                gem_graph_cluster_cutoff,
            ),
        )
        gem_graph_byte_size = gem_graph_storage_byte_size(segment_root / "gem_graph")
    var centroid_heads_byte_size = 0
    if centroid_head_posting_cap > 0:
        centroid_heads_byte_size = centroid_heads_storage_byte_size(
            segment_root / "centroid_heads"
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
    search_artifacts.append(
        centroid_postings_search_artifact("centroid_postings")
    )
    search_artifacts.append(document_proxy_search_artifact("document_proxy"))
    if centroid_head_posting_cap > 0:
        search_artifacts.append(centroid_heads_search_artifact("centroid_heads"))
    if gem_graph_byte_size > 0:
        search_artifacts.append(gem_graph_search_artifact("gem_graph"))

    var segment_stats = SegmentStats(
        stored_index.index.document_count,
        stored_index.index.total_vector_count,
        stored_index.index.total_vector_count,
        packed_index_storage_byte_size(segment_root / "packed_index")
            + centroid_postings_storage_byte_size(segment_root / "centroid_postings")
            + document_proxy_storage_byte_size(segment_root / "document_proxy")
            + centroid_heads_byte_size
            + gem_graph_byte_size
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


def ensure_one_segment_collection_mirror(
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    generation: Int,
    read stored_index: StoredPackedIndex,
    document_proxy_vector_budget: Int = 0,
    centroid_postings_vector_budget: Int = 0,
    centroid_head_posting_cap: Int = 0,
    gem_graph_fine_cluster_count: Int = 0,
    gem_graph_coarse_cluster_count: Int = 0,
    gem_graph_cluster_cutoff: Int = 0,
) raises -> Path:
    return ensure_one_segment_collection_mirror(
        collection_root,
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        generation,
        stored_index,
        DocumentTextCorpus(List[String](), List[String]()),
        document_proxy_vector_budget,
        centroid_postings_vector_budget,
        centroid_head_posting_cap,
        gem_graph_fine_cluster_count,
        gem_graph_coarse_cluster_count,
        gem_graph_cluster_cutoff,
    )
