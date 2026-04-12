from std.pathlib import Path

from kayak.index import build_centroid_head_index

from .centroid_postings_store import (
    CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC,
    CentroidPostingCacheEntry,
    build_stored_centroid_posting_index,
    centroid_postings_index_exists,
    centroid_postings_storage_byte_size,
    load_stored_centroid_posting_index_with_artifact_kind,
    save_stored_centroid_posting_index_with_artifact_kind,
)
from .metadata import StoredCentroidPostingIndex, StoredPackedIndex


comptime CENTROID_HEADS_ARTIFACT_KIND = "centroid_head_index"


def centroid_heads_index_exists(root: Path) -> Bool:
    return centroid_postings_index_exists(root)


def centroid_heads_storage_byte_size(root: Path) raises -> Int:
    return centroid_postings_storage_byte_size(root)


def build_stored_centroid_heads_index(
    read stored_packed_index: StoredPackedIndex,
    centroid_budget: Int,
    posting_cap: Int,
) raises -> StoredCentroidPostingIndex:
    if posting_cap <= 0:
        raise Error("centroid heads posting_cap must be positive")

    var full_index = build_stored_centroid_posting_index(
        stored_packed_index, centroid_budget
    )
    return StoredCentroidPostingIndex(
        stored_packed_index.dataset_id.copy(),
        stored_packed_index.model_name.copy(),
        stored_packed_index.vector_scalar_name.copy(),
        CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC,
        centroid_budget,
        posting_cap,
        0,
        build_centroid_head_index(full_index.index, posting_cap),
    )


def save_stored_centroid_heads_index(
    root: Path,
    read stored: StoredCentroidPostingIndex,
) raises:
    if stored.posting_cap <= 0:
        raise Error("centroid heads posting_cap must be positive")

    save_stored_centroid_posting_index_with_artifact_kind(
        root,
        stored,
        CENTROID_HEADS_ARTIFACT_KIND,
    )


def load_stored_centroid_heads_index(
    root: Path
) raises -> StoredCentroidPostingIndex:
    var stored = load_stored_centroid_posting_index_with_artifact_kind(
        root, CENTROID_HEADS_ARTIFACT_KIND
    )
    if stored.posting_cap <= 0:
        raise Error("stored centroid heads posting_cap must be positive")

    if (
        stored.posting_order_kind
        != CENTROID_POSTINGS_ORDER_WEIGHT_DESC_DOC_ASC
    ):
        raise Error("stored centroid heads must preserve weight-sorted postings")

    return stored^


def ensure_stored_centroid_heads_index(
    root: Path,
    read stored_packed_index: StoredPackedIndex,
    centroid_budget: Int,
    posting_cap: Int,
) raises -> CentroidPostingCacheEntry:
    if posting_cap <= 0:
        raise Error("centroid heads posting_cap must be positive")

    if centroid_heads_index_exists(root):
        var loaded = load_stored_centroid_heads_index(root)
        if (
            loaded.centroid_budget == centroid_budget
            and loaded.posting_cap == posting_cap
            and loaded.dataset_id == stored_packed_index.dataset_id
            and loaded.model_name == stored_packed_index.model_name
            and loaded.index.document_count
                == stored_packed_index.index.document_count
            and loaded.index.vector_dim == stored_packed_index.index.vector_dim
        ):
            return CentroidPostingCacheEntry(loaded.copy(), True)

    var stored_centroid_heads = build_stored_centroid_heads_index(
        stored_packed_index, centroid_budget, posting_cap
    )
    save_stored_centroid_heads_index(root, stored_centroid_heads)
    return CentroidPostingCacheEntry(load_stored_centroid_heads_index(root), False)
