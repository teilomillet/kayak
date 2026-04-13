from std.collections import List
from std.pathlib import Path

from kayak.contracts import EncodedDocument
from kayak.index import pack_documents
from kayak.storage import (
    StoredPackedIndex,
    centroid_postings_storage_byte_size,
    document_proxy_storage_byte_size,
    ensure_stored_centroid_posting_index,
    ensure_stored_document_proxy_index,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus

from .collection import CollectionManifest
from .document_metadata import (
    DocumentMetadataMap,
    StoredDocumentMetadataCorpus,
)
from .document_metadata_store import save_stored_document_metadata_corpus
from .ids import SegmentId
from .search_artifact import (
    SearchArtifactManifest,
    SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS,
    SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY,
    document_metadata_search_artifact,
)
from .search_artifact_policy import (
    SearchArtifactBuildSpec,
    build_spec_as_search_artifact_manifest,
)
from .segment import SealedSegmentManifest
from .segment_store import save_sealed_segment_manifest
from .stats import SegmentStats
from .text_corpus import StoredDocumentTextCorpus
from .text_corpus_store import save_stored_document_text_corpus


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "doc_ids.tsv").read_text().byte_length()
    total += (root / "doc_offsets.tsv").read_text().byte_length()
    if (root / "token_vectors.bin").exists():
        total += len((root / "token_vectors.bin").read_bytes())
    else:
        total += (root / "token_vectors.tsv").read_text().byte_length()
    return total


def text_corpus_storage_byte_size(
    root: Path, document_count: Int
) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "entries.tsv").read_text().byte_length()

    for index in range(document_count):
        total += (root / "texts" / (String(index) + ".txt")).read_text().byte_length()

    return total


def document_metadata_storage_byte_size(
    root: Path, document_count: Int
) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "entries.tsv").read_text().byte_length()

    for index in range(document_count):
        total += (
            root / "metadata" / (String(index) + ".tsv")
        ).read_text().byte_length()

    return total


def build_configured_search_artifact(
    segment_root: Path,
    read stored_index: StoredPackedIndex,
    read spec: SearchArtifactBuildSpec,
) raises -> Int:
    if spec.family == SEARCH_ARTIFACT_FAMILY_DOCUMENT_PROXY:
        _ = ensure_stored_document_proxy_index(
            segment_root / spec.root,
            stored_index,
            0,
        )
        return document_proxy_storage_byte_size(segment_root / spec.root)

    if spec.family == SEARCH_ARTIFACT_FAMILY_CENTROID_POSTINGS:
        _ = ensure_stored_centroid_posting_index(
            segment_root / spec.root,
            stored_index,
            0,
        )
        return centroid_postings_storage_byte_size(segment_root / spec.root)

    raise Error(
        "segment sealing does not yet support configured build family: "
        + spec.family
    )


def seal_single_segment(
    collection_root: Path,
    read collection: CollectionManifest,
    segment_id: SegmentId,
    generation: Int,
    read documents: List[EncodedDocument],
    read texts: List[String],
    read metadata_maps: List[DocumentMetadataMap],
) raises -> SealedSegmentManifest:
    if len(documents) == 0:
        raise Error("cannot seal an empty segment")

    if len(documents) != len(texts):
        raise Error("segment seal requires aligned documents and texts")
    if len(documents) != len(metadata_maps):
        raise Error("segment seal requires aligned documents and metadata")

    var packed_index = pack_documents(documents)
    var stored_index = StoredPackedIndex(
        "collection://" + collection.collection_id.value,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        packed_index.copy(),
    )
    var segment_root = collection_root / "segments" / segment_id.value
    var packed_index_root = segment_root / "packed_index"

    save_stored_packed_index(packed_index_root, stored_index.copy())

    var byte_size = packed_index_storage_byte_size(packed_index_root)
    var search_artifacts = List[SearchArtifactManifest]()

    for spec in collection.search_artifact_build_policy.stage1_artifacts:
        byte_size += build_configured_search_artifact(
            segment_root,
            stored_index,
            spec,
        )
        search_artifacts.append(build_spec_as_search_artifact_manifest(spec))

    var text_corpus_root_name = ""
    if len(texts) != 0:
        var doc_ids = List[String]()
        var has_any_text = False
        for index in range(len(documents)):
            doc_ids.append(documents[index].doc_id.copy())
            if texts[index].byte_length() != 0:
                has_any_text = True

        if has_any_text:
            text_corpus_root_name = "text_corpus"
            save_stored_document_text_corpus(
                segment_root / text_corpus_root_name,
                StoredDocumentTextCorpus(
                    collection.collection_id,
                    segment_id.copy(),
                    DocumentTextCorpus(doc_ids^, texts.copy()),
                ),
            )
            byte_size += text_corpus_storage_byte_size(
                segment_root / text_corpus_root_name,
                len(documents),
            )

    var has_any_metadata = False
    for metadata in metadata_maps:
        if not metadata.is_empty():
            has_any_metadata = True
            break

    if has_any_metadata:
        var metadata_root_name = "document_metadata"
        var doc_ids = List[String]()
        for document in documents:
            doc_ids.append(document.doc_id.copy())

        save_stored_document_metadata_corpus(
            segment_root / metadata_root_name,
            StoredDocumentMetadataCorpus(
                collection.collection_id,
                segment_id.copy(),
                doc_ids,
                metadata_maps,
            ),
        )
        byte_size += document_metadata_storage_byte_size(
            segment_root / metadata_root_name,
            len(documents),
        )
        search_artifacts.append(document_metadata_search_artifact(metadata_root_name))

    var manifest = SealedSegmentManifest(
        segment_id,
        collection.collection_id,
        collection.tenant_id,
        collection.namespace_id,
        generation,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        collection.vector_dim,
        "packed_index",
        search_artifacts^,
        text_corpus_root_name,
        SegmentStats(
            packed_index.document_count,
            packed_index.total_vector_count,
            packed_index.total_vector_count,
            byte_size,
        ),
    )
    save_sealed_segment_manifest(segment_root, manifest)
    return manifest^


def seal_single_segment(
    collection_root: Path,
    read collection: CollectionManifest,
    segment_id: SegmentId,
    generation: Int,
    read documents: List[EncodedDocument],
    read texts: List[String],
) raises -> SealedSegmentManifest:
    var empty_metadata_maps = List[DocumentMetadataMap]()
    for _ in range(len(documents)):
        empty_metadata_maps.append(DocumentMetadataMap())

    return seal_single_segment(
        collection_root,
        collection,
        segment_id,
        generation,
        documents,
        texts,
        empty_metadata_maps,
    )
