from std.collections import List
from std.pathlib import Path

from kayak.contracts import EncodedDocument
from kayak.index import pack_documents
from kayak.storage import (
    StoredPackedIndex,
    save_stored_packed_index,
)
from kayak.text import DocumentTextCorpus

from .collection import CollectionManifest
from .document_filter_index import build_stored_document_filter_index
from .document_filter_index_store import (
    document_filter_index_storage_byte_size,
    save_stored_document_filter_index,
)
from .document_representation_transform import DocumentRepresentationTransformManifest
from .document_representation_transform_runtime import (
    apply_document_representation_transforms_to_documents,
)
from .document_metadata import (
    DocumentMetadataMap,
    StoredDocumentMetadataCorpus,
)
from .document_metadata_store import save_stored_document_metadata_corpus
from .ids import SegmentId
from .search_artifact_builders import build_search_artifact_for_segment
from .search_artifact import (
    SearchArtifactManifest,
    document_filter_index_search_artifact,
    document_metadata_search_artifact,
)
from .search_artifact_policy import (
    build_spec_as_search_artifact_manifest,
)
from .search_artifact_builders import build_search_artifact_for_segment
from .segment import SealedSegmentManifest
from .segment_store import save_sealed_segment_manifest
from .stats import SegmentStats
from .text_corpus import StoredDocumentTextCorpus
from .text_corpus_store import save_stored_document_text_corpus


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = (root / "manifest.tsv").read_text().byte_length()
    total += (root / "doc_ids.tsv").read_text().byte_length()
    if (root / "doc_offsets.bin").exists():
        total += len((root / "doc_offsets.bin").read_bytes())
    else:
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


def stored_index_for_segment_documents(
    read collection: CollectionManifest,
    read documents: List[EncodedDocument],
) raises -> StoredPackedIndex:
    var packed_index = pack_documents(documents)
    return StoredPackedIndex(
        "collection://" + collection.collection_id.value,
        collection.model_name.copy(),
        collection.vector_scalar_name.copy(),
        packed_index^,
    )


def seal_single_segment_from_stored_documents(
    collection_root: Path,
    read collection: CollectionManifest,
    segment_id: SegmentId,
    generation: Int,
    read stored_documents: List[EncodedDocument],
    read document_representation_transforms: List[
        DocumentRepresentationTransformManifest
    ],
    read texts: List[String],
    read metadata_maps: List[DocumentMetadataMap],
) raises -> SealedSegmentManifest:
    if len(stored_documents) == 0:
        raise Error("cannot seal an empty segment")

    if len(stored_documents) != len(texts):
        raise Error("segment seal requires aligned documents and texts")
    if len(stored_documents) != len(metadata_maps):
        raise Error("segment seal requires aligned documents and metadata")

    var stored_index = stored_index_for_segment_documents(collection, stored_documents)
    var packed_index = stored_index.index.copy()
    var segment_root = collection_root / "segments" / segment_id.value
    var packed_index_root = segment_root / "packed_index"
    save_stored_packed_index(packed_index_root, stored_index.copy())

    var byte_size = packed_index_storage_byte_size(packed_index_root)
    var search_artifacts = List[SearchArtifactManifest]()
    for spec in collection.search_artifact_build_policy.stage1_artifacts:
        byte_size += build_search_artifact_for_segment(
            segment_root,
            stored_index,
            spec,
        )
        search_artifacts.append(build_spec_as_search_artifact_manifest(spec))

    var document_filter_root_name = "document_filter_index"
    save_stored_document_filter_index(
        segment_root / document_filter_root_name,
        build_stored_document_filter_index(
            collection.collection_id,
            segment_id.copy(),
            collection.tenant_id,
            collection.namespace_id,
            metadata_maps,
        ),
    )
    byte_size += document_filter_index_storage_byte_size(
        segment_root / document_filter_root_name
    )
    search_artifacts.append(
        document_filter_index_search_artifact(document_filter_root_name)
    )

    var text_corpus_root_name = ""
    if len(texts) != 0:
        var doc_ids = List[String]()
        var has_any_text = False
        for index in range(len(stored_documents)):
            doc_ids.append(stored_documents[index].doc_id.copy())
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
                len(stored_documents),
            )

    var has_any_metadata = False
    for metadata in metadata_maps:
        if not metadata.is_empty():
            has_any_metadata = True
            break

    if has_any_metadata:
        var metadata_root_name = "document_metadata"
        var doc_ids = List[String]()
        for document in stored_documents:
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
            len(stored_documents),
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
        document_representation_transforms,
        collection.document_encoder_compression,
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
    read metadata_maps: List[DocumentMetadataMap],
) raises -> SealedSegmentManifest:
    if len(documents) == 0:
        raise Error("cannot seal an empty segment")

    return seal_single_segment_from_stored_documents(
        collection_root,
        collection,
        segment_id,
        generation,
        documents,
        [],
        texts,
        metadata_maps,
    )


def seal_single_segment(
    collection_root: Path,
    read collection: CollectionManifest,
    segment_id: SegmentId,
    generation: Int,
    read documents: List[EncodedDocument],
    read texts: List[String],
    read metadata_maps: List[DocumentMetadataMap],
    read document_representation_transforms: List[
        DocumentRepresentationTransformManifest
    ],
) raises -> SealedSegmentManifest:
    return seal_single_segment_from_stored_documents(
        collection_root,
        collection,
        segment_id,
        generation,
        apply_document_representation_transforms_to_documents(
            documents,
            document_representation_transforms,
        ),
        document_representation_transforms,
        texts,
        metadata_maps,
    )


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


def seal_single_segment(
    collection_root: Path,
    read collection: CollectionManifest,
    segment_id: SegmentId,
    generation: Int,
    read documents: List[EncodedDocument],
    read texts: List[String],
    read document_representation_transforms: List[
        DocumentRepresentationTransformManifest
    ],
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
        document_representation_transforms,
    )
