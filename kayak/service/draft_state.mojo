# Mutable draft state used by the hosted collection loop.

from std.collections import List
from std.os import makedirs, remove, rmdir
from std.pathlib import Path

from kayak.collections import (
    CollectionManifest,
    DocumentMetadataMap,
    SegmentId,
    StoredDocumentMetadataCorpus,
    StoredDocumentTextCorpus,
    load_stored_document_metadata_corpus,
    load_stored_document_text_corpus,
    save_stored_document_metadata_corpus,
    save_stored_document_text_corpus,
)
from kayak.collections.artifact_manifest import (
    read_collection_artifact_manifest,
    require_current_vector_scalar_name,
    write_collection_artifact_manifest,
)
from kayak.collections.manifest_util import load_optional_manifest_value
from kayak.contracts import EncodedDocument
from kayak.index import pack_documents, unpack_documents
from kayak.storage import (
    StoredPackedIndex,
    load_stored_packed_index,
    save_stored_packed_index,
)
from kayak.storage.atomic_write import write_text_atomic
from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import append_line, parse_int, read_non_empty_lines
from kayak.text import DocumentTextCorpus

from .paths import (
    draft_state_document_metadata_root,
    draft_state_manifest_path,
    draft_state_mutation_doc_ids_path,
    draft_state_mutation_document_metadata_root,
    draft_state_mutation_manifest_path,
    draft_state_mutation_packed_index_root,
    draft_state_mutation_root,
    draft_state_mutation_text_corpus_root,
    draft_state_mutations_root,
    draft_state_packed_index_root,
    draft_state_text_corpus_root,
)

comptime DRAFT_STATE_STORAGE_KIND_MUTATION_LOG = "mutation_log"
comptime DRAFT_MUTATION_KIND_DELETE = "delete"
comptime DRAFT_MUTATION_KIND_UPSERT = "upsert"


struct DraftCollectionState(Copyable):
    var documents: List[EncodedDocument]
    var texts: List[String]
    var metadata_maps: List[DocumentMetadataMap]

    def __init__(
        out self,
        var documents: List[EncodedDocument],
        var texts: List[String],
        read metadata_maps: List[DocumentMetadataMap],
    ) raises:
        if len(documents) != len(texts) or len(documents) != len(metadata_maps):
            raise Error(
                "draft collection state requires aligned documents, texts, and metadata"
            )

        self.documents = documents^
        self.texts = texts^
        self.metadata_maps = metadata_maps.copy()

    def document_count(self) -> Int:
        return len(self.documents)

    def is_empty(self) -> Bool:
        return len(self.documents) == 0

    def has_any_text(self) -> Bool:
        for text in self.texts:
            if text.byte_length() != 0:
                return True

        return False


struct DraftStateMetadata(Copyable):
    var storage_kind: String
    var document_count: Int
    var mutation_count: Int

    def __init__(
        out self,
        var storage_kind: String,
        document_count: Int,
        mutation_count: Int,
    ) raises:
        if document_count < 0:
            raise Error("draft document_count must be non-negative")

        if mutation_count < 0:
            raise Error("draft mutation_count must be non-negative")

        self.storage_kind = storage_kind^
        self.document_count = document_count
        self.mutation_count = mutation_count


def empty_draft_collection_state() raises -> DraftCollectionState:
    return DraftCollectionState([], [], [])


def draft_state_exists(draft_root: Path) -> Bool:
    return draft_state_manifest_path(draft_root).exists()


def draft_state_has_compacted_baseline(draft_root: Path) -> Bool:
    return draft_state_packed_index_root(draft_root).exists()


def find_document_index(
    read documents: List[EncodedDocument], doc_id: String
) -> Int:
    for index in range(len(documents)):
        if documents[index].doc_id == doc_id:
            return index

    return -1


def encode_optional_draft_artifact_root(root: String) -> String:
    if root.byte_length() == 0:
        return "-"

    return root.copy()


def decode_optional_draft_artifact_root(root: String) -> String:
    if root == "-":
        return ""

    return root.copy()


def require_draft_state_matches_collection(
    read entries: List[ManifestEntry],
    read collection: CollectionManifest,
) raises -> DraftStateMetadata:
    if require_manifest_value(entries, "collection_id") != collection.collection_id.value:
        raise Error("draft collection_id does not match collection manifest")

    if require_manifest_value(entries, "tenant_id") != collection.tenant_id.value:
        raise Error("draft tenant_id does not match collection manifest")

    if require_manifest_value(entries, "namespace_id") != collection.namespace_id.value:
        raise Error("draft namespace_id does not match collection manifest")

    if require_manifest_value(entries, "model_name") != collection.model_name:
        raise Error("draft model_name does not match collection manifest")

    if (
        require_current_vector_scalar_name(entries, "draft collection state")
        != collection.vector_scalar_name
    ):
        raise Error("draft vector scalar does not match collection manifest")

    var vector_dim = parse_int(
        require_manifest_value(entries, "vector_dim"), "draft vector_dim"
    )
    if vector_dim != collection.vector_dim:
        raise Error("draft vector_dim does not match collection manifest")

    var storage_kind = load_optional_manifest_value(entries, "storage_kind")
    var mutation_count = 0
    if storage_kind == DRAFT_STATE_STORAGE_KIND_MUTATION_LOG:
        mutation_count = parse_int(
            require_manifest_value(entries, "mutation_count"),
            "draft mutation_count",
        )

    return DraftStateMetadata(
        storage_kind,
        parse_int(
            require_manifest_value(entries, "document_count"), "draft document_count"
        ),
        mutation_count,
    )


def write_draft_state_manifest(
    draft_root: Path,
    read collection: CollectionManifest,
    document_count: Int,
    mutation_count: Int,
) raises:
    makedirs(draft_root, exist_ok=True)

    write_collection_artifact_manifest(
        draft_state_manifest_path(draft_root),
        "draft_collection_state",
        [
            ManifestEntry("collection_id", collection.collection_id.value),
            ManifestEntry("tenant_id", collection.tenant_id.value),
            ManifestEntry("namespace_id", collection.namespace_id.value),
            ManifestEntry("model_name", collection.model_name),
            ManifestEntry("vector_scalar_name", collection.vector_scalar_name),
            ManifestEntry("vector_dim", String(collection.vector_dim)),
            ManifestEntry("storage_kind", DRAFT_STATE_STORAGE_KIND_MUTATION_LOG),
            ManifestEntry("document_count", String(document_count)),
            ManifestEntry("mutation_count", String(mutation_count)),
        ],
    )


def load_draft_state_metadata(
    draft_root: Path, read collection: CollectionManifest
) raises -> DraftStateMetadata:
    if not draft_state_exists(draft_root):
        return DraftStateMetadata(DRAFT_STATE_STORAGE_KIND_MUTATION_LOG, 0, 0)

    return require_draft_state_matches_collection(
        read_collection_artifact_manifest(
            draft_state_manifest_path(draft_root), "draft_collection_state"
        ),
        collection,
    )


def append_draft_upsert_batch(
    draft_root: Path,
    read collection: CollectionManifest,
    read documents: List[EncodedDocument],
    read texts: List[String],
    read metadata_maps: List[DocumentMetadataMap],
    document_count_after_batch: Int,
) raises:
    if len(documents) != len(texts) or len(documents) != len(metadata_maps):
        raise Error(
            "draft upsert batch requires aligned documents, texts, and metadata"
        )

    if len(documents) == 0:
        return

    var metadata = load_draft_state_metadata(draft_root, collection)
    var next_mutation_index = metadata.mutation_count + 1
    var mutation_root = draft_state_mutation_root(draft_root, next_mutation_index)
    var doc_ids = List[String]()
    for document in documents:
        doc_ids.append(document.doc_id.copy())

    write_collection_artifact_manifest(
        draft_state_mutation_manifest_path(mutation_root),
        "draft_mutation_batch",
        [
            ManifestEntry("collection_id", collection.collection_id.value),
            ManifestEntry("tenant_id", collection.tenant_id.value),
            ManifestEntry("namespace_id", collection.namespace_id.value),
            ManifestEntry("model_name", collection.model_name),
            ManifestEntry("vector_scalar_name", collection.vector_scalar_name),
            ManifestEntry("vector_dim", String(collection.vector_dim)),
            ManifestEntry("mutation_kind", DRAFT_MUTATION_KIND_UPSERT),
            ManifestEntry("mutation_index", String(next_mutation_index)),
            ManifestEntry("document_count", String(len(documents))),
            ManifestEntry(
                "document_metadata_root",
                encode_optional_draft_artifact_root("document_metadata"),
            ),
        ],
    )
    save_stored_packed_index(
        draft_state_mutation_packed_index_root(mutation_root),
        StoredPackedIndex(
            "collection://draft/" + collection.collection_id.value,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            pack_documents(documents),
        ),
    )
    save_stored_document_text_corpus(
        draft_state_mutation_text_corpus_root(mutation_root),
        StoredDocumentTextCorpus(
            collection.collection_id,
            SegmentId("draft-mutation-" + String(next_mutation_index)),
            DocumentTextCorpus(doc_ids.copy(), texts.copy()),
        ),
    )
    save_stored_document_metadata_corpus(
        draft_state_mutation_document_metadata_root(mutation_root),
        StoredDocumentMetadataCorpus(
            collection.collection_id,
            SegmentId("draft-mutation-" + String(next_mutation_index)),
            doc_ids,
            metadata_maps,
        ),
    )
    write_draft_state_manifest(
        draft_root,
        collection,
        document_count_after_batch,
        next_mutation_index,
    )


def append_draft_delete_batch(
    draft_root: Path,
    read collection: CollectionManifest,
    read doc_ids: List[String],
    document_count_after_batch: Int,
) raises:
    var unique_doc_ids = List[String]()
    for doc_id in doc_ids:
        if doc_id.byte_length() == 0:
            raise Error("draft delete batch requires non-empty document ids")

        var already_present = False
        for existing in unique_doc_ids:
            if existing == doc_id:
                already_present = True
                break

        if not already_present:
            unique_doc_ids.append(doc_id.copy())

    if len(unique_doc_ids) == 0:
        return

    var metadata = load_draft_state_metadata(draft_root, collection)
    var next_mutation_index = metadata.mutation_count + 1
    var mutation_root = draft_state_mutation_root(draft_root, next_mutation_index)
    var doc_id_lines = String()
    for doc_id in unique_doc_ids:
        append_line(doc_id_lines, doc_id)

    write_collection_artifact_manifest(
        draft_state_mutation_manifest_path(mutation_root),
        "draft_mutation_batch",
        [
            ManifestEntry("collection_id", collection.collection_id.value),
            ManifestEntry("tenant_id", collection.tenant_id.value),
            ManifestEntry("namespace_id", collection.namespace_id.value),
            ManifestEntry("model_name", collection.model_name),
            ManifestEntry("vector_scalar_name", collection.vector_scalar_name),
            ManifestEntry("vector_dim", String(collection.vector_dim)),
            ManifestEntry("mutation_kind", DRAFT_MUTATION_KIND_DELETE),
            ManifestEntry("mutation_index", String(next_mutation_index)),
            ManifestEntry("document_count", String(len(unique_doc_ids))),
        ],
    )
    write_text_atomic(draft_state_mutation_doc_ids_path(mutation_root), doc_id_lines)
    write_draft_state_manifest(
        draft_root,
        collection,
        document_count_after_batch,
        next_mutation_index,
    )


def load_compacted_draft_baseline(
    draft_root: Path,
    read collection: CollectionManifest,
) raises -> DraftCollectionState:
    var stored_index = load_stored_packed_index(
        draft_state_packed_index_root(draft_root)
    )
    var documents = unpack_documents(stored_index.index)

    var texts = List[String]()
    if draft_state_text_corpus_root(draft_root).exists():
        var stored_text = load_stored_document_text_corpus(
            draft_state_text_corpus_root(draft_root)
        )
        if len(stored_text.corpus.doc_ids) != len(documents):
            raise Error(
                "draft baseline document_count does not match stored text corpus"
            )
        for index in range(len(documents)):
            if documents[index].doc_id != stored_text.corpus.doc_ids[index]:
                raise Error(
                    "draft baseline document ids do not align with stored text corpus"
                )
            texts.append(stored_text.corpus.texts[index].copy())
    else:
        for _ in range(len(documents)):
            texts.append(String())

    var metadata_maps = List[DocumentMetadataMap]()
    if draft_state_document_metadata_root(draft_root).exists():
        var stored_document_metadata = load_stored_document_metadata_corpus(
            draft_state_document_metadata_root(draft_root)
        )
        if len(stored_document_metadata.doc_ids) != len(documents):
            raise Error(
                "draft baseline document_count does not match stored metadata corpus"
            )
        for index in range(len(documents)):
            if documents[index].doc_id != stored_document_metadata.doc_ids[index]:
                raise Error(
                    "draft baseline document ids do not align with stored metadata corpus"
                )
            metadata_maps.append(
                stored_document_metadata.metadata_maps[index].copy()
            )
    else:
        for _ in range(len(documents)):
            metadata_maps.append(DocumentMetadataMap())

    _ = collection
    return DraftCollectionState(documents^, texts^, metadata_maps)


def remove_tree(path: Path) raises:
    if not path.exists():
        return

    if path.is_dir():
        for child in path.listdir():
            remove_tree(path / child)
        rmdir(path)
        return

    remove(path)


def compact_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
) raises -> DraftCollectionState:
    var state = load_draft_collection_state(draft_root, collection)
    save_compacted_draft_collection_state(
        draft_root,
        collection,
        state,
    )
    return state^

def save_compacted_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
    read state: DraftCollectionState,
) raises:
    if state.is_empty():
        write_draft_state_manifest(draft_root, collection, 0, 0)
        remove_tree(draft_state_packed_index_root(draft_root))
        remove_tree(draft_state_text_corpus_root(draft_root))
        remove_tree(draft_state_document_metadata_root(draft_root))
        remove_tree(draft_state_mutations_root(draft_root))
        return

    save_stored_packed_index(
        draft_state_packed_index_root(draft_root),
        StoredPackedIndex(
            "collection://draft/" + collection.collection_id.value,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            pack_documents(state.documents),
        ),
    )

    var doc_ids = List[String]()
    for document in state.documents:
        doc_ids.append(document.doc_id.copy())

    save_stored_document_text_corpus(
        draft_state_text_corpus_root(draft_root),
        StoredDocumentTextCorpus(
            collection.collection_id,
            SegmentId("draft-baseline"),
            DocumentTextCorpus(doc_ids.copy(), state.texts.copy()),
        ),
    )
    save_stored_document_metadata_corpus(
        draft_state_document_metadata_root(draft_root),
        StoredDocumentMetadataCorpus(
            collection.collection_id,
            SegmentId("draft-baseline"),
            doc_ids,
            state.metadata_maps,
        ),
    )

    write_draft_state_manifest(
        draft_root,
        collection,
        state.document_count(),
        0,
    )
    remove_tree(draft_state_mutations_root(draft_root))


def load_legacy_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
    metadata: DraftStateMetadata,
) raises -> DraftCollectionState:
    if metadata.document_count == 0:
        return empty_draft_collection_state()

    var stored_index = load_stored_packed_index(
        draft_state_packed_index_root(draft_root)
    )
    var stored_text = load_stored_document_text_corpus(
        draft_state_text_corpus_root(draft_root)
    )
    var documents = unpack_documents(stored_index.index)
    if len(documents) != metadata.document_count:
        raise Error("draft document_count does not match stored packed index")

    if len(stored_text.corpus.doc_ids) != metadata.document_count:
        raise Error("draft document_count does not match stored text corpus")

    for index in range(metadata.document_count):
        if documents[index].doc_id != stored_text.corpus.doc_ids[index]:
            raise Error("draft document ids do not align with stored text corpus")

    var metadata_maps = List[DocumentMetadataMap]()
    for _ in range(metadata.document_count):
        metadata_maps.append(DocumentMetadataMap())

    return DraftCollectionState(
        documents^,
        stored_text.corpus.texts.copy(),
        metadata_maps,
    )


def load_mutation_log_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
    metadata: DraftStateMetadata,
) raises -> DraftCollectionState:
    if metadata.mutation_count == 0:
        if draft_state_has_compacted_baseline(draft_root):
            return load_compacted_draft_baseline(draft_root, collection)
        return empty_draft_collection_state()

    var documents = List[EncodedDocument]()
    var texts = List[String]()
    var metadata_maps = List[DocumentMetadataMap]()

    if draft_state_has_compacted_baseline(draft_root):
        var baseline = load_compacted_draft_baseline(draft_root, collection)
        documents = baseline.documents.copy()
        texts = baseline.texts.copy()
        metadata_maps = baseline.metadata_maps.copy()

    for mutation_index in range(1, metadata.mutation_count + 1):
        var mutation_root = draft_state_mutation_root(draft_root, mutation_index)
        var entries = read_collection_artifact_manifest(
            draft_state_mutation_manifest_path(mutation_root),
            "draft_mutation_batch",
        )
        var batch_metadata = require_draft_state_matches_collection(entries, collection)
        _ = batch_metadata

        var mutation_kind = require_manifest_value(entries, "mutation_kind")
        if mutation_kind == DRAFT_MUTATION_KIND_UPSERT:
            var expected_document_count = parse_int(
                require_manifest_value(entries, "document_count"),
                "draft upsert document_count",
            )
            var stored_index = load_stored_packed_index(
                draft_state_mutation_packed_index_root(mutation_root)
            )
            var stored_text = load_stored_document_text_corpus(
                draft_state_mutation_text_corpus_root(mutation_root)
            )
            var metadata_root = decode_optional_draft_artifact_root(
                load_optional_manifest_value(entries, "document_metadata_root")
            )
            var stored_document_metadata = StoredDocumentMetadataCorpus(
                collection.collection_id,
                SegmentId("draft-mutation-" + String(mutation_index)),
                [],
                [],
            )
            if metadata_root.byte_length() != 0:
                stored_document_metadata = load_stored_document_metadata_corpus(
                    draft_state_mutation_document_metadata_root(mutation_root)
                )
            var batch_documents = unpack_documents(stored_index.index)
            if len(batch_documents) != expected_document_count:
                raise Error(
                    "draft upsert document_count does not match stored packed index"
                )
            if len(stored_text.corpus.doc_ids) != expected_document_count:
                raise Error(
                    "draft upsert document_count does not match stored text corpus"
                )
            if metadata_root.byte_length() != 0 and (
                len(stored_document_metadata.doc_ids) != expected_document_count
            ):
                raise Error(
                    "draft upsert document_count does not match stored metadata corpus"
                )

            for index in range(expected_document_count):
                if batch_documents[index].doc_id != stored_text.corpus.doc_ids[index]:
                    raise Error(
                        "draft upsert document ids do not align with stored text corpus"
                    )
                if metadata_root.byte_length() != 0 and (
                    batch_documents[index].doc_id
                    != stored_document_metadata.doc_ids[index]
                ):
                    raise Error(
                        "draft upsert document ids do not align with stored metadata corpus"
                    )

                var existing_index = find_document_index(
                    documents, batch_documents[index].doc_id
                )
                var next_metadata = DocumentMetadataMap()
                if metadata_root.byte_length() != 0:
                    next_metadata = stored_document_metadata.metadata_maps[index].copy()
                if existing_index == -1:
                    documents.append(batch_documents[index].copy())
                    texts.append(stored_text.corpus.texts[index].copy())
                    metadata_maps.append(next_metadata.copy())
                else:
                    documents[existing_index] = batch_documents[index].copy()
                    texts[existing_index] = stored_text.corpus.texts[index].copy()
                    metadata_maps[existing_index] = next_metadata.copy()
        elif mutation_kind == DRAFT_MUTATION_KIND_DELETE:
            for doc_id in read_non_empty_lines(
                draft_state_mutation_doc_ids_path(mutation_root)
            ):
                var existing_index = find_document_index(documents, doc_id)
                if existing_index == -1:
                    continue

                var kept_documents = List[EncodedDocument]()
                var kept_texts = List[String]()
                var kept_metadata_maps = List[DocumentMetadataMap]()
                for index in range(len(documents)):
                    if index == existing_index:
                        continue

                    kept_documents.append(documents[index].copy())
                    kept_texts.append(texts[index].copy())
                    kept_metadata_maps.append(metadata_maps[index].copy())

                documents = kept_documents^
                texts = kept_texts^
                metadata_maps = kept_metadata_maps^
        else:
            raise Error("unknown draft mutation kind: " + mutation_kind)

    if len(documents) != metadata.document_count:
        raise Error("draft document_count does not match applied mutation log")

    return DraftCollectionState(documents^, texts^, metadata_maps)


def save_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
    read state: DraftCollectionState,
) raises:
    write_draft_state_manifest(draft_root, collection, 0, 0)
    if state.is_empty():
        remove_tree(draft_state_packed_index_root(draft_root))
        remove_tree(draft_state_text_corpus_root(draft_root))
        remove_tree(draft_state_document_metadata_root(draft_root))
        remove_tree(draft_state_mutations_root(draft_root))
        return

    save_compacted_draft_collection_state(
        draft_root,
        collection,
        state,
    )


def load_draft_collection_state(
    draft_root: Path, read collection: CollectionManifest
) raises -> DraftCollectionState:
    if not draft_state_exists(draft_root):
        return empty_draft_collection_state()

    var metadata = load_draft_state_metadata(draft_root, collection)
    if metadata.storage_kind == DRAFT_STATE_STORAGE_KIND_MUTATION_LOG:
        return load_mutation_log_draft_collection_state(
            draft_root, collection, metadata
        )

    return load_legacy_draft_collection_state(draft_root, collection, metadata)
