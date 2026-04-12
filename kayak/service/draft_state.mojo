# Mutable draft state used by the hosted collection loop.

from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    CollectionManifest,
    SegmentId,
    StoredDocumentTextCorpus,
    load_stored_document_text_corpus,
    save_stored_document_text_corpus,
)
from kayak.collections.artifact_manifest import (
    read_collection_artifact_manifest,
    require_current_vector_scalar_name,
    write_collection_artifact_manifest,
)
from kayak.contracts import EncodedDocument
from kayak.index import pack_documents, unpack_documents
from kayak.storage import (
    StoredPackedIndex,
    load_stored_packed_index,
    save_stored_packed_index,
)
from kayak.storage.manifest import ManifestEntry, require_manifest_value
from kayak.storage.text_codec import parse_int
from kayak.text import DocumentTextCorpus

from .paths import (
    draft_state_manifest_path,
    draft_state_packed_index_root,
    draft_state_text_corpus_root,
)


struct DraftCollectionState(Copyable):
    var documents: List[EncodedDocument]
    var texts: List[String]

    def __init__(
        out self,
        var documents: List[EncodedDocument],
        var texts: List[String],
    ) raises:
        if len(documents) != len(texts):
            raise Error("draft collection state requires aligned documents and texts")

        self.documents = documents^
        self.texts = texts^

    def document_count(self) -> Int:
        return len(self.documents)

    def is_empty(self) -> Bool:
        return len(self.documents) == 0

    def has_any_text(self) -> Bool:
        for text in self.texts:
            if text.byte_length() != 0:
                return True

        return False


def empty_draft_collection_state() raises -> DraftCollectionState:
    return DraftCollectionState([], [])


def draft_state_exists(draft_root: Path) -> Bool:
    return draft_state_manifest_path(draft_root).exists()


def require_draft_state_matches_collection(
    read entries: List[ManifestEntry],
    read collection: CollectionManifest,
) raises -> Int:
    if require_manifest_value(entries, "collection_id") != collection.collection_id.value:
        raise Error("draft collection_id does not match collection manifest")

    if require_manifest_value(entries, "tenant_id") != collection.tenant_id.value:
        raise Error("draft tenant_id does not match collection manifest")

    if require_manifest_value(entries, "namespace_id") != collection.namespace_id.value:
        raise Error("draft namespace_id does not match collection manifest")

    if require_manifest_value(entries, "model_name") != collection.model_name:
        raise Error("draft model_name does not match collection manifest")

    if require_current_vector_scalar_name(entries, "draft collection state") != collection.vector_scalar_name:
        raise Error("draft vector scalar does not match collection manifest")

    var vector_dim = parse_int(
        require_manifest_value(entries, "vector_dim"), "draft vector_dim"
    )
    if vector_dim != collection.vector_dim:
        raise Error("draft vector_dim does not match collection manifest")

    return parse_int(
        require_manifest_value(entries, "document_count"), "draft document_count"
    )


def save_draft_collection_state(
    draft_root: Path,
    read collection: CollectionManifest,
    read state: DraftCollectionState,
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
            ManifestEntry("document_count", String(state.document_count())),
        ],
    )

    if state.is_empty():
        return

    var doc_ids = List[String]()
    for document in state.documents:
        doc_ids.append(document.doc_id.copy())

    save_stored_packed_index(
        draft_state_packed_index_root(draft_root),
        StoredPackedIndex(
            "collection://draft/" + collection.collection_id.value,
            collection.model_name.copy(),
            collection.vector_scalar_name.copy(),
            pack_documents(state.documents),
        ),
    )
    save_stored_document_text_corpus(
        draft_state_text_corpus_root(draft_root),
        StoredDocumentTextCorpus(
            collection.collection_id,
            SegmentId("draft"),
            DocumentTextCorpus(doc_ids^, state.texts.copy()),
        ),
    )


def load_draft_collection_state(
    draft_root: Path, read collection: CollectionManifest
) raises -> DraftCollectionState:
    if not draft_state_exists(draft_root):
        return empty_draft_collection_state()

    var entries = read_collection_artifact_manifest(
        draft_state_manifest_path(draft_root), "draft_collection_state"
    )
    var document_count = require_draft_state_matches_collection(entries, collection)
    if document_count == 0:
        return empty_draft_collection_state()

    var stored_index = load_stored_packed_index(draft_state_packed_index_root(draft_root))
    var stored_text = load_stored_document_text_corpus(
        draft_state_text_corpus_root(draft_root)
    )
    var documents = unpack_documents(stored_index.index)
    if len(documents) != document_count:
        raise Error("draft document_count does not match stored packed index")

    if len(stored_text.corpus.doc_ids) != document_count:
        raise Error("draft document_count does not match stored text corpus")

    for index in range(document_count):
        if documents[index].doc_id != stored_text.corpus.doc_ids[index]:
            raise Error("draft document ids do not align with stored text corpus")

    return DraftCollectionState(documents^, stored_text.corpus.texts.copy())
