# Canonical document mutation requests for the hosted service boundary.

from std.collections import List

from kayak.collections import CollectionId, NamespaceId, TenantId
from kayak.collections.document_metadata import DocumentMetadataUpdate
from kayak.collections.validation import require_non_empty_string
from kayak.contracts import EncodedDocument


struct UpsertDocument(Copyable):
    var document: EncodedDocument
    var text: String
    var has_text: Bool
    var metadata_updates: List[DocumentMetadataUpdate]
    var has_metadata_updates: Bool

    def __init__(out self, document: EncodedDocument) raises:
        self = UpsertDocument(document, "", [])

    def __init__(out self, document: EncodedDocument, var text: String) raises:
        self = UpsertDocument(document, text^, [])

    def __init__(
        out self,
        document: EncodedDocument,
        read metadata_updates: List[DocumentMetadataUpdate],
    ) raises:
        self = UpsertDocument(document, "", metadata_updates)

    def __init__(
        out self,
        document: EncodedDocument,
        var text: String,
        read metadata_updates: List[DocumentMetadataUpdate],
    ) raises:
        _ = require_non_empty_string(document.doc_id, "document.doc_id")
        self.document = document.copy()
        self.text = text^
        self.has_text = self.text.byte_length() > 0
        self.metadata_updates = metadata_updates.copy()
        self.has_metadata_updates = len(self.metadata_updates) > 0


struct UpsertDocumentsRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var documents: List[UpsertDocument]

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        var documents: List[UpsertDocument],
    ) raises:
        if len(documents) == 0:
            raise Error("documents must contain at least one upsert")

        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.documents = documents^


struct DeleteDocumentsRequest(Copyable):
    var collection_id: CollectionId
    var tenant_id: TenantId
    var namespace_id: NamespaceId
    var doc_ids: List[String]

    def __init__(
        out self,
        collection_id: CollectionId,
        tenant_id: TenantId,
        namespace_id: NamespaceId,
        read doc_ids: List[String],
    ) raises:
        if len(doc_ids) == 0:
            raise Error("doc_ids must contain at least one document id")

        var validated_doc_ids = List[String]()
        for doc_id in doc_ids:
            validated_doc_ids.append(
                require_non_empty_string(doc_id, "delete doc_id")
            )

        self.collection_id = collection_id.copy()
        self.tenant_id = tenant_id.copy()
        self.namespace_id = namespace_id.copy()
        self.doc_ids = validated_doc_ids^
