# Canonical document mutation requests for the hosted service boundary.

from std.collections import List

from kayak.collections import CollectionId, NamespaceId, TenantId
from kayak.collections.validation import require_non_empty_string
from kayak.contracts import EncodedDocument


struct UpsertDocument(Copyable):
    var document: EncodedDocument
    var text: String
    var has_text: Bool

    def __init__(out self, document: EncodedDocument, var text: String = "") raises:
        _ = require_non_empty_string(document.doc_id, "document.doc_id")
        self.document = document.copy()
        self.text = text^
        self.has_text = self.text.byte_length() > 0


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
