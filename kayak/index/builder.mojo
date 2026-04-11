from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.numeric import VectorScalar

from .packed_index import PackedIndex


def pack_documents(documents: List[EncodedDocument]) raises -> PackedIndex:
    if len(documents) == 0:
        raise Error("cannot build an index from zero documents")

    var doc_ids = List[String]()
    var doc_offsets = [0]
    var token_vectors = List[List[VectorScalar]]()
    var vector_dim = documents[0].vector_dim

    for document in documents:
        if document.vector_dim != vector_dim:
            raise Error("all documents must share the same vector dimension")

        doc_ids.append(document.doc_id.copy())

        for token_vector in document.token_vectors:
            token_vectors.append(token_vector.copy())

        doc_offsets.append(len(token_vectors))

    return PackedIndex(doc_ids^, doc_offsets^, token_vectors^, vector_dim)
