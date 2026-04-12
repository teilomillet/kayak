from std.collections import List

from kayak.contracts import EncodedDocument
from kayak.numeric import VectorScalar

from .packed_index import PackedIndex


def unpack_documents(read index: PackedIndex) raises -> List[EncodedDocument]:
    var documents = List[EncodedDocument]()

    for document_index in range(index.document_count):
        var start = index.doc_offsets[document_index]
        var stop = index.doc_offsets[document_index + 1]
        var token_vectors = List[List[VectorScalar]]()

        for token_index in range(start, stop):
            token_vectors.append(index.token_vectors[token_index].copy())

        documents.append(
            EncodedDocument(
                index.doc_ids[document_index].copy(),
                token_vectors^,
            )
        )

    return documents^
