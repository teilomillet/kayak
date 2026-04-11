from std.collections import List

from kayak import EncodedDocument, EncodedQuery, ExactCpuBackend, pack_documents
from kayak import search_exact


def main() raises:
    var query = EncodedQuery([[1.0, 0.0], [0.0, 1.0]])

    var documents = List[EncodedDocument]()
    documents.append(
        EncodedDocument(
            "doc-perfect",
            [[1.0, 0.0], [0.0, 1.0]],
        )
    )
    documents.append(
        EncodedDocument(
            "doc-mixed",
            [[1.0, 0.0], [0.5, 0.5]],
        )
    )
    documents.append(
        EncodedDocument(
            "doc-weak",
            [[1.0, 0.0], [0.0, 0.0]],
        )
    )

    var index = pack_documents(documents)
    var backend = ExactCpuBackend()
    var hits = search_exact(backend, query, index, 3)

    for hit in hits:
        print(hit.doc_id, ": ", hit.score)
