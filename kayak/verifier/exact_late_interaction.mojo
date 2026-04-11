from std.collections import List

from kayak.contracts import EncodedQuery
from kayak.index import PackedIndex
from kayak.numeric import ScoreScalar
from kayak.scoring.maxsim import exact_score_for_document
from kayak.search import SearchHit
from kayak.search.topk import top_k_hits


def find_document_index(read index: PackedIndex, doc_id: String) raises -> Int:
    for document_index in range(index.document_count):
        if index.doc_ids[document_index] == doc_id:
            return document_index

    raise Error("candidate doc_id not found in index: " + doc_id)


def rerank_hits_exact_late_interaction(
    read query: EncodedQuery,
    read index: PackedIndex,
    read hits: List[SearchHit],
    k: Int,
) raises -> List[SearchHit]:
    var doc_ids = List[String]()
    var scores = List[ScoreScalar]()

    for hit in hits:
        var document_index = find_document_index(index, hit.doc_id)
        doc_ids.append(hit.doc_id.copy())
        scores.append(exact_score_for_document(query, index, document_index))

    return top_k_hits(doc_ids, scores, k)
