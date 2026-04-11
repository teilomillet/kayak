from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.numeric import VectorScalar


def basis_vector(vector_dim: Int, hot_index: Int) raises -> List[VectorScalar]:
    if hot_index < 0 or hot_index >= vector_dim:
        raise Error("concept index must fit within vector_dim")

    var values = List[VectorScalar]()
    for dim_index in range(vector_dim):
        if dim_index == hot_index:
            values.append(VectorScalar(1.0))
        else:
            values.append(VectorScalar(0.0))

    return values^


def concept_vectors(
    vector_dim: Int, concept_ids: List[Int]
) raises -> List[List[VectorScalar]]:
    var vectors = List[List[VectorScalar]]()

    for concept_id in concept_ids:
        vectors.append(basis_vector(vector_dim, concept_id))

    return vectors^


def make_proxy_document(
    doc_id: String, vector_dim: Int, concept_ids: List[Int]
) raises -> EncodedDocument:
    return EncodedDocument(
        doc_id.copy(), concept_vectors(vector_dim, concept_ids)
    )


def make_proxy_query(
    vector_dim: Int, concept_ids: List[Int]
) raises -> EncodedQuery:
    return EncodedQuery(concept_vectors(vector_dim, concept_ids))
