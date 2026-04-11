from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.index import PackedIndex, pack_documents
from kayak.numeric import VectorScalar

from .workload_profile import WorkloadProfile


struct ExactSearchFixture(Copyable):
    var query: EncodedQuery
    var index: PackedIndex
    var top_k: Int

    def __init__(
        out self, var query: EncodedQuery, var index: PackedIndex, top_k: Int
    ):
        self.query = query^
        self.index = index^
        self.top_k = top_k


def make_vector(seed_a: Int, seed_b: Int, vector_dim: Int) -> List[VectorScalar]:
    var values = List[VectorScalar]()

    for dim_index in range(vector_dim):
        var numerator = ((seed_a + 1) * (seed_b + 3) * (dim_index + 5)) % 17
        values.append(VectorScalar(Float64(numerator) / 17.0))

    return values^


def make_exact_search_fixture(
    document_count: Int = 128,
    document_vector_count: Int = 16,
    query_vector_count: Int = 8,
    vector_dim: Int = 16,
    top_k: Int = 10,
) raises -> ExactSearchFixture:
    var documents = List[EncodedDocument]()

    for document_index in range(document_count):
        var token_vectors = List[List[VectorScalar]]()

        for vector_index in range(document_vector_count):
            token_vectors.append(
                make_vector(document_index, vector_index, vector_dim)
            )

        documents.append(
            EncodedDocument("doc-" + String(document_index), token_vectors^)
        )

    var query_vectors = List[List[VectorScalar]]()
    for vector_index in range(query_vector_count):
        query_vectors.append(make_vector(1000, vector_index, vector_dim))

    return ExactSearchFixture(
        EncodedQuery(query_vectors^), pack_documents(documents), top_k
    )


def make_exact_search_fixture_for_profile(
    profile: WorkloadProfile
) raises -> ExactSearchFixture:
    return make_exact_search_fixture(
        profile.document_count,
        profile.document_vector_count,
        profile.query_vector_count,
        profile.vector_dim,
        profile.top_k,
    )
