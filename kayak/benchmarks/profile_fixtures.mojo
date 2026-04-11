from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.index import PackedIndex, pack_documents
from kayak.numeric import VectorScalar

from .fixtures import ExactSearchFixture, make_exact_search_fixture, make_vector
from .profile_shapes import DotProductProfile, ExactSearchProfile, PerDocumentScoreProfile


struct DotProductFixture(Copyable):
    var lhs: List[VectorScalar]
    var rhs: List[VectorScalar]

    def __init__(out self, var lhs: List[VectorScalar], var rhs: List[VectorScalar]):
        self.lhs = lhs^
        self.rhs = rhs^


struct PerDocumentScoreFixture(Copyable):
    var query: EncodedQuery
    var index: PackedIndex

    def __init__(out self, var query: EncodedQuery, var index: PackedIndex):
        self.query = query^
        self.index = index^


def make_dot_product_fixture(profile: DotProductProfile) -> DotProductFixture:
    return DotProductFixture(
        make_vector(7, 11, profile.vector_dim),
        make_vector(13, 17, profile.vector_dim),
    )


def make_per_document_score_fixture(
    profile: PerDocumentScoreProfile
) raises -> PerDocumentScoreFixture:
    var document_vectors = List[List[VectorScalar]]()
    for vector_index in range(profile.document_vector_count):
        document_vectors.append(make_vector(23, vector_index, profile.vector_dim))

    var query_vectors = List[List[VectorScalar]]()
    for vector_index in range(profile.query_vector_count):
        query_vectors.append(make_vector(29, vector_index, profile.vector_dim))

    var index = pack_documents(
        [EncodedDocument("doc-profile", document_vectors^)]
    )

    return PerDocumentScoreFixture(EncodedQuery(query_vectors^), index^)


def make_exact_search_fixture_for_profiled_search(
    profile: ExactSearchProfile
) raises -> ExactSearchFixture:
    return make_exact_search_fixture(
        profile.document_count,
        profile.document_vector_count,
        profile.query_vector_count,
        profile.vector_dim,
        profile.top_k,
    )
