from std.collections import List


struct DotProductProfile(Copyable):
    var vector_dim: Int

    def __init__(out self, vector_dim: Int):
        self.vector_dim = vector_dim


struct PerDocumentScoreProfile(Copyable):
    var query_vector_count: Int
    var document_vector_count: Int
    var vector_dim: Int

    def __init__(
        out self,
        query_vector_count: Int,
        document_vector_count: Int,
        vector_dim: Int,
    ):
        self.query_vector_count = query_vector_count
        self.document_vector_count = document_vector_count
        self.vector_dim = vector_dim


struct ExactSearchProfile(Copyable):
    var document_count: Int
    var document_vector_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var top_k: Int

    def __init__(
        out self,
        document_count: Int,
        document_vector_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
    ):
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k


def default_dot_product_profiles() -> List[DotProductProfile]:
    return [DotProductProfile(32), DotProductProfile(128), DotProductProfile(512)]


def default_per_document_score_profiles() -> List[PerDocumentScoreProfile]:
    return [
        PerDocumentScoreProfile(8, 32, 128),
        PerDocumentScoreProfile(16, 64, 128),
        PerDocumentScoreProfile(32, 128, 128),
    ]


def default_exact_search_profiles() -> List[ExactSearchProfile]:
    return [
        ExactSearchProfile(128, 16, 8, 128, 10),
        ExactSearchProfile(512, 32, 16, 128, 10),
        ExactSearchProfile(1024, 64, 32, 128, 10),
    ]
