struct WorkloadProfile(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var document_count: Int
    var document_vector_count: Int
    var query_vector_count: Int
    var vector_dim: Int
    var top_k: Int

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        document_count: Int,
        document_vector_count: Int,
        query_vector_count: Int,
        vector_dim: Int,
        top_k: Int,
    ):
        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.document_count = document_count
        self.document_vector_count = document_vector_count
        self.query_vector_count = query_vector_count
        self.vector_dim = vector_dim
        self.top_k = top_k
