from kayak.eval import JudgedTask
from kayak.index import DocumentProxyIndex
from kayak.index import HybridFlatDim128Index
from kayak.index import PackedIndex


struct StoredJudgedTask(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var task: JudgedTask

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        var task: JudgedTask,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.task = task^


struct StoredPackedIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var index: PackedIndex

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        var index: PackedIndex,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.index = index^


struct StoredDocumentProxyIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var document_vector_budget: Int
    var proxy_vector_count_per_document: Int
    var artifact_byte_size: Int
    var index: DocumentProxyIndex

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        document_vector_budget: Int,
        proxy_vector_count_per_document: Int,
        artifact_byte_size: Int,
        var index: DocumentProxyIndex,
    ) raises:
        if document_vector_budget < 0:
            raise Error("stored document proxy budget must be non-negative")

        if proxy_vector_count_per_document <= 0:
            raise Error(
                "stored document proxy count per document must be positive"
            )

        if artifact_byte_size < 0:
            raise Error("stored document proxy artifact_byte_size must be non-negative")

        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.document_vector_budget = document_vector_budget
        self.proxy_vector_count_per_document = proxy_vector_count_per_document
        self.artifact_byte_size = artifact_byte_size
        self.index = index^


struct StoredHybridFlatDim128Index(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var index: HybridFlatDim128Index

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        var index: HybridFlatDim128Index,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.index = index^
