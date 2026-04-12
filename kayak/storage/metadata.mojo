from kayak.eval import JudgedTask
from kayak.index import CentroidPostingIndex
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


struct StoredCentroidPostingIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var posting_order_kind: String
    var centroid_budget: Int
    var posting_cap: Int
    var artifact_byte_size: Int
    var index: CentroidPostingIndex

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        var posting_order_kind: String,
        centroid_budget: Int,
        posting_cap: Int,
        artifact_byte_size: Int,
        var index: CentroidPostingIndex,
    ) raises:
        if centroid_budget < 0:
            raise Error("stored centroid budget must be non-negative")

        if posting_cap < 0:
            raise Error("stored centroid posting_cap must be non-negative")

        if artifact_byte_size < 0:
            raise Error("stored centroid artifact_byte_size must be non-negative")

        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.posting_order_kind = posting_order_kind^
        self.centroid_budget = centroid_budget
        self.posting_cap = posting_cap
        self.artifact_byte_size = artifact_byte_size
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


struct StoredGemGraphIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var document_count: Int
    var cluster_count: Int
    var graph_edge_count: Int
    var shortcut_edge_count: Int
    var entry_point_count: Int
    var quantization_centroid_count: Int
    var artifact_byte_size: Int

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        document_count: Int,
        cluster_count: Int,
        graph_edge_count: Int,
        shortcut_edge_count: Int,
        entry_point_count: Int,
        quantization_centroid_count: Int,
        artifact_byte_size: Int,
    ) raises:
        if document_count < 0:
            raise Error("stored gem graph document_count must be non-negative")
        if cluster_count < 0:
            raise Error("stored gem graph cluster_count must be non-negative")
        if graph_edge_count < 0:
            raise Error("stored gem graph graph_edge_count must be non-negative")
        if shortcut_edge_count < 0:
            raise Error("stored gem graph shortcut_edge_count must be non-negative")
        if entry_point_count < 0:
            raise Error("stored gem graph entry_point_count must be non-negative")
        if quantization_centroid_count < 0:
            raise Error(
                "stored gem graph quantization_centroid_count must be non-negative"
            )
        if artifact_byte_size < 0:
            raise Error("stored gem graph artifact_byte_size must be non-negative")

        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.document_count = document_count
        self.cluster_count = cluster_count
        self.graph_edge_count = graph_edge_count
        self.shortcut_edge_count = shortcut_edge_count
        self.entry_point_count = entry_point_count
        self.quantization_centroid_count = quantization_centroid_count
        self.artifact_byte_size = artifact_byte_size


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
