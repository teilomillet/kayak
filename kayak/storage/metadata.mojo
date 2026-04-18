from kayak.eval import JudgedTask
from kayak.index import CentroidPostingIndex
from kayak.index import DocumentProxyIndex
from kayak.index import GemGraphIndex
from kayak.index import HybridFlatDim128Index
from kayak.index import LatentProxyIndex
from kayak.index import LatentQueryProjection
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


struct StoredLatentProxyIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var input_vector_dim: Int
    var artifact_byte_size: Int
    var query_projection: LatentQueryProjection
    var index: LatentProxyIndex

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        input_vector_dim: Int,
        artifact_byte_size: Int,
        query_projection: LatentQueryProjection,
        var index: LatentProxyIndex,
    ) raises:
        if input_vector_dim <= 0:
            raise Error("stored latent proxy input_vector_dim must be positive")
        if artifact_byte_size < 0:
            raise Error("stored latent proxy artifact_byte_size must be non-negative")
        if query_projection.input_vector_dim != input_vector_dim:
            raise Error(
                "stored latent proxy input_vector_dim must match the query projection"
            )
        if query_projection.output_vector_dim != index.vector_dim:
            raise Error(
                "stored latent proxy query projection output must match index vector_dim"
            )
        if index.document_count > 0 and len(query_projection.blocks) == 0:
            raise Error(
                "stored latent proxy artifacts with documents must define a query projection"
            )

        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.input_vector_dim = input_vector_dim
        self.artifact_byte_size = artifact_byte_size
        self.query_projection = query_projection.copy()
        self.index = index^


struct StoredGemGraphIndex(Copyable):
    var dataset_id: String
    var model_name: String
    var vector_scalar_name: String
    var cluster_cutoff: Int
    var adaptive_cluster_cutoff_enabled: Bool
    var adaptive_cluster_cutoff_max: Int
    var construction_neighbor_count: Int
    var degree_limit: Int
    var shortcuts_enabled: Bool
    var document_count: Int
    var cluster_count: Int
    var graph_edge_count: Int
    var shortcut_edge_count: Int
    var entry_point_count: Int
    var quantization_centroid_count: Int
    var artifact_byte_size: Int
    var index: GemGraphIndex

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var vector_scalar_name: String,
        cluster_cutoff: Int,
        adaptive_cluster_cutoff_enabled: Bool,
        adaptive_cluster_cutoff_max: Int,
        construction_neighbor_count: Int,
        degree_limit: Int,
        shortcuts_enabled: Bool,
        document_count: Int,
        cluster_count: Int,
        graph_edge_count: Int,
        shortcut_edge_count: Int,
        entry_point_count: Int,
        quantization_centroid_count: Int,
        artifact_byte_size: Int,
        var index: GemGraphIndex,
    ) raises:
        if cluster_cutoff < 0:
            raise Error("stored gem graph cluster_cutoff must be non-negative")
        if adaptive_cluster_cutoff_max < 0:
            raise Error(
                "stored gem graph adaptive_cluster_cutoff_max must be non-negative"
            )
        if construction_neighbor_count < 0:
            raise Error(
                "stored gem graph construction_neighbor_count must be non-negative"
            )
        if degree_limit < 0:
            raise Error("stored gem graph degree_limit must be non-negative")
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
        if index.document_count != document_count:
            raise Error("stored gem graph index document_count must match metadata")
        if index.cluster_count != cluster_count:
            raise Error("stored gem graph index cluster_count must match metadata")
        if index.graph_edge_count != graph_edge_count:
            raise Error("stored gem graph index graph_edge_count must match metadata")
        if index.shortcut_edge_count != shortcut_edge_count:
            raise Error(
                "stored gem graph index shortcut_edge_count must match metadata"
            )
        if index.quantization_centroid_count != quantization_centroid_count:
            raise Error(
                "stored gem graph quantization centroid count must match metadata"
            )
        var derived_entry_point_count = 0
        for entry_doc in index.entry_doc_indices:
            if entry_doc != -1:
                derived_entry_point_count += 1
        if derived_entry_point_count != entry_point_count:
            raise Error(
                "stored gem graph entry_point_count must match index entry docs"
            )
        if index.cluster_cutoff != cluster_cutoff:
            raise Error("stored gem graph cluster_cutoff must match index metadata")
        if (
            index.adaptive_cluster_cutoff_enabled
            != adaptive_cluster_cutoff_enabled
        ):
            raise Error(
                "stored gem graph adaptive_cluster_cutoff_enabled must match index metadata"
            )
        if index.adaptive_cluster_cutoff_max != adaptive_cluster_cutoff_max:
            raise Error(
                "stored gem graph adaptive_cluster_cutoff_max must match index metadata"
            )
        if (
            index.construction_neighbor_count
            != construction_neighbor_count
        ):
            raise Error(
                "stored gem graph construction_neighbor_count must match index metadata"
            )
        if index.degree_limit != degree_limit:
            raise Error("stored gem graph degree_limit must match index metadata")
        if index.shortcuts_enabled != shortcuts_enabled:
            raise Error("stored gem graph shortcuts_enabled must match index metadata")

        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.vector_scalar_name = vector_scalar_name^
        self.cluster_cutoff = cluster_cutoff
        self.adaptive_cluster_cutoff_enabled = adaptive_cluster_cutoff_enabled
        self.adaptive_cluster_cutoff_max = adaptive_cluster_cutoff_max
        self.construction_neighbor_count = construction_neighbor_count
        self.degree_limit = degree_limit
        self.shortcuts_enabled = shortcuts_enabled
        self.document_count = document_count
        self.cluster_count = cluster_count
        self.graph_edge_count = graph_edge_count
        self.shortcut_edge_count = shortcut_edge_count
        self.entry_point_count = entry_point_count
        self.quantization_centroid_count = quantization_centroid_count
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
