from kayak.eval import JudgedTask
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
