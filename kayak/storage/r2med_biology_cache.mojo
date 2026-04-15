from std.pathlib import Path

from kayak.index import pack_documents
from kayak.interop import load_r2med_biology_real_subset
from kayak.numeric import VECTOR_SCALAR_NAME

from .judged_task_store import (
    judged_task_exists,
    load_stored_judged_task,
    save_stored_judged_task,
)
from .metadata import StoredJudgedTask, StoredPackedIndex
from .packed_index_store import (
    load_stored_packed_index,
    packed_index_exists,
    save_stored_packed_index,
)


struct R2medBiologyRealSubsetCache(Copyable):
    var stored_task: StoredJudgedTask
    var stored_index: StoredPackedIndex
    var loaded_task_from_storage: Bool
    var loaded_index_from_storage: Bool

    def __init__(
        out self,
        var stored_task: StoredJudgedTask,
        var stored_index: StoredPackedIndex,
        loaded_task_from_storage: Bool,
        loaded_index_from_storage: Bool,
    ):
        self.stored_task = stored_task^
        self.stored_index = stored_index^
        self.loaded_task_from_storage = loaded_task_from_storage
        self.loaded_index_from_storage = loaded_index_from_storage


def build_r2med_biology_stored_task(
    query_limit: Int,
    negative_doc_limit: Int,
    model_name: String,
) raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "R2MED/Biology",
        model_name.copy(),
        VECTOR_SCALAR_NAME,
        load_r2med_biology_real_subset(
            query_limit,
            negative_doc_limit,
            model_name,
        ),
    )


def build_stored_index_from_task(
    stored_task: StoredJudgedTask
) raises -> StoredPackedIndex:
    return StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )


def load_or_build_r2med_biology_stored_task(
    task_root: Path,
    query_limit: Int,
    negative_doc_limit: Int,
    model_name: String,
) raises -> StoredJudgedTask:
    if judged_task_exists(task_root):
        return load_stored_judged_task(task_root)

    var stored_task = build_r2med_biology_stored_task(
        query_limit,
        negative_doc_limit,
        model_name,
    )
    save_stored_judged_task(task_root, stored_task.copy())
    return stored_task^


def load_or_build_stored_index(
    index_root: Path, stored_task: StoredJudgedTask
) raises -> StoredPackedIndex:
    if packed_index_exists(index_root):
        return load_stored_packed_index(index_root)

    var stored_index = build_stored_index_from_task(stored_task)
    save_stored_packed_index(index_root, stored_index.copy())
    return stored_index^


def ensure_r2med_biology_real_subset_cache(
    cache_root: Path = ".cache/kayak/r2med_biology_real_subset",
    query_limit: Int = 8,
    negative_doc_limit: Int = 128,
    model_name: String = "colbert-ir/colbertv2.0",
) raises -> R2medBiologyRealSubsetCache:
    var task_root = cache_root / "judged_task"
    var index_root = cache_root / "packed_index"

    var loaded_task_from_storage = judged_task_exists(task_root)
    var stored_task = load_or_build_r2med_biology_stored_task(
        task_root,
        query_limit,
        negative_doc_limit,
        model_name,
    )

    var loaded_index_from_storage = packed_index_exists(index_root)
    var stored_index = load_or_build_stored_index(index_root, stored_task)

    return R2medBiologyRealSubsetCache(
        stored_task^,
        stored_index^,
        loaded_task_from_storage,
        loaded_index_from_storage,
    )
