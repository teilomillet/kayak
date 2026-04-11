from std.collections import List
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    MetricScalar,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
)
from kayak import evaluate_task, load_stored_judged_task, load_stored_packed_index
from kayak import pack_documents, save_stored_judged_task, save_stored_packed_index
from kayak import search_exact


def copy_documents(documents: List[EncodedDocument]) -> List[EncodedDocument]:
    var copied = List[EncodedDocument]()
    for document in documents:
        copied.append(document.copy())
    return copied^


def make_storage_roundtrip_task() raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "storage_roundtrip",
        "Roundtrip fixture for task and index persistence.",
        "mrr",
        2,
        2,
        2,
        2,
        [
            EncodedDocument("doc-a", [[1.0, 0.0], [0.0, 1.0]]),
            EncodedDocument("doc-b", [[1.0, 0.0], [0.5, 0.5]]),
        ],
        [
            JudgedQuery(
                "q-1",
                "mock query",
                EncodedQuery([[1.0, 0.0], [0.0, 1.0]]),
                ["doc-a"],
            )
        ],
    )


def test_storage_roundtrip_preserves_task_and_index() raises:
    var root = Path("/tmp/kayak-storage-roundtrip")
    var task_root = root / "judged_task"
    var index_root = root / "packed_index"

    var task = make_storage_roundtrip_task()
    var stored_task = StoredJudgedTask(
        "mock://storage-roundtrip",
        "mock-model",
        VECTOR_SCALAR_NAME,
        task.copy(),
    )
    save_stored_judged_task(task_root, stored_task)

    var loaded_task = load_stored_judged_task(task_root)
    var evaluation = evaluate_task(ExactCpuBackend(), loaded_task.task.copy())

    assert_equal(loaded_task.dataset_id, "mock://storage-roundtrip")
    assert_equal(loaded_task.model_name, "mock-model")
    assert_equal(loaded_task.vector_scalar_name, VECTOR_SCALAR_NAME)
    assert_equal(len(loaded_task.task.documents), 2)
    assert_equal(len(loaded_task.task.queries), 1)
    assert_equal(evaluation.mean_reciprocal_rank, MetricScalar(1.0))

    var stored_index = StoredPackedIndex(
        loaded_task.dataset_id.copy(),
        loaded_task.model_name.copy(),
        loaded_task.vector_scalar_name.copy(),
        pack_documents(copy_documents(loaded_task.task.documents)),
    )
    save_stored_packed_index(index_root, stored_index)

    var loaded_index = load_stored_packed_index(index_root)
    var hits = search_exact(
        ExactCpuBackend(),
        loaded_task.task.queries[0].query.copy(),
        loaded_index.index.copy(),
        loaded_task.task.k,
    )

    assert_equal(loaded_index.dataset_id, "mock://storage-roundtrip")
    assert_equal(loaded_index.model_name, "mock-model")
    assert_equal(loaded_index.index.document_count, 2)
    assert_equal(loaded_index.index.total_vector_count, 4)
    assert_equal(hits[0].doc_id, "doc-a")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
