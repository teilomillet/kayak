from std.pathlib import Path
from std.testing import TestSuite, assert_equal
from std.collections import List

from kayak import (
    EncodedDocument,
    EncodedQuery,
    JudgedQuery,
    JudgedTask,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
)
from kayak import (
    build_stored_hybrid_flat_dim128_index,
    load_stored_judged_task,
    load_stored_hybrid_flat_dim128_index,
    load_stored_packed_index,
    pack_documents,
    save_stored_hybrid_flat_dim128_index,
    save_stored_judged_task,
    save_stored_packed_index,
)


def make_storage_roundtrip_task() raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "storage_invariants",
        "Corruption and compatibility fixture.",
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


def make_dim128_token(seed: Int) -> List[Float32]:
    var values = List[Float32]()

    for index in range(128):
        values.append(Float32(((seed + 1) * (index + 3)) % 17) / 17.0)

    return values^


def test_stored_task_rejects_scalar_type_mismatch() raises:
    var root = Path("/tmp/kayak-storage-scalar-mismatch")
    var stored_task = StoredJudgedTask(
        "mock://storage-scalar-mismatch",
        "mock-model",
        VECTOR_SCALAR_NAME,
        make_storage_roundtrip_task(),
    )
    save_stored_judged_task(root, stored_task)

    var manifest_path = root / "manifest.tsv"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "vector_scalar_name\t" + VECTOR_SCALAR_NAME,
            "vector_scalar_name\tFloat16",
        )
    )

    var raised = False
    try:
        _ = load_stored_judged_task(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_stored_task_rejects_truncated_vector_payload() raises:
    var root = Path("/tmp/kayak-storage-truncated-payload")
    var stored_task = StoredJudgedTask(
        "mock://storage-truncated",
        "mock-model",
        VECTOR_SCALAR_NAME,
        make_storage_roundtrip_task(),
    )
    save_stored_judged_task(root, stored_task)

    var document_vectors_path = root / "document_vectors.bin"
    var truncated = "bad"
    document_vectors_path.write_bytes(truncated.as_bytes())

    var raised = False
    try:
        _ = load_stored_judged_task(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_stored_index_rejects_artifact_kind_mismatch() raises:
    var root = Path("/tmp/kayak-storage-index-kind-mismatch")
    var task = make_storage_roundtrip_task()
    var stored_index = StoredPackedIndex(
        "mock://storage-index-kind-mismatch",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )
    save_stored_packed_index(root, stored_index)

    var manifest_path = root / "manifest.tsv"
    manifest_path.write_text(
        manifest_path.read_text().replace(
            "artifact_kind\tpacked_index", "artifact_kind\tjudged_task"
        )
    )

    var raised = False
    try:
        _ = load_stored_packed_index(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_stored_index_rejects_invalid_binary_vector_payload() raises:
    var root = Path("/tmp/kayak-storage-index-width-mismatch")
    var task = make_storage_roundtrip_task()
    var stored_index = StoredPackedIndex(
        "mock://storage-index-width-mismatch",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )
    save_stored_packed_index(root, stored_index)

    var token_vectors_path = root / "token_vectors.bin"
    var invalid = "broken"
    token_vectors_path.write_bytes(invalid.as_bytes())

    var raised = False
    try:
        _ = load_stored_packed_index(root)
    except:
        raised = True

    assert_equal(raised, True)


def test_stored_hybrid_flat_index_rejects_invalid_scalar_payload() raises:
    var root = Path("/tmp/kayak-storage-hybrid-flat-invalid-payload")
    var stored_packed_index = StoredPackedIndex(
        "mock://storage-hybrid-flat-invalid-payload",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(
            [
                EncodedDocument(
                    "doc-a", [make_dim128_token(0), make_dim128_token(1)]
                ),
                EncodedDocument(
                    "doc-b", [make_dim128_token(2), make_dim128_token(3)]
                ),
            ]
        ),
    )
    var stored_hybrid_index = build_stored_hybrid_flat_dim128_index(
        stored_packed_index
    )
    save_stored_hybrid_flat_dim128_index(root, stored_hybrid_index)

    var token_values_path = root / "token_values.bin"
    var invalid = "broken"
    token_values_path.write_bytes(invalid.as_bytes())

    var raised = False
    try:
        _ = load_stored_hybrid_flat_dim128_index(root)
    except:
        raised = True

    assert_equal(raised, True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
