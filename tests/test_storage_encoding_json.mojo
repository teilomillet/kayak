from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    JudgedQuery,
    JudgedTask,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
)
from kayak.benchmarks import (
    StorageEncodingSummary,
    storage_encoding_summary_json,
)
from kayak.index import pack_documents
from kayak.runtime import ExactCpuBackend
from kayak.search import search_exact
from kayak.storage import (
    VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    load_stored_packed_index,
    save_stored_packed_index_with_encoding,
)


def make_storage_encoding_fixture() raises -> StoredJudgedTask:
    return StoredJudgedTask(
        "mock://storage-encoding",
        "mock-model",
        VECTOR_SCALAR_NAME,
        JudgedTask(
            "mock",
            "storage_encoding_fixture",
            "Small exact fixture for storage-encoding summaries.",
            "mrr",
            2,
            2,
            2,
            4,
            [
                EncodedDocument("doc-a", [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                EncodedDocument("doc-b", [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
            ],
            [
                JudgedQuery(
                    "q-1",
                    "mock query one",
                    EncodedQuery([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
                    ["doc-a"],
                ),
                JudgedQuery(
                    "q-2",
                    "mock query two",
                    EncodedQuery([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
                    ["doc-b"],
                ),
            ],
        ),
    )


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    if (root / "doc_offsets.bin").exists():
        total += file_size_bytes(root / "doc_offsets.bin")
    else:
        total += file_size_bytes(root / "doc_offsets.tsv")
    total += file_size_bytes(root / "token_vectors.bin")
    return total


def test_storage_encoding_summary_json_contains_encoding_fields() raises:
    var summary = StorageEncodingSummary(
        "mock://storage-encoding",
        "mock-model",
        "mock",
        "storage_encoding_fixture",
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
        "mrr",
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.001,
        0.002,
        0.003,
        2,
        2,
        4,
        4,
        128,
        64.0,
        32.0,
    )
    var json = storage_encoding_summary_json(summary)

    assert_equal(
        json.find("\"encoding_kind\":\"" + VECTOR_PAYLOAD_ENCODING_BINARY_LE + "\"")
        != -1,
        True,
    )
    assert_equal(json.find("\"artifact_bytes_per_vector\":") != -1, True)
    assert_equal(json.find("\"mean_build_seconds\":") != -1, True)
    assert_equal(json.find("\"mean_load_seconds\":") != -1, True)


def test_storage_encoding_f16_summary_is_smaller_and_keeps_quality() raises:
    var stored_task = make_storage_encoding_fixture()
    var stored_index = StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )
    var binary_root = Path("/tmp/kayak-storage-encoding-summary-binary")
    var f16_root = Path("/tmp/kayak-storage-encoding-summary-f16")
    save_stored_packed_index_with_encoding(
        binary_root,
        stored_index,
        VECTOR_PAYLOAD_ENCODING_BINARY_LE,
    )
    save_stored_packed_index_with_encoding(
        f16_root,
        stored_index,
        VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE,
    )

    var binary_size = packed_index_storage_byte_size(binary_root)
    var f16_size = packed_index_storage_byte_size(f16_root)
    var loaded_f16 = load_stored_packed_index(f16_root)
    var backend = ExactCpuBackend()
    var first_hits = search_exact(
        backend,
        stored_task.task.queries[0].query,
        loaded_f16.index,
        stored_task.task.k,
    )
    var second_hits = search_exact(
        backend,
        stored_task.task.queries[1].query,
        loaded_f16.index,
        stored_task.task.k,
    )

    assert_equal(f16_size < binary_size, True)
    assert_equal(first_hits[0].doc_id, "doc-a")
    assert_equal(second_hits[0].doc_id, "doc-b")
    var manifest_text = (f16_root / "manifest.tsv").read_text()
    assert_equal(
        manifest_text.find(
            "vector_payload_encoding\t" + VECTOR_PAYLOAD_ENCODING_BINARY_F16_LE
        ) != -1,
        True,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
