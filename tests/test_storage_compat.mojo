from std.os import makedirs
from std.pathlib import Path
from std.testing import TestSuite, assert_equal

from kayak import (
    EncodedDocument,
    EncodedQuery,
    ExactCpuBackend,
    JudgedQuery,
    JudgedTask,
    StoredJudgedTask,
    StoredPackedIndex,
    VECTOR_SCALAR_NAME,
    load_stored_judged_task,
    load_stored_packed_index,
    pack_documents,
    search_exact,
)
from kayak.storage.manifest import ManifestEntry, write_manifest
from kayak.storage.binary_int_codec import (
    NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE,
    write_non_negative_int_payload_with_encoding,
)
from kayak.storage.binary_vector_codec import write_binary_vector_payload
from kayak.storage.text_codec import (
    append_line,
    encode_vector_line,
    normalize_inline_text,
)


def make_storage_roundtrip_task() raises -> JudgedTask:
    return JudgedTask(
        "mock",
        "storage_compat",
        "Legacy compatibility fixture.",
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


def write_v1_stored_judged_task(root: Path, stored: StoredJudgedTask) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        root / "manifest.tsv",
        [
            ManifestEntry("format_version", "1"),
            ManifestEntry("artifact_kind", "judged_task"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("family", stored.task.family),
            ManifestEntry("slice_name", stored.task.slice_name),
            ManifestEntry("why", normalize_inline_text(stored.task.why)),
            ManifestEntry("primary_metric", stored.task.primary_metric),
            ManifestEntry("k", String(stored.task.k)),
            ManifestEntry(
                "nominal_query_vector_count",
                String(stored.task.nominal_query_vector_count),
            ),
            ManifestEntry(
                "nominal_document_vector_count",
                String(stored.task.nominal_document_vector_count),
            ),
            ManifestEntry("vector_dim", String(stored.task.vector_dim)),
            ManifestEntry("document_count", String(len(stored.task.documents))),
            ManifestEntry("query_count", String(len(stored.task.queries))),
        ],
    )

    var document_lines = String()
    var document_vector_lines = String()
    for document in stored.task.documents:
        append_line(
            document_lines,
            document.doc_id + "\t" + String(document.vector_count),
        )

        for token_vector in document.token_vectors:
            append_line(document_vector_lines, encode_vector_line(token_vector))

    (root / "documents.tsv").write_text(document_lines)
    (root / "document_vectors.tsv").write_text(document_vector_lines)

    var query_lines = String()
    var qrel_lines = String()
    var query_vector_lines = String()

    for query in stored.task.queries:
        append_line(
            query_lines,
            query.query_id
                + "\t"
                + String(query.query.vector_count)
                + "\t"
                + normalize_inline_text(query.description),
        )

        for doc_id in query.relevant_doc_ids:
            append_line(qrel_lines, query.query_id + "\t" + doc_id)

        for token_vector in query.query.token_vectors:
            append_line(query_vector_lines, encode_vector_line(token_vector))

    (root / "queries.tsv").write_text(query_lines)
    (root / "qrels.tsv").write_text(qrel_lines)
    (root / "query_vectors.tsv").write_text(query_vector_lines)


def write_v1_stored_packed_index(root: Path, stored: StoredPackedIndex) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        root / "manifest.tsv",
        [
            ManifestEntry("format_version", "1"),
            ManifestEntry("artifact_kind", "packed_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "total_vector_count", String(stored.index.total_vector_count)
            ),
        ],
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    var doc_offset_lines = String()
    for doc_offset in stored.index.doc_offsets:
        append_line(doc_offset_lines, String(doc_offset))
    (root / "doc_offsets.tsv").write_text(doc_offset_lines)

    var token_vector_lines = String()
    for token_vector in stored.index.token_vectors:
        append_line(token_vector_lines, encode_vector_line(token_vector))
    (root / "token_vectors.tsv").write_text(token_vector_lines)


def test_load_stored_judged_task_supports_v1_text_payloads() raises:
    var root = Path("/tmp/kayak-storage-compat-task-v1")
    var stored_task = StoredJudgedTask(
        "mock://storage-compat-task-v1",
        "mock-model",
        VECTOR_SCALAR_NAME,
        make_storage_roundtrip_task(),
    )

    write_v1_stored_judged_task(root, stored_task)
    var loaded_task = load_stored_judged_task(root)

    assert_equal(loaded_task.dataset_id, "mock://storage-compat-task-v1")
    assert_equal(loaded_task.model_name, "mock-model")
    assert_equal(len(loaded_task.task.documents), 2)
    assert_equal(len(loaded_task.task.queries), 1)
    assert_equal(loaded_task.task.documents[0].doc_id, "doc-a")


def test_load_stored_packed_index_supports_v1_text_payloads() raises:
    var root = Path("/tmp/kayak-storage-compat-index-v1")
    var task = make_storage_roundtrip_task()
    var stored_index = StoredPackedIndex(
        "mock://storage-compat-index-v1",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )

    write_v1_stored_packed_index(root, stored_index)
    var loaded_index = load_stored_packed_index(root)
    var hits = search_exact(
        ExactCpuBackend(), task.queries[0].query, loaded_index.index, task.k
    )

    assert_equal(loaded_index.dataset_id, "mock://storage-compat-index-v1")
    assert_equal(loaded_index.model_name, "mock-model")
    assert_equal(loaded_index.index.document_count, 2)
    assert_equal(hits[0].doc_id, "doc-a")


def write_v2_stored_packed_index_without_doc_offsets_encoding(
    root: Path, stored: StoredPackedIndex
) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        root / "manifest.tsv",
        [
            ManifestEntry("format_version", "2"),
            ManifestEntry("artifact_kind", "packed_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", "binary_le"),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "total_vector_count", String(stored.index.total_vector_count)
            ),
        ],
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    var doc_offset_lines = String()
    for doc_offset in stored.index.doc_offsets:
        append_line(doc_offset_lines, String(doc_offset))
    (root / "doc_offsets.tsv").write_text(doc_offset_lines)

    write_binary_vector_payload(root / "token_vectors.bin", stored.index.token_vectors)


def test_load_stored_packed_index_supports_v2_without_doc_offsets_encoding() raises:
    var root = Path("/tmp/kayak-storage-compat-index-v2-no-offset-encoding")
    var task = make_storage_roundtrip_task()
    var stored_index = StoredPackedIndex(
        "mock://storage-compat-index-v2-no-offset-encoding",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )

    write_v2_stored_packed_index_without_doc_offsets_encoding(root, stored_index)
    var loaded_index = load_stored_packed_index(root)
    var hits = search_exact(
        ExactCpuBackend(), task.queries[0].query, loaded_index.index, task.k
    )

    assert_equal(
        loaded_index.dataset_id,
        "mock://storage-compat-index-v2-no-offset-encoding",
    )
    assert_equal(loaded_index.index.document_count, 2)
    assert_equal(hits[0].doc_id, "doc-a")


def write_v2_stored_packed_index_with_binary_doc_offsets(
    root: Path, stored: StoredPackedIndex
) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        root / "manifest.tsv",
        [
            ManifestEntry("format_version", "2"),
            ManifestEntry("artifact_kind", "packed_index"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("vector_payload_encoding", "binary_le"),
            ManifestEntry(
                "doc_offsets_encoding",
                NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE,
            ),
            ManifestEntry("vector_dim", String(stored.index.vector_dim)),
            ManifestEntry("document_count", String(stored.index.document_count)),
            ManifestEntry(
                "total_vector_count", String(stored.index.total_vector_count)
            ),
        ],
    )

    var doc_id_lines = String()
    for doc_id in stored.index.doc_ids:
        append_line(doc_id_lines, doc_id)
    (root / "doc_ids.tsv").write_text(doc_id_lines)

    write_non_negative_int_payload_with_encoding(
        root / "doc_offsets.bin",
        stored.index.doc_offsets,
        NON_NEGATIVE_INT_PAYLOAD_ENCODING_BINARY_U64_LE,
    )
    write_binary_vector_payload(root / "token_vectors.bin", stored.index.token_vectors)


def test_load_stored_packed_index_supports_v2_binary_doc_offsets() raises:
    var root = Path("/tmp/kayak-storage-compat-index-v2-binary-offsets")
    var task = make_storage_roundtrip_task()
    var stored_index = StoredPackedIndex(
        "mock://storage-compat-index-v2-binary-offsets",
        "mock-model",
        VECTOR_SCALAR_NAME,
        pack_documents(task.documents),
    )

    write_v2_stored_packed_index_with_binary_doc_offsets(root, stored_index)
    var loaded_index = load_stored_packed_index(root)
    var hits = search_exact(
        ExactCpuBackend(), task.queries[0].query, loaded_index.index, task.k
    )

    assert_equal(
        loaded_index.dataset_id,
        "mock://storage-compat-index-v2-binary-offsets",
    )
    assert_equal(loaded_index.index.document_count, 2)
    assert_equal(hits[0].doc_id, "doc-a")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
