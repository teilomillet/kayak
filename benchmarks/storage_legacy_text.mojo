from std.os import makedirs
from std.pathlib import Path

from kayak import StoredJudgedTask, StoredPackedIndex
from kayak.storage.manifest import ManifestEntry, write_manifest
from kayak.storage.text_codec import (
    append_line,
    encode_vector_line,
    normalize_inline_text,
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


def file_size_bytes(path: Path) raises -> Int:
    if path.suffix() == ".bin":
        return len(path.read_bytes())

    return path.read_text().byte_length()


def judged_task_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "documents.tsv")
    total += file_size_bytes(root / "queries.tsv")
    total += file_size_bytes(root / "qrels.tsv")

    if (root / "document_vectors.bin").exists():
        total += file_size_bytes(root / "document_vectors.bin")
    else:
        total += file_size_bytes(root / "document_vectors.tsv")

    if (root / "query_vectors.bin").exists():
        total += file_size_bytes(root / "query_vectors.bin")
    else:
        total += file_size_bytes(root / "query_vectors.tsv")

    return total


def packed_index_storage_byte_size(root: Path) raises -> Int:
    var total = file_size_bytes(root / "manifest.tsv")
    total += file_size_bytes(root / "doc_ids.tsv")
    total += file_size_bytes(root / "doc_offsets.tsv")

    if (root / "token_vectors.bin").exists():
        total += file_size_bytes(root / "token_vectors.bin")
    else:
        total += file_size_bytes(root / "token_vectors.tsv")

    return total
