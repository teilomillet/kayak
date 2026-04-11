from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.numeric import STORAGE_FORMAT_VERSION, VECTOR_SCALAR_NAME, VectorScalar

from .binary_vector_codec import (
    read_binary_vector_payload,
    write_binary_vector_payload,
)
from .manifest import (
    ManifestEntry,
    read_manifest,
    require_manifest_value,
    require_supported_storage_format,
    write_manifest,
)
from .metadata import StoredJudgedTask
from .text_codec import (
    append_line,
    decode_vector_line,
    normalize_inline_text,
    parse_int,
    read_non_empty_lines,
    split_tab_fields,
)


struct QrelEntry(Copyable):
    var query_id: String
    var doc_id: String

    def __init__(out self, var query_id: String, var doc_id: String):
        self.query_id = query_id^
        self.doc_id = doc_id^


def task_manifest_path(root: Path) -> Path:
    return root / "manifest.tsv"


def judged_task_exists(root: Path) -> Bool:
    return task_manifest_path(root).exists()


def query_relevant_doc_ids(
    query_id: String, qrels: List[QrelEntry]
) -> List[String]:
    var doc_ids = List[String]()

    for qrel in qrels:
        if qrel.query_id == query_id:
            doc_ids.append(qrel.doc_id.copy())

    return doc_ids^


def save_stored_judged_task(root: Path, stored: StoredJudgedTask) raises:
    makedirs(root, exist_ok=True)

    write_manifest(
        task_manifest_path(root),
        [
            ManifestEntry("format_version", String(STORAGE_FORMAT_VERSION)),
            ManifestEntry("artifact_kind", "judged_task"),
            ManifestEntry("vector_scalar_name", stored.vector_scalar_name),
            ManifestEntry("dataset_id", stored.dataset_id),
            ManifestEntry("model_name", stored.model_name),
            ManifestEntry("family", stored.task.family),
            ManifestEntry("slice_name", stored.task.slice_name),
            ManifestEntry("why", normalize_inline_text(stored.task.why)),
            ManifestEntry("vector_payload_encoding", "binary_le"),
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
    for document in stored.task.documents:
        append_line(
            document_lines,
            document.doc_id + "\t" + String(document.vector_count),
        )

    var documents_path = root / "documents.tsv"
    documents_path.write_text(document_lines)
    write_binary_vector_payload(
        root / "document_vectors.bin", flatten_document_vectors(stored.task.documents)
    )

    var query_lines = String()
    var qrel_lines = String()

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

    var queries_path = root / "queries.tsv"
    queries_path.write_text(query_lines)
    var qrels_path = root / "qrels.tsv"
    qrels_path.write_text(qrel_lines)
    write_binary_vector_payload(
        root / "query_vectors.bin", flatten_query_vectors(stored.task.queries)
    )


def load_stored_judged_task(root: Path) raises -> StoredJudgedTask:
    var manifest = read_manifest(task_manifest_path(root))
    var format_version = require_supported_storage_format(manifest)

    if require_manifest_value(manifest, "artifact_kind") != "judged_task":
        raise Error("storage artifact is not a judged task")

    var vector_dim = parse_int(
        require_manifest_value(manifest, "vector_dim"), "task vector_dim"
    )
    var document_vectors = List[List[VectorScalar]]()
    if format_version >= 2:
        if require_manifest_value(manifest, "vector_payload_encoding") != "binary_le":
            raise Error("unsupported judged task vector payload encoding")

        document_vectors = read_binary_vector_payload(
            root / "document_vectors.bin", vector_dim
        )
    else:
        for line in read_non_empty_lines(root / "document_vectors.tsv"):
            document_vectors.append(decode_vector_line(line))

    var document_vector_cursor = 0
    var documents = List[EncodedDocument]()

    for line in read_non_empty_lines(root / "documents.tsv"):
        var fields = split_tab_fields(line, 2, "document metadata")
        var vector_count = parse_int(fields[1], "document vector_count")
        var token_vectors = List[List[VectorScalar]]()

        for _ in range(vector_count):
            if document_vector_cursor >= len(document_vectors):
                raise Error("document vector payload ended early")

            token_vectors.append(document_vectors[document_vector_cursor].copy())
            document_vector_cursor += 1

        documents.append(EncodedDocument(fields[0], token_vectors^))

    if document_vector_cursor != len(document_vectors):
        raise Error("document vector payload contains extra rows")

    var qrels = List[QrelEntry]()
    for line in read_non_empty_lines(root / "qrels.tsv"):
        var fields = split_tab_fields(line, 2, "qrel row")
        qrels.append(QrelEntry(fields[0], fields[1]))

    var query_vectors = List[List[VectorScalar]]()
    if format_version >= 2:
        query_vectors = read_binary_vector_payload(root / "query_vectors.bin", vector_dim)
    else:
        for line in read_non_empty_lines(root / "query_vectors.tsv"):
            query_vectors.append(decode_vector_line(line))

    var query_vector_cursor = 0
    var queries = List[JudgedQuery]()

    for line in read_non_empty_lines(root / "queries.tsv"):
        var fields = split_tab_fields(line, 3, "query metadata")
        var vector_count = parse_int(fields[1], "query vector_count")
        var token_vectors = List[List[VectorScalar]]()

        for _ in range(vector_count):
            if query_vector_cursor >= len(query_vectors):
                raise Error("query vector payload ended early")

            token_vectors.append(query_vectors[query_vector_cursor].copy())
            query_vector_cursor += 1

        queries.append(
            JudgedQuery(
                fields[0],
                fields[2],
                EncodedQuery(token_vectors^),
                query_relevant_doc_ids(fields[0], qrels),
            )
        )

    if query_vector_cursor != len(query_vectors):
        raise Error("query vector payload contains extra rows")

    var task = JudgedTask(
        require_manifest_value(manifest, "family"),
        require_manifest_value(manifest, "slice_name"),
        require_manifest_value(manifest, "why"),
        require_manifest_value(manifest, "primary_metric"),
        parse_int(require_manifest_value(manifest, "k"), "task k"),
        parse_int(
            require_manifest_value(manifest, "nominal_query_vector_count"),
            "nominal query vector count",
        ),
        parse_int(
            require_manifest_value(manifest, "nominal_document_vector_count"),
            "nominal document vector count",
        ),
        vector_dim,
        documents^,
        queries^,
    )

    if len(task.documents) != parse_int(
        require_manifest_value(manifest, "document_count"), "document count"
    ):
        raise Error("stored task document_count does not match payload")

    if len(task.queries) != parse_int(
        require_manifest_value(manifest, "query_count"), "query count"
    ):
        raise Error("stored task query_count does not match payload")

    return StoredJudgedTask(
        require_manifest_value(manifest, "dataset_id"),
        require_manifest_value(manifest, "model_name"),
        VECTOR_SCALAR_NAME,
        task^,
    )


def flatten_document_vectors(
    read documents: List[EncodedDocument]
) -> List[List[VectorScalar]]:
    var token_vectors = List[List[VectorScalar]]()

    for document in documents:
        for token_vector in document.token_vectors:
            token_vectors.append(token_vector.copy())

    return token_vectors^


def flatten_query_vectors(
    read queries: List[JudgedQuery]
) -> List[List[VectorScalar]]:
    var token_vectors = List[List[VectorScalar]]()

    for query in queries:
        for token_vector in query.query.token_vectors:
            token_vectors.append(token_vector.copy())

    return token_vectors^
