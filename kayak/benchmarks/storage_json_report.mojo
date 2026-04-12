from std.collections import List

from kayak.collections import CollectionStorageReport


struct RealSliceCollectionStorageSummary(Copyable):
    var dataset_id: String
    var collection_id: String
    var model_name: String
    var snapshot_id: String
    var segment_count: Int
    var document_count: Int
    var token_count: Int
    var vector_count: Int
    var byte_size: Int
    var bytes_per_document: Float64
    var bytes_per_token: Float64
    var bytes_per_vector: Float64
    var segment_count_with_text: Int
    var segment_count_without_text: Int

    def __init__(
        out self,
        var dataset_id: String,
        var collection_id: String,
        var model_name: String,
        var snapshot_id: String,
        segment_count: Int,
        document_count: Int,
        token_count: Int,
        vector_count: Int,
        byte_size: Int,
        bytes_per_document: Float64,
        bytes_per_token: Float64,
        bytes_per_vector: Float64,
        segment_count_with_text: Int,
        segment_count_without_text: Int,
    ):
        self.dataset_id = dataset_id^
        self.collection_id = collection_id^
        self.model_name = model_name^
        self.snapshot_id = snapshot_id^
        self.segment_count = segment_count
        self.document_count = document_count
        self.token_count = token_count
        self.vector_count = vector_count
        self.byte_size = byte_size
        self.bytes_per_document = bytes_per_document
        self.bytes_per_token = bytes_per_token
        self.bytes_per_vector = bytes_per_vector
        self.segment_count_with_text = segment_count_with_text
        self.segment_count_without_text = segment_count_without_text


def build_real_slice_collection_storage_summary(
    dataset_id: String,
    collection_id: String,
    model_name: String,
    snapshot_id: String,
    read report: CollectionStorageReport,
) -> RealSliceCollectionStorageSummary:
    return RealSliceCollectionStorageSummary(
        dataset_id,
        collection_id,
        model_name,
        snapshot_id,
        report.stats.segment_count,
        report.stats.document_count,
        report.stats.token_count,
        report.stats.total_vector_count,
        report.stats.byte_size,
        Float64(report.density.bytes_per_document),
        Float64(report.density.bytes_per_token),
        Float64(report.density.bytes_per_vector),
        report.segment_count_with_text,
        report.segment_count_without_text,
    )


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_real_slice_collection_storage_summary_json(
    mut buffer: String, read summary: RealSliceCollectionStorageSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"segment_count\":" + String(summary.segment_count) + ","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"token_count\":" + String(summary.token_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"byte_size\":" + String(summary.byte_size) + ","
    buffer += "\"bytes_per_document\":"
    buffer += String(summary.bytes_per_document) + ","
    buffer += "\"bytes_per_token\":" + String(summary.bytes_per_token) + ","
    buffer += "\"bytes_per_vector\":" + String(summary.bytes_per_vector) + ","
    buffer += "\"segment_count_with_text\":"
    buffer += String(summary.segment_count_with_text) + ","
    buffer += "\"segment_count_without_text\":"
    buffer += String(summary.segment_count_without_text)
    buffer += "}"


def real_slice_collection_storage_summaries_json(
    read summaries: List[RealSliceCollectionStorageSummary]
) -> String:
    var buffer = String()
    buffer += "["

    for index in range(len(summaries)):
        if index > 0:
            buffer += ","

        append_real_slice_collection_storage_summary_json(
            buffer, summaries[index]
        )
    buffer += "]"
    return buffer^
