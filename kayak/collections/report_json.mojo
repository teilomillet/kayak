from std.collections import List

from .density import StorageDensity
from .report import CollectionStorageReport
from .segment_report import SegmentStorageReport


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_density_json(mut buffer: String, read density: StorageDensity):
    buffer += "{"
    buffer += "\"byte_size\":" + String(density.byte_size) + ","
    buffer += "\"document_count\":" + String(density.document_count) + ","
    buffer += "\"token_count\":" + String(density.token_count) + ","
    buffer += "\"vector_count\":" + String(density.vector_count) + ","
    buffer += "\"bytes_per_document\":" + String(density.bytes_per_document) + ","
    buffer += "\"bytes_per_token\":" + String(density.bytes_per_token) + ","
    buffer += "\"bytes_per_vector\":" + String(density.bytes_per_vector)
    buffer += "}"


def append_segment_storage_report_json(
    mut buffer: String, read report: SegmentStorageReport
):
    buffer += "{"
    buffer += "\"segment_id\":\"" + json_escape(report.segment_id) + "\","
    buffer += "\"has_text_corpus\":"
    if report.has_text_corpus:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"stats\":{"
    buffer += "\"document_count\":" + String(report.stats.document_count) + ","
    buffer += "\"token_count\":" + String(report.stats.token_count) + ","
    buffer += "\"total_vector_count\":"
    buffer += String(report.stats.total_vector_count) + ","
    buffer += "\"byte_size\":" + String(report.stats.byte_size)
    buffer += "},"
    buffer += "\"density\":"
    append_density_json(buffer, report.density)
    buffer += "}"


def append_segment_storage_reports_json(
    mut buffer: String, read reports: List[SegmentStorageReport]
):
    buffer += "["
    for index in range(len(reports)):
        if index > 0:
            buffer += ","
        append_segment_storage_report_json(buffer, reports[index])
    buffer += "]"


def collection_storage_report_json(
    read report: CollectionStorageReport
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"stats\":{"
    buffer += "\"segment_count\":" + String(report.stats.segment_count) + ","
    buffer += "\"document_count\":" + String(report.stats.document_count) + ","
    buffer += "\"token_count\":" + String(report.stats.token_count) + ","
    buffer += "\"total_vector_count\":" + String(report.stats.total_vector_count) + ","
    buffer += "\"byte_size\":" + String(report.stats.byte_size) + ","
    buffer += "\"average_vectors_per_document\":"
    buffer += String(report.stats.average_vectors_per_document)
    buffer += "},"
    buffer += "\"density\":"
    append_density_json(buffer, report.density)
    buffer += ","
    buffer += "\"segment_count_with_text\":"
    buffer += String(report.segment_count_with_text) + ","
    buffer += "\"segment_count_without_text\":"
    buffer += String(report.segment_count_without_text) + ","
    buffer += "\"segments\":"
    append_segment_storage_reports_json(buffer, report.segment_reports)
    buffer += "}"
    return buffer^
