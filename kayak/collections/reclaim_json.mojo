from std.collections import List

from .ids import SegmentId
from .reclaim import (
    CollectionReclaimExecutionResult,
    CollectionReclaimPlan,
    SnapshotRetentionDecision,
)
from .ids import SnapshotId


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_snapshot_retention_decision_json(
    mut buffer: String, read decision: SnapshotRetentionDecision
):
    buffer += "{"
    buffer += "\"snapshot_id\":\"" + json_escape(decision.snapshot_id.value) + "\","
    buffer += "\"generation\":" + String(decision.generation) + ","
    buffer += "\"segment_count\":" + String(decision.segment_count) + ","
    buffer += "\"byte_size\":" + String(decision.byte_size) + ","
    buffer += "\"retain\":"
    if decision.retain:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"reason\":\"" + json_escape(decision.reason) + "\""
    buffer += "}"


def append_snapshot_retention_decisions_json(
    mut buffer: String, read decisions: List[SnapshotRetentionDecision]
):
    buffer += "["
    for index in range(len(decisions)):
        if index > 0:
            buffer += ","
        append_snapshot_retention_decision_json(buffer, decisions[index])
    buffer += "]"


def append_segment_ids_json(mut buffer: String, read segment_ids: List[SegmentId]):
    buffer += "["
    for index in range(len(segment_ids)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(segment_ids[index].value)
        buffer += "\""
    buffer += "]"


def append_snapshot_ids_json(mut buffer: String, read snapshot_ids: List[SnapshotId]):
    buffer += "["
    for index in range(len(snapshot_ids)):
        if index > 0:
            buffer += ","
        buffer += "\""
        buffer += json_escape(snapshot_ids[index].value)
        buffer += "\""
    buffer += "]"


def collection_reclaim_plan_json(read plan: CollectionReclaimPlan) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(plan.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(plan.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(plan.namespace_id.value) + "\","
    buffer += "\"active_snapshot_id\":\""
    buffer += json_escape(plan.active_snapshot_id) + "\","
    buffer += "\"total_snapshot_count\":"
    buffer += String(plan.total_snapshot_count) + ","
    buffer += "\"inactive_snapshot_count\":"
    buffer += String(plan.inactive_snapshot_count) + ","
    buffer += "\"retained_inactive_snapshot_count\":"
    buffer += String(plan.retained_inactive_snapshot_count) + ","
    buffer += "\"reclaimable_snapshot_count\":"
    buffer += String(plan.reclaimable_snapshot_count) + ","
    buffer += "\"reclaimable_unique_segment_count\":"
    buffer += String(plan.reclaimable_unique_segment_count) + ","
    buffer += "\"reclaimable_unique_byte_size\":"
    buffer += String(plan.reclaimable_unique_byte_size) + ","
    buffer += "\"reclaimable_unique_segment_ids\":"
    append_segment_ids_json(buffer, plan.reclaimable_unique_segment_ids)
    buffer += ",\"decisions\":"
    append_snapshot_retention_decisions_json(buffer, plan.decisions)
    buffer += "}"
    return buffer^


def collection_reclaim_execution_result_json(
    read result: CollectionReclaimExecutionResult
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(result.collection_id.value) + "\","
    buffer += "\"tenant_id\":\"" + json_escape(result.tenant_id.value) + "\","
    buffer += "\"namespace_id\":\"" + json_escape(result.namespace_id.value) + "\","
    buffer += "\"applied\":"
    if result.applied:
        buffer += "true,"
    else:
        buffer += "false,"
    buffer += "\"snapshot_count\":" + String(result.snapshot_count) + ","
    buffer += "\"unique_segment_count\":" + String(result.unique_segment_count) + ","
    buffer += "\"unique_byte_size\":" + String(result.unique_byte_size) + ","
    buffer += "\"snapshot_ids\":"
    append_snapshot_ids_json(buffer, result.snapshot_ids)
    buffer += ",\"unique_segment_ids\":"
    append_segment_ids_json(buffer, result.unique_segment_ids)
    buffer += "}"
    return buffer^
