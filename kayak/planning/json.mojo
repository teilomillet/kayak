from std.collections import List

from .candidate_set import CandidateSet
from .collection_hit import CollectionHit
from .explain import CollectionSearchExplain
from .stage_profile import SearchStageProfile


def json_escape(text: String) -> String:
    var escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace("\"", "\\\"")
    escaped = escaped.replace("\n", "\\n")
    escaped = escaped.replace("\r", "\\r")
    escaped = escaped.replace("\t", "\\t")
    return escaped


def append_json_hit(mut buffer: String, read hit: CollectionHit):
    buffer += "{"
    buffer += "\"segment_id\":\"" + json_escape(hit.segment_id) + "\","
    buffer += "\"doc_id\":\"" + json_escape(hit.doc_id) + "\","
    buffer += "\"score\":" + String(hit.score)
    buffer += "}"


def append_json_hit_list(mut buffer: String, read hits: List[CollectionHit]):
    buffer += "["
    for index in range(len(hits)):
        if index > 0:
            buffer += ","
        append_json_hit(buffer, hits[index])
    buffer += "]"


def append_json_stage_profile(
    mut buffer: String, read profile: SearchStageProfile
):
    buffer += "{"
    buffer += "\"stage_name\":\"" + json_escape(profile.stage_name) + "\","
    buffer += "\"input_hit_count\":" + String(profile.input_hit_count) + ","
    buffer += "\"output_hit_count\":" + String(profile.output_hit_count) + ","
    buffer += "\"segment_count\":" + String(profile.segment_count) + ","
    buffer += "\"document_count\":" + String(profile.document_count) + ","
    buffer += "\"token_count\":" + String(profile.token_count) + ","
    buffer += "\"vector_count\":" + String(profile.vector_count) + ","
    buffer += "\"byte_size\":" + String(profile.byte_size)
    buffer += "}"


def append_json_candidate_set(mut buffer: String, read candidate_set: CandidateSet):
    buffer += "{"
    buffer += "\"generator_kind\":\"" + json_escape(candidate_set.generator_kind) + "\","
    buffer += "\"segment_count\":" + String(candidate_set.segment_count) + ","
    buffer += "\"document_count\":" + String(candidate_set.document_count) + ","
    buffer += "\"token_count\":" + String(candidate_set.token_count) + ","
    buffer += "\"vector_count\":" + String(candidate_set.vector_count) + ","
    buffer += "\"byte_size\":" + String(candidate_set.byte_size) + ","
    buffer += "\"hit_count\":" + String(len(candidate_set.hits)) + ","
    buffer += "\"hits\":"
    append_json_hit_list(buffer, candidate_set.hits)
    buffer += "}"


def collection_search_explain_json(
    read explain: CollectionSearchExplain
) -> String:
    var buffer = String()
    buffer += "{"
    buffer += "\"collection_id\":\"" + json_escape(explain.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(explain.snapshot_id) + "\","
    buffer += "\"plan\":{"
    buffer += "\"candidate_generator_kind\":\""
    buffer += json_escape(explain.plan.candidate_generator.kind)
    buffer += "\","
    buffer += "\"candidate_k\":" + String(explain.plan.candidate_budget.candidate_k) + ","
    buffer += "\"final_k\":" + String(explain.plan.candidate_budget.final_k) + ","
    buffer += "\"exact_stage_kind\":\"" + json_escape(explain.plan.exact_stage_kind) + "\","
    buffer += "\"reranker_kind\":\"" + json_escape(explain.plan.reranker_kind) + "\""
    buffer += "},"
    buffer += "\"candidate_set\":"
    append_json_candidate_set(buffer, explain.candidate_set)
    buffer += ","
    buffer += "\"candidate_stage\":"
    append_json_stage_profile(buffer, explain.candidate_stage)
    buffer += ","
    buffer += "\"exact_stage\":"
    append_json_stage_profile(buffer, explain.exact_stage)
    buffer += ","
    buffer += "\"candidate_recall_at_final_k\":"
    buffer += String(explain.candidate_recall_at_final_k)
    buffer += ","
    buffer += "\"final_hits\":"
    append_json_hit_list(buffer, explain.final_hits)
    buffer += "}"
    return buffer^
