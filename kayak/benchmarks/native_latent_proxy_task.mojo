from std.pathlib import Path

from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror_with_latent_proxy,
    load_resolved_collection_snapshot,
)
from kayak.eval import JudgedTask
from kayak.index import pack_documents
from kayak.interop import load_document_text_corpus_json, load_task_json
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.planning import (
    CandidateBudget,
    CandidateGenerator,
    SearchPlan,
    best_effort_faithfulness_policy,
    exact_late_interaction_reference_scoring_semantics,
    exact_late_interaction_stage2_reference_operator,
    none_stage3_verifier_operator,
)
from kayak.runtime import ExactCpuBackend
from kayak.storage import (
    StoredJudgedTask,
    StoredPackedIndex,
    load_stored_latent_proxy_index,
)
from kayak.text import DocumentTextCorpus

from .faithfulness_frontier_json import (
    FaithfulnessFrontierSummary,
    build_faithfulness_frontier_summary_for_plan,
    faithfulness_frontier_summary_json,
)
from .json_common import json_escape


# Owns one explicit bridge from task JSON + exported latent proxy artifact into
# the native collection runtime. It does not own training or artifact export.


struct NativeLatentProxyCollectionSummary(Copyable):
    var dataset_id: String
    var model_name: String
    var collection_id: String
    var snapshot_id: String
    var document_count: Int
    var vector_count: Int
    var vector_dim: Int
    var latent_dim: Int
    var stage1_byte_size: Int
    var text_corpus_loaded: Bool

    def __init__(
        out self,
        var dataset_id: String,
        var model_name: String,
        var collection_id: String,
        var snapshot_id: String,
        document_count: Int,
        vector_count: Int,
        vector_dim: Int,
        latent_dim: Int,
        stage1_byte_size: Int,
        text_corpus_loaded: Bool,
    ):
        self.dataset_id = dataset_id^
        self.model_name = model_name^
        self.collection_id = collection_id^
        self.snapshot_id = snapshot_id^
        self.document_count = document_count
        self.vector_count = vector_count
        self.vector_dim = vector_dim
        self.latent_dim = latent_dim
        self.stage1_byte_size = stage1_byte_size
        self.text_corpus_loaded = text_corpus_loaded


def max_query_vector_budget(read task: JudgedTask) -> Int:
    var max_budget = 0
    for judged_query in task.queries:
        if judged_query.query.vector_count > max_budget:
            max_budget = judged_query.query.vector_count

    if max_budget <= 0:
        return 1
    return max_budget


def load_stored_task_from_task_json(
    dataset_id: String,
    model_name: String,
    task_path: String,
) raises -> StoredJudgedTask:
    return StoredJudgedTask(
        dataset_id.copy(),
        model_name.copy(),
        VECTOR_SCALAR_NAME,
        load_task_json(task_path),
    )


def build_stored_packed_index_for_task(
    read stored_task: StoredJudgedTask
) raises -> StoredPackedIndex:
    return StoredPackedIndex(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        stored_task.vector_scalar_name.copy(),
        pack_documents(stored_task.task.documents),
    )


def task_text_corpus(task_path: String, load_text_corpus: Bool) raises -> DocumentTextCorpus:
    if load_text_corpus:
        return load_document_text_corpus_json(task_path)
    return DocumentTextCorpus([], [])


def materialize_native_latent_proxy_task_collection(
    dataset_id: String,
    model_name: String,
    task_path: String,
    artifact_root: Path,
    collection_root: Path,
    collection_id: CollectionId,
    tenant_id: TenantId,
    namespace_id: NamespaceId,
    snapshot_id: SnapshotId,
    load_text_corpus: Bool = True,
) raises -> NativeLatentProxyCollectionSummary:
    var stored_task = load_stored_task_from_task_json(
        dataset_id,
        model_name,
        task_path,
    )
    var stored_index = build_stored_packed_index_for_task(stored_task)
    var stored_latent_proxy_index = load_stored_latent_proxy_index(artifact_root)
    var text_corpus = task_text_corpus(task_path, load_text_corpus)
    _ = ensure_one_segment_collection_mirror_with_latent_proxy(
        collection_root,
        collection_id,
        tenant_id,
        namespace_id,
        snapshot_id,
        1,
        stored_index,
        stored_latent_proxy_index,
        text_corpus,
    )
    return NativeLatentProxyCollectionSummary(
        stored_task.dataset_id.copy(),
        stored_task.model_name.copy(),
        collection_id.value.copy(),
        snapshot_id.value.copy(),
        stored_index.index.document_count,
        stored_index.index.total_vector_count,
        stored_index.index.vector_dim,
        stored_latent_proxy_index.index.vector_dim,
        stored_latent_proxy_index.artifact_byte_size,
        load_text_corpus,
    )


def build_materialized_collection_search_summary(
    dataset_id: String,
    model_name: String,
    task_path: String,
    collection_root: Path,
    snapshot_id: SnapshotId,
    candidate_generator_kind: String,
    candidate_k: Int,
) raises -> FaithfulnessFrontierSummary:
    var stored_task = load_stored_task_from_task_json(
        dataset_id,
        model_name,
        task_path,
    )
    var snapshot = load_resolved_collection_snapshot(collection_root, snapshot_id)
    var task = stored_task.task.copy()
    var plan = SearchPlan(
        CandidateGenerator(candidate_generator_kind),
        CandidateBudget(task.k, candidate_k),
        best_effort_faithfulness_policy(),
        exact_late_interaction_reference_scoring_semantics(),
        exact_late_interaction_stage2_reference_operator(),
        none_stage3_verifier_operator(),
    )
    return build_faithfulness_frontier_summary_for_plan(
        ExactCpuBackend(),
        stored_task,
        snapshot,
        plan,
        max_query_vector_budget(task),
        0,
        0,
    )


def append_native_latent_proxy_collection_summary_json(
    mut buffer: String, read summary: NativeLatentProxyCollectionSummary
):
    buffer += "{"
    buffer += "\"dataset_id\":\"" + json_escape(summary.dataset_id) + "\","
    buffer += "\"model_name\":\"" + json_escape(summary.model_name) + "\","
    buffer += "\"collection_id\":\"" + json_escape(summary.collection_id) + "\","
    buffer += "\"snapshot_id\":\"" + json_escape(summary.snapshot_id) + "\","
    buffer += "\"document_count\":" + String(summary.document_count) + ","
    buffer += "\"vector_count\":" + String(summary.vector_count) + ","
    buffer += "\"vector_dim\":" + String(summary.vector_dim) + ","
    buffer += "\"latent_dim\":" + String(summary.latent_dim) + ","
    buffer += "\"stage1_byte_size\":" + String(summary.stage1_byte_size) + ","
    buffer += "\"text_corpus_loaded\":"
    if summary.text_corpus_loaded:
        buffer += "true"
    else:
        buffer += "false"
    buffer += "}"


def native_latent_proxy_collection_summary_json(
    read summary: NativeLatentProxyCollectionSummary
) -> String:
    var buffer = String()
    append_native_latent_proxy_collection_summary_json(buffer, summary)
    return buffer^
