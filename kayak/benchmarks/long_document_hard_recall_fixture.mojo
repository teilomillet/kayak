# Synthetic long-document hard-recall fixtures.
#
# This module owns deterministic workloads where the exact matching concepts
# appear late in long document representations behind a noisy prefix. It does
# not own stage-aware evaluation or collection mirroring. The goal is to create
# a long-document retrieval family that stresses budgeted stage-1 sidecars
# without hiding the vector-count choices.

from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.index import pack_documents
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .proxy_vectors import concept_vectors


struct LongDocumentHardRecallProfile(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var slot_count: Int
    var values_per_slot: Int
    var prefix_vector_count: Int
    var prefix_concept_pool_size: Int
    var duplicates_per_combination: Int
    var query_count: Int
    var final_k: Int
    var document_proxy_vector_budget: Int
    var centroid_budget: Int
    var centroid_head_posting_cap: Int

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        slot_count: Int,
        values_per_slot: Int,
        prefix_vector_count: Int,
        prefix_concept_pool_size: Int,
        duplicates_per_combination: Int,
        query_count: Int,
        final_k: Int,
        document_proxy_vector_budget: Int,
        centroid_budget: Int,
        centroid_head_posting_cap: Int,
    ) raises:
        if slot_count <= 0:
            raise Error(
                "long-document hard-recall profile slot_count must be positive"
            )

        if values_per_slot <= 1:
            raise Error(
                "long-document hard-recall profile values_per_slot must exceed one"
            )

        if prefix_vector_count <= 0:
            raise Error(
                "long-document hard-recall profile prefix_vector_count must be positive"
            )

        if prefix_concept_pool_size <= 0:
            raise Error(
                "long-document hard-recall profile prefix_concept_pool_size must be positive"
            )

        if duplicates_per_combination <= 0:
            raise Error(
                "long-document hard-recall profile duplicates_per_combination must be positive"
            )

        if query_count <= 0:
            raise Error(
                "long-document hard-recall profile query_count must be positive"
            )

        if final_k <= 0:
            raise Error(
                "long-document hard-recall profile final_k must be positive"
            )

        if final_k > duplicates_per_combination:
            raise Error(
                "long-document hard-recall profile final_k must fit within the relevant duplicate count"
            )

        if document_proxy_vector_budget < 0:
            raise Error(
                "long-document hard-recall profile document_proxy_vector_budget must be non-negative"
            )

        if centroid_budget < 0:
            raise Error(
                "long-document hard-recall profile centroid_budget must be non-negative"
            )

        if centroid_head_posting_cap < 0:
            raise Error(
                "long-document hard-recall profile centroid_head_posting_cap must be non-negative"
            )

        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.slot_count = slot_count
        self.values_per_slot = values_per_slot
        self.prefix_vector_count = prefix_vector_count
        self.prefix_concept_pool_size = prefix_concept_pool_size
        self.duplicates_per_combination = duplicates_per_combination
        self.query_count = query_count
        self.final_k = final_k
        self.document_proxy_vector_budget = document_proxy_vector_budget
        self.centroid_budget = centroid_budget
        self.centroid_head_posting_cap = centroid_head_posting_cap


struct LongDocumentHardRecallFixture(Copyable):
    var stored_task: StoredJudgedTask
    var stored_index: StoredPackedIndex

    def __init__(
        out self,
        stored_task: StoredJudgedTask,
        stored_index: StoredPackedIndex,
    ):
        self.stored_task = stored_task.copy()
        self.stored_index = stored_index.copy()


def int_pow(base: Int, exponent: Int) raises -> Int:
    if exponent < 0:
        raise Error("long-document hard-recall exponent must be non-negative")

    var value = 1
    for _ in range(exponent):
        value *= base

    return value


def combination_count(read profile: LongDocumentHardRecallProfile) raises -> Int:
    return int_pow(profile.values_per_slot, profile.slot_count)


def document_count(read profile: LongDocumentHardRecallProfile) raises -> Int:
    return (
        combination_count(profile)
        * profile.duplicates_per_combination
        + profile.query_count
        * profile.slot_count
    )


def nominal_document_vector_count(
    read profile: LongDocumentHardRecallProfile
) -> Int:
    return profile.prefix_vector_count + profile.slot_count


def vector_dim(read profile: LongDocumentHardRecallProfile) -> Int:
    return (
        profile.slot_count * profile.values_per_slot
        + profile.prefix_concept_pool_size
        + profile.slot_count
    )


def attribute_concept_id(
    read profile: LongDocumentHardRecallProfile,
    slot_index: Int,
    value_index: Int,
) -> Int:
    return slot_index * profile.values_per_slot + value_index


def prefix_concept_base(read profile: LongDocumentHardRecallProfile) -> Int:
    return profile.slot_count * profile.values_per_slot


def mismatch_concept_base(read profile: LongDocumentHardRecallProfile) -> Int:
    return prefix_concept_base(profile) + profile.prefix_concept_pool_size


def mismatch_concept_id(
    read profile: LongDocumentHardRecallProfile, slot_index: Int
) -> Int:
    return mismatch_concept_base(profile) + slot_index


def combination_values(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var total_combinations = combination_count(profile)
    if combination_index < 0 or combination_index >= total_combinations:
        raise Error(
            "long-document hard-recall combination index must fit within the Cartesian product"
        )

    var values = List[Int]()
    var residual = combination_index

    for _ in range(profile.slot_count):
        values.append(residual % profile.values_per_slot)
        residual = residual // profile.values_per_slot

    return values^


def query_concepts_for_combination(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var values = combination_values(profile, combination_index)
    var concepts = List[Int]()

    for slot_index in range(profile.slot_count):
        concepts.append(
            attribute_concept_id(profile, slot_index, values[slot_index])
        )

    return concepts^


def prefix_concepts_for_combination(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
) -> List[Int]:
    var concepts = List[Int]()
    var base = prefix_concept_base(profile)
    var offset = combination_index % profile.prefix_concept_pool_size

    for prefix_index in range(profile.prefix_vector_count):
        concepts.append(
            base
                + (
                    (offset + prefix_index)
                    % profile.prefix_concept_pool_size
                )
        )

    return concepts^


def relevant_document_concepts_for_combination(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var concepts = prefix_concepts_for_combination(profile, combination_index)
    for query_concept in query_concepts_for_combination(profile, combination_index):
        concepts.append(query_concept)

    return concepts^


def near_miss_document_concepts(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
    omitted_slot_index: Int,
) raises -> List[Int]:
    var concepts = prefix_concepts_for_combination(profile, combination_index)
    var values = combination_values(profile, combination_index)

    for slot_index in range(profile.slot_count):
        if slot_index == omitted_slot_index:
            concepts.append(mismatch_concept_id(profile, slot_index))
            continue

        concepts.append(
            attribute_concept_id(profile, slot_index, values[slot_index])
        )

    return concepts^


def document_id_for_combination(
    combination_index: Int, duplicate_index: Int
) -> String:
    return "doc-long-c" + String(combination_index) + "-d" + String(duplicate_index)


def relevant_doc_ids_for_combination(
    read profile: LongDocumentHardRecallProfile,
    combination_index: Int,
) -> List[String]:
    var doc_ids = List[String]()

    for duplicate_index in range(profile.final_k):
        doc_ids.append(
            document_id_for_combination(combination_index, duplicate_index)
        )

    return doc_ids^


def adversarial_doc_id(query_index: Int, omitted_slot_index: Int) -> String:
    return (
        "doc-long-adv-q"
        + String(query_index)
        + "-s"
        + String(omitted_slot_index)
    )


def selected_query_combination_index(
    read profile: LongDocumentHardRecallProfile,
    query_index: Int,
) raises -> Int:
    var total_combinations = combination_count(profile)
    if query_index < 0 or query_index >= profile.query_count:
        raise Error(
            "long-document hard-recall query index must fit within query_count"
        )

    if profile.query_count >= total_combinations:
        raise Error(
            "long-document hard-recall query_count must stay below the combination count"
        )

    return ((query_index + 1) * total_combinations) // (profile.query_count + 1)


def default_long_document_hard_recall_profiles(
) raises -> List[LongDocumentHardRecallProfile]:
    var profiles = List[LongDocumentHardRecallProfile]()
    profiles.append(
        LongDocumentHardRecallProfile(
            "long_document_hard_recall",
            "late_suffix_docs544_vec100",
            "Long noisy-prefix workload where the exact matching vectors appear only at the end of each 100-vector document.",
            4,
            4,
            96,
            24,
            2,
            8,
            2,
            16,
            16,
            16,
        )
    )
    profiles.append(
        LongDocumentHardRecallProfile(
            "long_document_hard_recall",
            "late_suffix_docs2088_vec133",
            "Long noisy-prefix workload with larger corpus size and 133-vector documents whose relevant token concepts live after a dominant common prefix.",
            5,
            4,
            128,
            32,
            2,
            8,
            2,
            16,
            16,
            16,
        )
    )
    return profiles^


def make_long_document_hard_recall_fixture(
    read profile: LongDocumentHardRecallProfile
) raises -> LongDocumentHardRecallFixture:
    var total_combinations = combination_count(profile)
    if profile.query_count >= total_combinations:
        raise Error(
            "long-document hard-recall profile query_count must stay below the combination count"
        )

    var documents = List[EncodedDocument]()
    var queries = List[JudgedQuery]()

    for combination_index in range(total_combinations):
        var concepts = relevant_document_concepts_for_combination(
            profile, combination_index
        )
        for duplicate_index in range(profile.duplicates_per_combination):
            documents.append(
                EncodedDocument(
                    document_id_for_combination(combination_index, duplicate_index),
                    concept_vectors(vector_dim(profile), concepts),
                )
            )

    for query_index in range(profile.query_count):
        var combination_index = selected_query_combination_index(profile, query_index)
        for omitted_slot_index in range(profile.slot_count):
            documents.append(
                EncodedDocument(
                    adversarial_doc_id(query_index, omitted_slot_index),
                    concept_vectors(
                        vector_dim(profile),
                        near_miss_document_concepts(
                            profile,
                            combination_index,
                            omitted_slot_index,
                        ),
                    ),
                )
            )

        queries.append(
            JudgedQuery(
                "query-long-" + String(query_index),
                "Long-document retrieval query whose exact supporting concepts appear only after a noisy shared prefix.",
                EncodedQuery(
                    concept_vectors(
                        vector_dim(profile),
                        query_concepts_for_combination(profile, combination_index),
                    )
                ),
                relevant_doc_ids_for_combination(profile, combination_index),
            )
        )

    var dataset_id = "synthetic://long-document-hard-recall/" + profile.slice_name
    var model_name = "synthetic-long-document-late-needles"
    var packed_index = pack_documents(documents)

    return LongDocumentHardRecallFixture(
        StoredJudgedTask(
            dataset_id.copy(),
            model_name.copy(),
            VECTOR_SCALAR_NAME,
            JudgedTask(
                profile.family.copy(),
                profile.slice_name.copy(),
                profile.why.copy(),
                "recall",
                profile.final_k,
                profile.slot_count,
                nominal_document_vector_count(profile),
                vector_dim(profile),
                documents^,
                queries^,
            ),
        ),
        StoredPackedIndex(
            dataset_id,
            model_name,
            VECTOR_SCALAR_NAME,
            packed_index^,
        ),
    )
