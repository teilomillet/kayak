# Synthetic contradiction-heavy hard-recall fixtures.
#
# This module owns deterministic workloads where each query specifies several
# topical slot values plus an explicit polarity concept. For every topical
# combination, the corpus contains both matching-polarity documents and
# opposite-polarity documents that share every topical slot. It does not own
# stage-aware evaluation; it only constructs a reusable judged task and packed
# index for contradiction-sensitive retrieval experiments.

from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.index import pack_documents
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .proxy_vectors import concept_vectors


struct ContradictionHardRecallProfile(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var slot_count: Int
    var values_per_slot: Int
    var filler_vector_count: Int
    var filler_concept_pool_size: Int
    var duplicates_per_polarity: Int
    var query_count: Int
    var final_k: Int
    var centroid_head_posting_cap: Int

    def __init__(
        out self,
        var family: String,
        var slice_name: String,
        var why: String,
        slot_count: Int,
        values_per_slot: Int,
        filler_vector_count: Int,
        filler_concept_pool_size: Int,
        duplicates_per_polarity: Int,
        query_count: Int,
        final_k: Int,
        centroid_head_posting_cap: Int,
    ) raises:
        if slot_count <= 0:
            raise Error(
                "contradiction hard-recall profile slot_count must be positive"
            )

        if values_per_slot <= 1:
            raise Error(
                "contradiction hard-recall profile values_per_slot must exceed one"
            )

        if filler_vector_count < 0:
            raise Error(
                "contradiction hard-recall profile filler_vector_count must be non-negative"
            )

        if (
            filler_vector_count > 0
            and filler_concept_pool_size < filler_vector_count
        ):
            raise Error(
                "contradiction hard-recall profile filler_concept_pool_size must cover all filler vectors"
            )

        if duplicates_per_polarity <= 0:
            raise Error(
                "contradiction hard-recall profile duplicates_per_polarity must be positive"
            )

        if query_count <= 0:
            raise Error(
                "contradiction hard-recall profile query_count must be positive"
            )

        if final_k <= 0:
            raise Error(
                "contradiction hard-recall profile final_k must be positive"
            )

        if final_k > duplicates_per_polarity:
            raise Error(
                "contradiction hard-recall profile final_k must fit within the relevant duplicate count"
            )

        if centroid_head_posting_cap < 0:
            raise Error(
                "contradiction hard-recall profile centroid_head_posting_cap must be non-negative"
            )

        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.slot_count = slot_count
        self.values_per_slot = values_per_slot
        self.filler_vector_count = filler_vector_count
        self.filler_concept_pool_size = filler_concept_pool_size
        self.duplicates_per_polarity = duplicates_per_polarity
        self.query_count = query_count
        self.final_k = final_k
        self.centroid_head_posting_cap = centroid_head_posting_cap


struct ContradictionHardRecallFixture(Copyable):
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
        raise Error("contradiction hard-recall exponent must be non-negative")

    var value = 1
    for _ in range(exponent):
        value *= base

    return value


def combination_count(read profile: ContradictionHardRecallProfile) raises -> Int:
    return int_pow(profile.values_per_slot, profile.slot_count)


def document_count(read profile: ContradictionHardRecallProfile) raises -> Int:
    return (
        combination_count(profile)
        * 2
        * profile.duplicates_per_polarity
        + profile.query_count * profile.slot_count
    )


def nominal_document_vector_count(
    read profile: ContradictionHardRecallProfile
) -> Int:
    return profile.slot_count + 1 + profile.filler_vector_count


def vector_dim(read profile: ContradictionHardRecallProfile) -> Int:
    return (
        profile.slot_count * profile.values_per_slot
        + 2
        + profile.slot_count
        + profile.filler_concept_pool_size
    )


def attribute_concept_id(
    read profile: ContradictionHardRecallProfile,
    slot_index: Int,
    value_index: Int,
) -> Int:
    return slot_index * profile.values_per_slot + value_index


def support_concept_id(read profile: ContradictionHardRecallProfile) -> Int:
    return profile.slot_count * profile.values_per_slot


def refute_concept_id(read profile: ContradictionHardRecallProfile) -> Int:
    return support_concept_id(profile) + 1


def polarity_concept_id(
    read profile: ContradictionHardRecallProfile,
    prefers_support: Bool,
) -> Int:
    if prefers_support:
        return support_concept_id(profile)

    return refute_concept_id(profile)


def adversarial_mismatch_concept_id(
    read profile: ContradictionHardRecallProfile,
    omitted_slot_index: Int,
) -> Int:
    return refute_concept_id(profile) + 1 + omitted_slot_index


def filler_concept_base(read profile: ContradictionHardRecallProfile) -> Int:
    return refute_concept_id(profile) + 1 + profile.slot_count


def combination_values(
    read profile: ContradictionHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var total_combinations = combination_count(profile)
    if combination_index < 0 or combination_index >= total_combinations:
        raise Error(
            "contradiction hard-recall combination index must fit within the Cartesian product"
        )

    var values = List[Int]()
    var residual = combination_index

    for _ in range(profile.slot_count):
        values.append(residual % profile.values_per_slot)
        residual = residual // profile.values_per_slot

    return values^


def query_concepts_for_combination(
    read profile: ContradictionHardRecallProfile,
    combination_index: Int,
    prefers_support: Bool,
) raises -> List[Int]:
    var values = combination_values(profile, combination_index)
    var concepts = List[Int]()

    for slot_index in range(profile.slot_count):
        concepts.append(
            attribute_concept_id(profile, slot_index, values[slot_index])
        )

    concepts.append(polarity_concept_id(profile, prefers_support))
    return concepts^


def filler_concepts_for_combination(
    read profile: ContradictionHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var concepts = List[Int]()
    if profile.filler_vector_count == 0:
        return concepts^

    var base = filler_concept_base(profile)
    var start = combination_index % profile.filler_concept_pool_size
    for filler_index in range(profile.filler_vector_count):
        concepts.append(
            base + ((start + filler_index) % profile.filler_concept_pool_size)
        )

    return concepts^


def document_concepts_for_combination(
    read profile: ContradictionHardRecallProfile,
    combination_index: Int,
    prefers_support: Bool,
) raises -> List[Int]:
    var concepts = query_concepts_for_combination(
        profile,
        combination_index,
        prefers_support,
    )
    for filler_concept in filler_concepts_for_combination(profile, combination_index):
        concepts.append(filler_concept)

    return concepts^


def adversarial_near_miss_concepts(
    read profile: ContradictionHardRecallProfile,
    query_combination_index: Int,
    prefers_support: Bool,
    omitted_slot_index: Int,
) raises -> List[Int]:
    var values = combination_values(profile, query_combination_index)
    var matched_concepts = List[Int]()
    var concepts = List[Int]()

    for slot_index in range(profile.slot_count):
        if slot_index == omitted_slot_index:
            concepts.append(
                adversarial_mismatch_concept_id(profile, omitted_slot_index)
            )
            continue

        var concept = attribute_concept_id(profile, slot_index, values[slot_index])
        concepts.append(concept)
        matched_concepts.append(concept)

    var polarity_concept = polarity_concept_id(profile, prefers_support)
    concepts.append(polarity_concept)
    matched_concepts.append(polarity_concept)

    for filler_index in range(profile.filler_vector_count):
        concepts.append(matched_concepts[filler_index % len(matched_concepts)])

    return concepts^


def selected_query_combination_index(
    read profile: ContradictionHardRecallProfile,
    query_index: Int,
) raises -> Int:
    var total_combinations = combination_count(profile)
    if query_index < 0 or query_index >= profile.query_count:
        raise Error(
            "contradiction hard-recall query index must fit within query_count"
        )

    if profile.query_count > total_combinations:
        raise Error(
            "contradiction hard-recall query_count must not exceed the combination count"
        )

    return (query_index * total_combinations) // profile.query_count


def query_prefers_support(query_index: Int) -> Bool:
    return query_index % 2 == 0


def polarity_label(prefers_support: Bool) -> String:
    if prefers_support:
        return "support"

    return "refute"


def document_id_for_combination(
    combination_index: Int,
    prefers_support: Bool,
    duplicate_index: Int,
) -> String:
    return (
        "doc-c"
        + String(combination_index)
        + "-"
        + polarity_label(prefers_support)
        + "-d"
        + String(duplicate_index)
    )


def relevant_doc_ids_for_query(
    read profile: ContradictionHardRecallProfile,
    combination_index: Int,
    prefers_support: Bool,
) -> List[String]:
    var doc_ids = List[String]()

    for duplicate_index in range(profile.final_k):
        doc_ids.append(
            document_id_for_combination(
                combination_index,
                prefers_support,
                duplicate_index,
            )
        )

    return doc_ids^


def adversarial_doc_id(
    query_index: Int,
    omitted_slot_index: Int,
    prefers_support: Bool,
) -> String:
    return (
        "doc-adv-q"
        + String(query_index)
        + "-s"
        + String(omitted_slot_index)
        + "-"
        + polarity_label(prefers_support)
    )


def default_contradiction_hard_recall_profiles(
) raises -> List[ContradictionHardRecallProfile]:
    var profiles = List[ContradictionHardRecallProfile]()
    profiles.append(
        ContradictionHardRecallProfile(
            "contradiction_hard_recall",
            "slots6_values3_docs2988",
            "High-overlap contradiction workload where correct and opposite-polarity documents share all six topical slot vectors and differ only on one polarity vector.",
            6,
            3,
            24,
            48,
            2,
            12,
            2,
            16,
        )
    )
    return profiles^


def make_contradiction_hard_recall_fixture(
    read profile: ContradictionHardRecallProfile
) raises -> ContradictionHardRecallFixture:
    var total_combinations = combination_count(profile)
    if profile.query_count > total_combinations:
        raise Error(
            "contradiction hard-recall profile query_count must fit within the combination count"
        )

    var documents = List[EncodedDocument]()
    var queries = List[JudgedQuery]()

    for combination_index in range(total_combinations):
        for polarity_index in range(2):
            var prefers_support = polarity_index == 0
            var concepts = document_concepts_for_combination(
                profile,
                combination_index,
                prefers_support,
            )
            for duplicate_index in range(profile.duplicates_per_polarity):
                documents.append(
                    EncodedDocument(
                        document_id_for_combination(
                            combination_index,
                            prefers_support,
                            duplicate_index,
                        ),
                        concept_vectors(vector_dim(profile), concepts),
                    )
                )

    for query_index in range(profile.query_count):
        var combination_index = selected_query_combination_index(profile, query_index)
        var prefers_support = query_prefers_support(query_index)

        for omitted_slot_index in range(profile.slot_count):
            documents.append(
                EncodedDocument(
                    adversarial_doc_id(
                        query_index,
                        omitted_slot_index,
                        prefers_support,
                    ),
                    concept_vectors(
                        vector_dim(profile),
                        adversarial_near_miss_concepts(
                            profile,
                            combination_index,
                            prefers_support,
                            omitted_slot_index,
                        ),
                    ),
                )
            )

        queries.append(
            JudgedQuery(
                "query-" + String(query_index),
                "Synthetic polarity-sensitive query over shared slot values with opposite-polarity distractors.",
                EncodedQuery(
                    concept_vectors(
                        vector_dim(profile),
                        query_concepts_for_combination(
                            profile,
                            combination_index,
                            prefers_support,
                        ),
                    )
                ),
                relevant_doc_ids_for_query(
                    profile,
                    combination_index,
                    prefers_support,
                ),
            )
        )

    var dataset_id = "synthetic://contradiction-hard-recall/" + profile.slice_name
    var model_name = "synthetic-slot-polarity"
    var packed_index = pack_documents(documents)

    return ContradictionHardRecallFixture(
        StoredJudgedTask(
            dataset_id.copy(),
            model_name.copy(),
            VECTOR_SCALAR_NAME,
            JudgedTask(
                profile.family.copy(),
                profile.slice_name.copy(),
                profile.why.copy(),
                "ndcg",
                profile.final_k,
                profile.slot_count + 1,
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
