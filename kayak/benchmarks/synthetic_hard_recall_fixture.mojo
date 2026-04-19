# Synthetic conjunction-style hard-recall fixtures.
#
# This module owns deterministic synthetic workloads where each query specifies
# one value per shared attribute slot and documents vary over the Cartesian
# product of those slot values. It does not own stage-aware evaluation or
# collection mirroring. The goal is to create a harder stage-1 recall family
# than the grouped single-core scale fixture without hiding the vector-count
# choices.

from std.collections import List

from kayak.contracts import EncodedDocument, EncodedQuery
from kayak.eval import JudgedQuery, JudgedTask
from kayak.index import pack_documents
from kayak.numeric import VECTOR_SCALAR_NAME
from kayak.storage import StoredJudgedTask, StoredPackedIndex

from .proxy_vectors import concept_vectors


struct SyntheticHardRecallProfile(Copyable):
    var family: String
    var slice_name: String
    var why: String
    var slot_count: Int
    var values_per_slot: Int
    var filler_vector_count: Int
    var filler_concept_pool_size: Int
    var duplicates_per_combination: Int
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
        duplicates_per_combination: Int,
        query_count: Int,
        final_k: Int,
        centroid_head_posting_cap: Int,
    ) raises:
        if slot_count <= 0:
            raise Error("synthetic hard-recall profile slot_count must be positive")

        if values_per_slot <= 1:
            raise Error(
                "synthetic hard-recall profile values_per_slot must exceed one"
            )

        if filler_vector_count < 0:
            raise Error(
                "synthetic hard-recall profile filler_vector_count must be non-negative"
            )

        if filler_vector_count > 0 and filler_concept_pool_size < filler_vector_count:
            raise Error(
                "synthetic hard-recall profile filler_concept_pool_size must cover all filler vectors"
            )

        if duplicates_per_combination <= 0:
            raise Error(
                "synthetic hard-recall profile duplicates_per_combination must be positive"
            )

        if query_count <= 0:
            raise Error("synthetic hard-recall profile query_count must be positive")

        if final_k <= 0:
            raise Error("synthetic hard-recall profile final_k must be positive")

        if final_k > duplicates_per_combination:
            raise Error(
                "synthetic hard-recall profile final_k must fit within the relevant duplicate count"
            )

        if centroid_head_posting_cap < 0:
            raise Error(
                "synthetic hard-recall profile centroid_head_posting_cap must be non-negative"
            )

        self.family = family^
        self.slice_name = slice_name^
        self.why = why^
        self.slot_count = slot_count
        self.values_per_slot = values_per_slot
        self.filler_vector_count = filler_vector_count
        self.filler_concept_pool_size = filler_concept_pool_size
        self.duplicates_per_combination = duplicates_per_combination
        self.query_count = query_count
        self.final_k = final_k
        self.centroid_head_posting_cap = centroid_head_posting_cap


struct SyntheticHardRecallFixture(Copyable):
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
        raise Error("synthetic hard-recall exponent must be non-negative")

    var value = 1
    for _ in range(exponent):
        value *= base

    return value


def combination_count(read profile: SyntheticHardRecallProfile) raises -> Int:
    return int_pow(profile.values_per_slot, profile.slot_count)


def document_count(read profile: SyntheticHardRecallProfile) raises -> Int:
    return (
        combination_count(profile)
        * profile.duplicates_per_combination
        + profile.query_count
        * profile.slot_count
    )


def nominal_document_vector_count(
    read profile: SyntheticHardRecallProfile
) -> Int:
    return profile.slot_count + profile.filler_vector_count


def vector_dim(read profile: SyntheticHardRecallProfile) -> Int:
    return (
        profile.slot_count * profile.values_per_slot
        + profile.slot_count
        + profile.filler_concept_pool_size
    )


def attribute_concept_id(
    read profile: SyntheticHardRecallProfile,
    slot_index: Int,
    value_index: Int,
) -> Int:
    return slot_index * profile.values_per_slot + value_index


def filler_concept_base(read profile: SyntheticHardRecallProfile) -> Int:
    return profile.slot_count * profile.values_per_slot + profile.slot_count


def combination_values(
    read profile: SyntheticHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var total_combinations = combination_count(profile)
    if combination_index < 0 or combination_index >= total_combinations:
        raise Error(
            "synthetic hard-recall combination index must fit within the Cartesian product"
        )

    var values = List[Int]()
    var residual = combination_index

    for _ in range(profile.slot_count):
        values.append(residual % profile.values_per_slot)
        residual = residual // profile.values_per_slot

    return values^


def query_concepts_for_combination(
    read profile: SyntheticHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var values = combination_values(profile, combination_index)
    var concepts = List[Int]()

    for slot_index in range(profile.slot_count):
        concepts.append(
            attribute_concept_id(profile, slot_index, values[slot_index])
        )

    return concepts^


def filler_concepts_for_combination(
    read profile: SyntheticHardRecallProfile,
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
    read profile: SyntheticHardRecallProfile,
    combination_index: Int,
) raises -> List[Int]:
    var concepts = query_concepts_for_combination(profile, combination_index)
    for filler_concept in filler_concepts_for_combination(profile, combination_index):
        concepts.append(filler_concept)

    return concepts^


def document_id_for_combination(
    combination_index: Int, duplicate_index: Int
) -> String:
    return "doc-c" + String(combination_index) + "-d" + String(duplicate_index)


def relevant_doc_ids_for_combination(
    read profile: SyntheticHardRecallProfile,
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
        "doc-adv-q"
        + String(query_index)
        + "-s"
        + String(omitted_slot_index)
    )


def adversarial_mismatch_concept_id(
    read profile: SyntheticHardRecallProfile, omitted_slot_index: Int
) -> Int:
    return profile.slot_count * profile.values_per_slot + omitted_slot_index


def adversarial_near_miss_concepts(
    read profile: SyntheticHardRecallProfile,
    query_combination_index: Int,
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

    for filler_index in range(profile.filler_vector_count):
        concepts.append(matched_concepts[filler_index % len(matched_concepts)])

    return concepts^


def selected_query_combination_index(
    read profile: SyntheticHardRecallProfile,
    query_index: Int,
) raises -> Int:
    var total_combinations = combination_count(profile)
    if query_index < 0 or query_index >= profile.query_count:
        raise Error("synthetic hard-recall query index must fit within query_count")

    if profile.query_count > total_combinations:
        raise Error(
            "synthetic hard-recall query_count must not exceed the combination count"
        )

    return (query_index * total_combinations) // profile.query_count


def default_synthetic_hard_recall_profiles(
) raises -> List[SyntheticHardRecallProfile]:
    var profiles = List[SyntheticHardRecallProfile]()
    profiles.append(
        SyntheticHardRecallProfile(
            "synthetic_hard_recall",
            "slots6_values3_docs1530",
            "Shared-slot conjunction workload with six query vectors, a 1,458-document Cartesian base, and boosted one-slot adversarial near misses with exact mismatches.",
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
    profiles.append(
        SyntheticHardRecallProfile(
            "synthetic_hard_recall",
            "slots6_values4_docs8288",
            "Shared-slot conjunction workload with fixed query/document vector counts, an 8,192-document Cartesian base, and boosted one-slot adversarial near misses with exact mismatches.",
            6,
            4,
            24,
            48,
            2,
            16,
            2,
            16,
        )
    )
    return profiles^


def smoke_synthetic_hard_recall_profile(
) raises -> SyntheticHardRecallProfile:
    return SyntheticHardRecallProfile(
        "synthetic_hard_recall",
        "slots3_values2_docs28",
        "Small shared-slot conjunction workload for benchmark smoke coverage.",
        3,
        2,
        4,
        8,
        2,
        4,
        1,
        8,
    )


def high_centroid_synthetic_hard_recall_profile(
) raises -> SyntheticHardRecallProfile:
    return SyntheticHardRecallProfile(
        "synthetic_hard_recall",
        "imputed_probe_centroids180_docs544",
        "Benchmark-only shared-slot conjunction workload with a large filler concept pool so centroid_count exceeds the current imputed bound.",
        4,
        4,
        32,
        160,
        2,
        8,
        2,
        16,
    )


def make_synthetic_hard_recall_fixture(
    read profile: SyntheticHardRecallProfile
) raises -> SyntheticHardRecallFixture:
    var total_combinations = combination_count(profile)
    if profile.query_count > total_combinations:
        raise Error(
            "synthetic hard-recall profile query_count must fit within the combination count"
        )

    var documents = List[EncodedDocument]()
    var queries = List[JudgedQuery]()

    for combination_index in range(total_combinations):
        var concepts = document_concepts_for_combination(profile, combination_index)
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
                        adversarial_near_miss_concepts(
                            profile,
                            combination_index,
                            omitted_slot_index,
                        ),
                    ),
                )
            )

        queries.append(
            JudgedQuery(
                "query-" + String(query_index),
                "Synthetic conjunction query over shared slot values with boosted high-overlap near misses.",
                EncodedQuery(
                    concept_vectors(
                        vector_dim(profile),
                        query_concepts_for_combination(profile, combination_index),
                    )
                ),
                relevant_doc_ids_for_combination(profile, combination_index),
            )
        )

    var dataset_id = "synthetic://hard-recall/" + profile.slice_name
    var model_name = "synthetic-slot-conjunctions"
    var packed_index = pack_documents(documents)

    return SyntheticHardRecallFixture(
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
