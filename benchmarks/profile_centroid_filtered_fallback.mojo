import std.benchmark as benchmark
import std.benchmark.compiler as bench_compiler

from std.collections import List

from kayak.numeric import ScoreScalar
from kayak.planning.collection_hit import CollectionHit
from kayak.planning.topk import insert_descending_collection_hit


comptime DOCUMENT_COUNT = 50_000
comptime FILTER_MATCH_COUNT = 64
comptime ACTIVE_MATCH_COUNT = 8
comptime CANDIDATE_K = 40


def make_doc_ids() -> List[String]:
    var doc_ids = List[String]()
    for document_index in range(DOCUMENT_COUNT):
        doc_ids.append("doc-" + String(document_index))
    return doc_ids^


def make_allowed_doc_indices() -> List[Int]:
    var doc_indices = List[Int]()
    var stride = DOCUMENT_COUNT // FILTER_MATCH_COUNT
    if stride <= 0:
        stride = 1

    var document_index = 0
    while len(doc_indices) < FILTER_MATCH_COUNT and document_index < DOCUMENT_COUNT:
        doc_indices.append(document_index)
        document_index += stride

    return doc_indices^


def make_allowed_flags(read allowed_doc_indices: List[Int]) -> List[Int]:
    var flags = List[Int]()
    for _ in range(DOCUMENT_COUNT):
        flags.append(0)

    for doc_index in allowed_doc_indices:
        flags[doc_index] = 1

    return flags^


def make_active_doc_indices(read allowed_doc_indices: List[Int]) -> List[Int]:
    var active_doc_indices = List[Int]()
    for active_index in range(ACTIVE_MATCH_COUNT):
        active_doc_indices.append(allowed_doc_indices[active_index])
    return active_doc_indices^


def make_scores(read active_doc_indices: List[Int]) -> List[ScoreScalar]:
    var scores = List[ScoreScalar]()
    for _ in range(DOCUMENT_COUNT):
        scores.append(ScoreScalar(0.0))

    for active_index in range(len(active_doc_indices)):
        var doc_index = active_doc_indices[active_index]
        scores[doc_index] = ScoreScalar(1.0 - Float64(active_index) * 0.01)

    return scores^


def insert_active_hits(
    mut hits: List[CollectionHit],
    segment_id: String,
    read doc_ids: List[String],
    read scores: List[ScoreScalar],
    read active_doc_indices: List[Int],
):
    for document_index in active_doc_indices:
        insert_descending_collection_hit(
            hits,
            CollectionHit(
                segment_id.copy(),
                doc_ids[document_index].copy(),
                scores[document_index],
            ),
            CANDIDATE_K,
        )


def dense_filtered_fallback(
    segment_id: String,
    read doc_ids: List[String],
    read scores: List[ScoreScalar],
    read active_doc_indices: List[Int],
    read allowed_flags: List[Int],
) -> List[CollectionHit]:
    var hits = List[CollectionHit]()
    insert_active_hits(
        hits,
        segment_id,
        doc_ids,
        scores,
        active_doc_indices,
    )

    var active_flags = List[Int]()
    for _ in range(len(doc_ids)):
        active_flags.append(0)

    for document_index in active_doc_indices:
        active_flags[document_index] = 1

    for document_index in range(len(doc_ids)):
        if active_flags[document_index] != 0:
            continue
        if allowed_flags[document_index] == 0:
            continue
        insert_descending_collection_hit(
            hits,
            CollectionHit(
                segment_id.copy(),
                doc_ids[document_index].copy(),
                ScoreScalar(0.25),
            ),
            CANDIDATE_K,
        )

    return hits^


def indexed_filtered_fallback(
    segment_id: String,
    read doc_ids: List[String],
    read scores: List[ScoreScalar],
    read active_doc_indices: List[Int],
    read allowed_doc_indices: List[Int],
) -> List[CollectionHit]:
    var hits = List[CollectionHit]()
    insert_active_hits(
        hits,
        segment_id,
        doc_ids,
        scores,
        active_doc_indices,
    )

    var active_flags = List[Int]()
    for _ in range(len(doc_ids)):
        active_flags.append(0)

    for document_index in active_doc_indices:
        active_flags[document_index] = 1

    for document_index in allowed_doc_indices:
        if active_flags[document_index] != 0:
            continue
        insert_descending_collection_hit(
            hits,
            CollectionHit(
                segment_id.copy(),
                doc_ids[document_index].copy(),
                ScoreScalar(0.25),
            ),
            CANDIDATE_K,
        )

    return hits^


def main() raises:
    var segment_id = String("segment-0001")
    var doc_ids = make_doc_ids()
    var allowed_doc_indices = make_allowed_doc_indices()
    var allowed_flags = make_allowed_flags(allowed_doc_indices)
    var active_doc_indices = make_active_doc_indices(allowed_doc_indices)
    var scores = make_scores(active_doc_indices)

    print("document_count=", DOCUMENT_COUNT)
    print("filter_match_count=", FILTER_MATCH_COUNT)
    print("active_match_count=", ACTIVE_MATCH_COUNT)
    print("")

    print("== dense_filtered_fallback ==")

    def dense_once() capturing:
        bench_compiler.keep(
            dense_filtered_fallback(
                segment_id,
                doc_ids,
                scores,
                active_doc_indices,
                allowed_flags,
            )
        )

    var dense_report = benchmark.run[dense_once]()
    dense_report.print()
    print("")

    print("== indexed_filtered_fallback ==")

    def indexed_once() capturing:
        bench_compiler.keep(
            indexed_filtered_fallback(
                segment_id,
                doc_ids,
                scores,
                active_doc_indices,
                allowed_doc_indices,
            )
        )

    var indexed_report = benchmark.run[indexed_once]()
    indexed_report.print()
    print("")

    print("dense_mean_seconds=", dense_report.mean())
    print("indexed_mean_seconds=", indexed_report.mean())
    print("speedup=", dense_report.mean() / indexed_report.mean())
