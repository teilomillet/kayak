# Text-family stage-2 refinement over a bounded candidate window.

from std.collections import List

from kayak.collections import ResolvedCollectionSnapshot
from kayak.search import SearchHit
from kayak.text import DocumentTextCorpus, normalize_text
from kayak.verifier import (
    default_clause_text_rerank_config,
    rerank_hits_clause_text,
)

from .collection_hit import CollectionHit, to_search_hit
from .stage_artifact_materialization import StageArtifactMaterialization
from .stage2_operator import STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT
from .stage2_result import Stage2Result


struct CandidateTextWindow(Copyable):
    var corpus: DocumentTextCorpus
    var token_count: Int
    var byte_size: Int

    def __init__(
        out self,
        corpus: DocumentTextCorpus,
        token_count: Int,
        byte_size: Int,
    ):
        self.corpus = corpus.copy()
        self.token_count = token_count
        self.byte_size = byte_size


def count_text_tokens(text: String) -> Int:
    var normalized = normalize_text(text)
    if normalized.byte_length() == 0:
        return 0

    var token_count = 0
    for token in normalized.split(" "):
        if String(token).byte_length() == 0:
            continue
        token_count += 1

    return token_count


def count_unique_segment_ids_in_hits(read hits: List[CollectionHit]) -> Int:
    var unique_segment_ids = List[String]()

    for hit in hits:
        var already_seen = False
        for unique_segment_id in unique_segment_ids:
            if unique_segment_id == hit.segment_id:
                already_seen = True
                break

        if not already_seen:
            unique_segment_ids.append(hit.segment_id.copy())

    return len(unique_segment_ids)


def candidate_text_corpus_for_hits(
    read snapshot: ResolvedCollectionSnapshot,
    read hits: List[CollectionHit],
) raises -> CandidateTextWindow:
    var doc_ids = List[String]()
    var texts = List[String]()
    var token_count = 0
    var byte_size = 0

    for hit in hits:
        var matched_segment = False

        for segment in snapshot.segments:
            if segment.manifest.segment_id.value != hit.segment_id:
                continue

            if not segment.has_text_corpus:
                raise Error(
                    "clause_text stage-2 requires text_corpus for every candidate segment"
                )

            var matched_document = False
            for document_index in range(
                len(segment.stored_text_corpus.corpus.doc_ids)
            ):
                if segment.stored_text_corpus.corpus.doc_ids[document_index] != hit.doc_id:
                    continue

                var text = segment.stored_text_corpus.corpus.texts[document_index].copy()
                doc_ids.append(hit.doc_id.copy())
                texts.append(text.copy())
                token_count += count_text_tokens(text)
                byte_size += text.byte_length()
                matched_document = True
                break

            if not matched_document:
                raise Error(
                    "candidate doc_id not found in text corpus: " + hit.doc_id
                )

            matched_segment = True
            break

        if not matched_segment:
            raise Error(
                "candidate segment_id not found in resolved snapshot: " + hit.segment_id
            )

    return CandidateTextWindow(
        DocumentTextCorpus(doc_ids^, texts^),
        token_count,
        byte_size,
    )


def clause_text_rerank_candidates_for_plan(
    query_text: String,
    read snapshot: ResolvedCollectionSnapshot,
    read hits: List[CollectionHit],
    final_k: Int,
) raises -> Stage2Result:
    if len(hits) == 0:
        return Stage2Result([], 0, 0, 0, 0, 0)

    if query_text.byte_length() == 0:
        raise Error("clause_text stage-2 requires non-empty query_text")

    var candidate_texts = candidate_text_corpus_for_hits(
        snapshot,
        hits,
    )
    var search_hits = List[SearchHit]()
    for hit in hits:
        search_hits.append(to_search_hit(hit))
    var reranked_hits = rerank_hits_clause_text(
        query_text,
        search_hits^,
        candidate_texts.corpus,
        final_k,
        default_clause_text_rerank_config(),
    )

    var final_hits = List[CollectionHit]()
    for reranked_hit in reranked_hits:
        var matched = False
        for hit in hits:
            if hit.doc_id != reranked_hit.doc_id:
                continue
            final_hits.append(
                CollectionHit(
                    hit.segment_id.copy(),
                    hit.doc_id.copy(),
                    reranked_hit.score,
                )
            )
            matched = True
            break

        if not matched:
            raise Error(
                "reranked clause_text doc_id not found in candidate window: "
                + reranked_hit.doc_id
            )

    return Stage2Result(
        final_hits^,
        [
            StageArtifactMaterialization(
                STAGE2_REQUIRED_ARTIFACT_DOCUMENT_TEXT,
                count_unique_segment_ids_in_hits(hits),
                len(hits),
                candidate_texts.token_count,
                0,
                candidate_texts.byte_size,
            )
        ],
        count_unique_segment_ids_in_hits(hits),
        len(hits),
        candidate_texts.token_count,
        0,
        candidate_texts.byte_size,
    )
