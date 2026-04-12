from std.collections import List
from std.math import min

from kayak.numeric import ScoreScalar
from kayak.search import SearchHit
from kayak.text import (
    DocumentTextCorpus,
    document_text_for_doc_id,
    normalize_text,
)


struct ClauseTextRerankConfig(Copyable):
    var token_weight_scale: ScoreScalar
    var phrase_weight: ScoreScalar
    var clause_match_bonus: ScoreScalar
    var final_clause_bonus: ScoreScalar
    var blend_weight: ScoreScalar
    var context_clause_scale: ScoreScalar
    var final_clause_scale: ScoreScalar

    def __init__(out self):
        self.token_weight_scale = 1.0
        self.phrase_weight = 1.5
        self.clause_match_bonus = 1.0
        self.final_clause_bonus = 0.5
        self.blend_weight = 0.2
        self.context_clause_scale = 0.35
        self.final_clause_scale = 1.0

    def __init__(
        out self,
        token_weight_scale: ScoreScalar,
        phrase_weight: ScoreScalar,
        clause_match_bonus: ScoreScalar,
        final_clause_bonus: ScoreScalar,
        blend_weight: ScoreScalar,
        context_clause_scale: ScoreScalar,
        final_clause_scale: ScoreScalar,
    ):
        self.token_weight_scale = token_weight_scale
        self.phrase_weight = phrase_weight
        self.clause_match_bonus = clause_match_bonus
        self.final_clause_bonus = final_clause_bonus
        self.blend_weight = blend_weight
        self.context_clause_scale = context_clause_scale
        self.final_clause_scale = final_clause_scale


def default_clause_text_rerank_config() -> ClauseTextRerankConfig:
    return ClauseTextRerankConfig()


def is_stopword(token: String) -> Bool:
    if token == "a" or token == "an" or token == "and":
        return True
    if token == "are" or token == "as" or token == "at":
        return True
    if token == "be" or token == "been" or token == "but":
        return True
    if token == "by" or token == "can" or token == "could":
        return True
    if token == "did" or token == "do" or token == "does":
        return True
    if token == "for" or token == "from" or token == "had":
        return True
    if token == "has" or token == "have" or token == "held":
        return True
    if token == "in" or token == "into" or token == "is":
        return True
    if token == "it" or token == "its" or token == "me":
        return True
    if token == "of" or token == "on" or token == "or":
        return True
    if token == "our" or token == "please" or token == "provide":
        return True
    if token == "said" or token == "seven" or token == "so":
        return True
    if token == "tell" or token == "than" or token == "that":
        return True
    if token == "the" or token == "their" or token == "there":
        return True
    if token == "these" or token == "this" or token == "those":
        return True
    if token == "through" or token == "to" or token == "today":
        return True
    if token == "was" or token == "were" or token == "what":
        return True
    if token == "when" or token == "where" or token == "which":
        return True
    if token == "who" or token == "with" or token == "would":
        return True
    if token == "years":
        return True

    return False


def token_weight(token: String, clause_frequency: Int) -> ScoreScalar:
    var weight = ScoreScalar(1.0)

    if token.byte_length() >= 6:
        weight += 0.5

    if clause_frequency == 1:
        weight += 1.0
    elif clause_frequency == 2:
        weight += 0.5

    return weight


def unique_tokens(tokens: List[String]) -> List[String]:
    var unique = List[String]()

    for token in tokens:
        var already_seen = False

        for seen in unique:
            if seen == token:
                already_seen = True
                break

        if not already_seen:
            unique.append(token.copy())

    return unique^


def tokenize_content(text: String) -> List[String]:
    var tokens = List[String]()

    for token in normalize_text(text).split(" "):
        var cleaned = String(token)

        if cleaned.byte_length() < 3:
            continue

        if is_stopword(cleaned):
            continue

        tokens.append(cleaned^)
    return unique_tokens(tokens)


def split_query_clauses(query_text: String) -> List[String]:
    var clauses = List[String]()
    var normalized = query_text.lower()
    normalized = normalized.replace("?", ".")
    normalized = normalized.replace("!", ".")
    normalized = normalized.replace(":", ".")

    for clause in normalized.split("."):
        var trimmed = normalize_text(String(clause))
        if trimmed.byte_length() == 0:
            continue

        clauses.append(trimmed^)

    return clauses^


def token_present(normalized_doc_text: String, token: String) -> Bool:
    return normalized_doc_text.find(" " + token + " ") != -1


def phrase_present(normalized_doc_text: String, phrase: String) -> Bool:
    return normalized_doc_text.find(phrase) != -1


def clause_frequency(token: String, clause_tokens: List[List[String]]) -> Int:
    var frequency = 0

    for tokens in clause_tokens:
        for clause_token in tokens:
            if clause_token == token:
                frequency += 1
                break

    return frequency


def clause_text_boost(
    query_text: String,
    doc_text: String,
    read config: ClauseTextRerankConfig,
) -> ScoreScalar:
    var clauses = split_query_clauses(query_text)
    var clause_tokens = List[List[String]]()

    for clause in clauses:
        clause_tokens.append(tokenize_content(clause))

    var normalized_doc_text = " " + normalize_text(doc_text) + " "
    var total = ScoreScalar(0.0)

    for clause_index in range(len(clause_tokens)):
        var tokens = clause_tokens[clause_index].copy()
        if len(tokens) == 0:
            continue

        var clause_scale = config.context_clause_scale
        if clause_index == len(clause_tokens) - 1:
            clause_scale = config.final_clause_scale

        var token_matches = 0

        for token in tokens:
            if token_present(normalized_doc_text, token):
                token_matches += 1
                total += (
                    clause_scale
                    * 
                    config.token_weight_scale
                    * token_weight(token, clause_frequency(token, clause_tokens))
                )

        if token_matches >= 2:
            total += clause_scale * config.clause_match_bonus

        if clause_index == len(clause_tokens) - 1:
            total += config.final_clause_bonus * ScoreScalar(token_matches)

        for token_index in range(len(tokens) - 1):
            var phrase = tokens[token_index] + " " + tokens[token_index + 1]
            if phrase_present(normalized_doc_text, phrase):
                total += clause_scale * config.phrase_weight

    return total


def insert_descending(mut hits: List[SearchHit], var hit: SearchHit):
    var insert_at = 0
    while insert_at < len(hits) and hits[insert_at].score >= hit.score:
        insert_at += 1

    hits.append(hit.copy())

    var current = len(hits) - 1
    while current > insert_at:
        hits[current] = hits[current - 1].copy()
        current -= 1

    hits[insert_at] = hit^


def rescore_hits_clause_text(
    query_text: String,
    hits: List[SearchHit],
    read document_texts: DocumentTextCorpus,
    read config: ClauseTextRerankConfig = default_clause_text_rerank_config(),
) raises -> List[SearchHit]:
    var reranked_hits = List[SearchHit]()

    for hit in hits:
        var doc_text = document_text_for_doc_id(document_texts, hit.doc_id)
        var lexical_boost = clause_text_boost(query_text, doc_text, config)
        insert_descending(
            reranked_hits,
            SearchHit(
                hit.doc_id.copy(),
                hit.score + (config.blend_weight * lexical_boost),
            ),
        )

    return reranked_hits^


def rerank_hits_clause_text(
    query_text: String,
    hits: List[SearchHit],
    read document_texts: DocumentTextCorpus,
    k: Int,
    read config: ClauseTextRerankConfig = default_clause_text_rerank_config(),
) raises -> List[SearchHit]:
    var reranked_hits = rescore_hits_clause_text(
        query_text, hits, document_texts, config
    )
    var final_hits = List[SearchHit]()
    var limit = min(k, len(reranked_hits))

    for index in range(limit):
        final_hits.append(reranked_hits[index].copy())

    return final_hits^
