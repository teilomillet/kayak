"""Reference text-family stage-2 scoring for the Python SDK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dtypes import SCORE_DTYPE
from .late_scores import LateScores, SearchHit


@dataclass(frozen=True, slots=True)
class ClauseTextStageConfig:
    token_weight_scale: float = 1.0
    phrase_weight: float = 1.5
    clause_match_bonus: float = 1.0
    final_clause_bonus: float = 0.5
    blend_weight: float = 0.2
    context_clause_scale: float = 0.35
    final_clause_scale: float = 1.0


def default_clause_text_stage_config() -> ClauseTextStageConfig:
    return ClauseTextStageConfig()


def normalize_text(text: str) -> str:
    normalized = text.lower()
    for source in (
        "\r",
        "\n",
        "\t",
        ".",
        ",",
        ":",
        ";",
        "?",
        "!",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        '"',
        "'",
        "-",
        "/",
        "\\",
    ):
        normalized = normalized.replace(source, " ")
    return normalized


def _is_stopword(token: str) -> bool:
    return token in {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "held",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "of",
        "on",
        "or",
        "our",
        "please",
        "provide",
        "said",
        "seven",
        "so",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "to",
        "today",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "would",
        "years",
    }


def _token_weight(token: str, clause_frequency: int) -> float:
    weight = 1.0
    if len(token) >= 6:
        weight += 0.5
    if clause_frequency == 1:
        weight += 1.0
    elif clause_frequency == 2:
        weight += 0.5
    return weight


def _unique_tokens(tokens: list[str]) -> list[str]:
    unique: list[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
    return unique


def _tokenize_content(text: str) -> list[str]:
    tokens = []
    for token in normalize_text(text).split(" "):
        if len(token) < 3:
            continue
        if _is_stopword(token):
            continue
        tokens.append(token)
    return _unique_tokens(tokens)


def _split_query_clauses(query_text: str) -> list[str]:
    normalized = query_text.lower()
    normalized = normalized.replace("?", ".")
    normalized = normalized.replace("!", ".")
    normalized = normalized.replace(":", ".")
    clauses: list[str] = []
    for clause in normalized.split("."):
        trimmed = normalize_text(clause).strip()
        if trimmed:
            clauses.append(trimmed)
    return clauses


def _clause_frequency(token: str, clause_tokens: list[list[str]]) -> int:
    frequency = 0
    for tokens in clause_tokens:
        if token in tokens:
            frequency += 1
    return frequency


def clause_text_boost(
    query_text: str,
    doc_text: str,
    *,
    config: ClauseTextStageConfig = ClauseTextStageConfig(),
) -> float:
    clauses = _split_query_clauses(query_text)
    clause_tokens = [_tokenize_content(clause) for clause in clauses]
    normalized_doc_text = f" {normalize_text(doc_text)} "
    total = 0.0

    for clause_index, tokens in enumerate(clause_tokens):
        if not tokens:
            continue

        clause_scale = config.context_clause_scale
        if clause_index == len(clause_tokens) - 1:
            clause_scale = config.final_clause_scale

        token_matches = 0
        for token in tokens:
            if f" {token} " in normalized_doc_text:
                token_matches += 1
                total += (
                    clause_scale
                    * config.token_weight_scale
                    * _token_weight(token, _clause_frequency(token, clause_tokens))
                )

        if token_matches >= 2:
            total += clause_scale * config.clause_match_bonus

        if clause_index == len(clause_tokens) - 1:
            total += config.final_clause_bonus * float(token_matches)

        for token_index in range(len(tokens) - 1):
            phrase = f"{tokens[token_index]} {tokens[token_index + 1]}"
            if phrase in normalized_doc_text:
                total += clause_scale * config.phrase_weight

    return total


def clause_text_scores(
    query_text: str,
    hits: tuple[SearchHit, ...],
    doc_texts: tuple[str, ...],
    *,
    config: ClauseTextStageConfig = ClauseTextStageConfig(),
) -> LateScores:
    if len(hits) != len(doc_texts):
        raise ValueError("clause-text hits and document texts must align")

    values = np.empty((len(hits),), dtype=SCORE_DTYPE)
    for index, (hit, doc_text) in enumerate(zip(hits, doc_texts, strict=True)):
        values[index] = np.float32(
            hit.score + (config.blend_weight * clause_text_boost(query_text, doc_text))
        )

    return LateScores.from_values(
        "clause_text",
        tuple(hit.doc_id for hit in hits),
        values,
    )
