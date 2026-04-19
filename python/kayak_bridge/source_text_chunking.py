"""Owns explicit source-text chunking primitives for retrieval baselines.

This module owns source-token chunk specifications plus tokenizer-driven text
windowing. It does not own embedding, scoring, or judged-metric evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping
from typing import Protocol
from typing import Sequence

from transformers import AutoTokenizer


class SourceTokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], skip_special_tokens: bool = True
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class RawTextChunkSpec:
    max_chunk_tokens: int
    overlap_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be non-negative")
        if self.overlap_tokens >= self.max_chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than max_chunk_tokens")

    @property
    def label(self) -> str:
        return (
            f"raw_text_chunk_{self.max_chunk_tokens}_overlap_{self.overlap_tokens}"
        )


@lru_cache(maxsize=8)
def load_hf_source_tokenizer(model_name: str) -> SourceTokenizer:
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def effective_model_max_length(tokenizer: SourceTokenizer) -> int | None:
    raw_value = getattr(tokenizer, "model_max_length", None)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if value > 100_000:
        return None
    return value


def source_token_ids(
    text: str,
    *,
    tokenizer: SourceTokenizer,
) -> tuple[int, ...]:
    """Return full source-token ids without tokenizer length warnings.

    Source chunking needs the entire token stream before windowing, so this path
    must stay explicitly untruncated. Fast Hugging Face tokenizers support a
    callable interface with `verbose=False`, which suppresses misleading
    max-length warnings for this pre-chunk stage. Non-callable tokenizers fall
    back to `encode()`.
    """

    tokenizer_call = getattr(tokenizer, "__call__", None)
    if callable(tokenizer_call):
        for kwargs in (
            {
                "add_special_tokens": False,
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
                "verbose": False,
            },
            {
                "add_special_tokens": False,
                "truncation": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            },
        ):
            try:
                encoded = tokenizer_call(text, **kwargs)
            except TypeError:
                continue
            if isinstance(encoded, Mapping) and "input_ids" in encoded:
                input_ids = encoded["input_ids"]
                if (
                    isinstance(input_ids, Sequence)
                    and input_ids
                    and isinstance(input_ids[0], Sequence)
                ):
                    input_ids = input_ids[0]
                return tuple(int(token_id) for token_id in input_ids)
    return tuple(int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False))


def _chunk_token_ids(
    token_ids: Sequence[int],
    *,
    max_chunk_tokens: int,
    overlap_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, max_chunk_tokens)")
    if not token_ids:
        return ((),)

    step = max_chunk_tokens - overlap_tokens
    chunks: list[tuple[int, ...]] = []
    start = 0
    while start < len(token_ids):
        stop = min(start + max_chunk_tokens, len(token_ids))
        chunks.append(tuple(token_ids[start:stop]))
        if stop >= len(token_ids):
            break
        start += step
    return tuple(chunks)


def chunk_text_by_source_tokens(
    text: str,
    *,
    tokenizer: SourceTokenizer,
    spec: RawTextChunkSpec,
) -> tuple[str, ...]:
    token_ids = source_token_ids(text, tokenizer=tokenizer)
    chunk_token_ids = _chunk_token_ids(
        token_ids,
        max_chunk_tokens=spec.max_chunk_tokens,
        overlap_tokens=spec.overlap_tokens,
    )
    if chunk_token_ids == ((),):
        return (text,)
    return tuple(
        tokenizer.decode(chunk_ids, skip_special_tokens=True)
        for chunk_ids in chunk_token_ids
    )
