"""Owns OpenAI-like helpers for single-vector dense chunk baselines.

This module owns:
- an explicit OpenAI embeddings adapter with optional on-disk caching
- a tiktoken-backed source tokenizer wrapper for OpenAI-style chunk sizes

It does not own judged-metric evaluation or chunk-search orchestration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from .cache_paths import CACHE_ROOT
from .cache_paths import configure_local_caches


configure_local_caches()

OPENAI_EMBEDDING_CACHE_ROOT = CACHE_ROOT / "openai_embeddings"


class TiktokenSourceTokenizer:
    """Wrap one tiktoken encoding behind the source-tokenizer protocol."""

    def __init__(
        self,
        model_name: str,
        *,
        encoding_name: str | None = None,
        model_max_length: int = 4096,
    ) -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError(
                "tiktoken is required for the OpenAI-like dense chunk baseline. "
                "Install it with `uv run --with tiktoken ...` or "
                "`uv pip install --python ./.venv/bin/python tiktoken`."
            ) from exc

        self.model_name = model_name
        self.model_max_length = int(model_max_length)
        if encoding_name is None:
            self._encoding = tiktoken.encoding_for_model(model_name)
            self.encoding_name = str(self._encoding.name)
        else:
            self._encoding = tiktoken.get_encoding(encoding_name)
            self.encoding_name = encoding_name

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError(
                "tiktoken-based source chunking does not inject special tokens"
            )
        return list(self._encoding.encode(text))

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
    ) -> str:
        del skip_special_tokens
        return str(self._encoding.decode(list(token_ids)))


class OpenAIEmbeddingTextBatcher:
    """Batch text embedding through OpenAI with optional explicit file caching."""

    def __init__(
        self,
        *,
        client: object | None = None,
        dimensions: int | None = None,
        cache_dir: str | Path | None = OPENAI_EMBEDDING_CACHE_ROOT,
    ) -> None:
        self._client = client
        self.dimensions = dimensions
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolved_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is required for the OpenAI-like dense chunk baseline. "
                "Install it with `uv run --with openai ...` or "
                "`uv pip install --python ./.venv/bin/python openai`."
            ) from exc
        self._client = OpenAI()
        return self._client

    def _cache_key(self, *, model_name: str, text: str) -> str:
        payload = json.dumps(
            {
                "model_name": model_name,
                "dimensions": self.dimensions,
                "text": text,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, *, model_name: str, text: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is not configured")
        return self.cache_dir / f"{self._cache_key(model_name=model_name, text=text)}.json"

    def _load_cached_embedding(
        self,
        *,
        model_name: str,
        text: str,
    ) -> list[float] | None:
        if self.cache_dir is None:
            return None
        path = self._cache_path(model_name=model_name, text=text)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [float(value) for value in payload["embedding"]]

    def _store_cached_embedding(
        self,
        *,
        model_name: str,
        text: str,
        embedding: Sequence[float],
    ) -> None:
        if self.cache_dir is None:
            return
        path = self._cache_path(model_name=model_name, text=text)
        payload = {
            "model_name": model_name,
            "dimensions": self.dimensions,
            "embedding": [float(value) for value in embedding],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def embed_texts(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ) -> tuple[list[float], ...]:
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        if not texts:
            return ()

        embeddings: list[list[float] | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self._load_cached_embedding(model_name=model_name, text=text)
            if cached is not None:
                embeddings[index] = cached
                continue
            pending_indices.append(index)
            pending_texts.append(text)

        if pending_texts:
            client = self._resolved_client()
            request_kwargs: dict[str, object] = {"model": model_name}
            if self.dimensions is not None:
                request_kwargs["dimensions"] = int(self.dimensions)
            for start in range(0, len(pending_texts), batch_size):
                batch_texts = pending_texts[start : start + batch_size]
                response = client.embeddings.create(
                    input=batch_texts,
                    **request_kwargs,
                )
                batch_vectors = [
                    [float(value) for value in item.embedding]
                    for item in response.data
                ]
                batch_indices = pending_indices[start : start + batch_size]
                for row_index, vector in zip(
                    batch_indices,
                    batch_vectors,
                    strict=True,
                ):
                    text = texts[row_index]
                    embeddings[row_index] = vector
                    self._store_cached_embedding(
                        model_name=model_name,
                        text=text,
                        embedding=vector,
                    )

        if any(vector is None for vector in embeddings):
            raise RuntimeError("OpenAI embedding batch returned incomplete results")
        return tuple(embedding for embedding in embeddings if embedding is not None)
