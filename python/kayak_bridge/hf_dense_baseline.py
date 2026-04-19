"""Owns local Hugging Face helpers for one-vector dense chunk baselines.

This module owns:
- one lightweight transformer mean-pooling embedder
- optional explicit file caching for repeated local benchmark runs

It does not own chunking, retrieval, or judged-metric evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from .cache_paths import CACHE_ROOT
from .cache_paths import configure_local_caches


configure_local_caches()

HF_DENSE_EMBEDDING_CACHE_ROOT = CACHE_ROOT / "hf_dense_embeddings"


class HFDenseTextBatcher:
    """Batch local transformer embeddings with explicit mean pooling."""

    def __init__(
        self,
        model_name: str,
        *,
        tokenizer: object | None = None,
        model: object | None = None,
        cache_dir: str | Path | None = HF_DENSE_EMBEDDING_CACHE_ROOT,
        max_length: int | None = None,
        text_prefix: str = "",
    ) -> None:
        self.model_name = model_name
        self._tokenizer = tokenizer
        self._model = model
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.max_length = max_length
        self.text_prefix = text_prefix
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolved_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        return self._tokenizer

    def _resolved_model(self):
        if self._model is not None:
            return self._model
        import torch
        from transformers import AutoModel

        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        torch.set_num_threads(1)
        return self._model

    def _cache_key(self, *, model_name: str, text: str) -> str:
        payload = json.dumps(
            {
                "model_name": model_name,
                "max_length": self.max_length,
                "text_prefix": self.text_prefix,
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
            "max_length": self.max_length,
            "text_prefix": self.text_prefix,
            "embedding": [float(value) for value in embedding],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _prefixed_text(self, text: str) -> str:
        if self.text_prefix == "":
            return text
        return f"{self.text_prefix}{text}"

    def embed_texts(
        self,
        texts: list[str],
        model_name: str,
        batch_size: int,
    ) -> tuple[list[float], ...]:
        if batch_size <= 0:
            raise ValueError("embedding batch_size must be positive")
        if model_name != self.model_name:
            raise ValueError(
                "HFDenseTextBatcher received a mismatched model_name; "
                f"expected {self.model_name}, got {model_name}"
            )
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
            pending_texts.append(self._prefixed_text(text))

        if pending_texts:
            import torch

            tokenizer = self._resolved_tokenizer()
            model = self._resolved_model()
            resolved_max_length = (
                int(self.max_length)
                if self.max_length is not None
                else getattr(tokenizer, "model_max_length", None)
            )
            with torch.inference_mode():
                for start in range(0, len(pending_texts), batch_size):
                    batch_texts = pending_texts[start : start + batch_size]
                    tokenized = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=resolved_max_length,
                        return_tensors="pt",
                    )
                    outputs = model(**tokenized)
                    hidden = outputs.last_hidden_state
                    attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                    masked_hidden = hidden * attention_mask
                    row_sums = masked_hidden.sum(dim=1)
                    token_counts = attention_mask.sum(dim=1).clamp(min=1)
                    pooled = row_sums / token_counts
                    pooled_rows = pooled.detach().cpu().tolist()
                    batch_indices = pending_indices[start : start + batch_size]
                    for row_index, vector in zip(
                        batch_indices,
                        pooled_rows,
                        strict=True,
                    ):
                        raw_text = texts[row_index]
                        materialized = [float(value) for value in vector]
                        embeddings[row_index] = materialized
                        self._store_cached_embedding(
                            model_name=model_name,
                            text=raw_text,
                            embedding=materialized,
                        )

        if any(vector is None for vector in embeddings):
            raise RuntimeError("HF dense embedding batch returned incomplete results")
        return tuple(embedding for embedding in embeddings if embedding is not None)
