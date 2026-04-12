"""Owns score vectors and stable top-k ranking over explicit document ids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_conversions import readonly_array
from .dtypes import SCORE_DTYPE

try:
    import torch
except ImportError:  # pragma: no cover - torch is present in the managed env.
    torch = None


@dataclass(frozen=True, slots=True)
class SearchHit:
    doc_id: str
    score: float


@dataclass(frozen=True, slots=True)
class LateScores:
    backend: str
    doc_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.doc_ids) == 0:
            raise ValueError("scores must contain at least one document")
        if self.values.shape != (len(self.doc_ids),):
            raise ValueError("scores and doc_ids must have matching lengths")

    def numpy(self) -> np.ndarray:
        return self.values

    def torch(self) -> "torch.Tensor":
        if torch is None:  # pragma: no cover - torch is present in the env.
            raise RuntimeError("torch is not installed")
        return torch.tensor(self.values, dtype=torch.float32)

    def topk(self, k: int) -> tuple[SearchHit, ...]:
        if k < 0:
            raise ValueError("top-k requires a non-negative k")
        if k == 0:
            return ()

        hits: list[SearchHit] = []
        for doc_id, score in zip(self.doc_ids, self.values, strict=True):
            hit = SearchHit(doc_id=doc_id, score=float(score))
            insert_at = 0
            while insert_at < len(hits) and hits[insert_at].score >= hit.score:
                insert_at += 1

            if insert_at >= k:
                if len(hits) < k:
                    hits.append(hit)
                continue

            hits.insert(insert_at, hit)
            if len(hits) > k:
                hits.pop()

        return tuple(hits)

    @classmethod
    def from_values(
        cls, backend: str, doc_ids: tuple[str, ...], values: np.ndarray
    ) -> "LateScores":
        return cls(
            backend=backend,
            doc_ids=doc_ids,
            values=readonly_array(values, dtype=SCORE_DTYPE),
        )
