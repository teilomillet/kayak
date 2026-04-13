"""Owns explicit late-query layouts; it does not own scoring or backend dispatch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array_conversions import (
    flatten_vector_matrix,
    reshape_flat_values,
    to_optional_text,
    to_flat_vector_values,
    to_vector_matrix,
)
from .dtypes import FLAT_DIM128_VECTOR_DIM
from .layouts import QUERY_LAYOUT_FLAT_DIM128, QUERY_LAYOUT_NESTED


@dataclass(frozen=True, slots=True)
class LateQuery:
    layout: str
    vector_dim: int
    vector_count: int
    text: str | None = None
    token_vectors: np.ndarray | None = None
    token_values: np.ndarray | None = None

    @classmethod
    def from_vectors(
        cls, token_vectors: object, *, text: object | None = None
    ) -> "LateQuery":
        matrix = to_vector_matrix(token_vectors, "query")
        return cls(
            layout=QUERY_LAYOUT_NESTED,
            vector_dim=int(matrix.shape[1]),
            vector_count=int(matrix.shape[0]),
            text=to_optional_text(text, "query"),
            token_vectors=matrix,
        )

    @classmethod
    def from_flat_values(
        cls,
        token_values: object,
        *,
        vector_dim: int,
        text: object | None = None,
    ) -> "LateQuery":
        values = to_flat_vector_values(token_values, "flat query")
        if vector_dim != FLAT_DIM128_VECTOR_DIM:
            raise ValueError("flat_dim128 query requires vector_dim=128")
        if values.size % vector_dim != 0:
            raise ValueError(
                "flat_dim128 query values must be aligned to vector_dim"
            )

        return cls(
            layout=QUERY_LAYOUT_FLAT_DIM128,
            vector_dim=vector_dim,
            vector_count=int(values.size // vector_dim),
            text=to_optional_text(text, "query"),
            token_values=values,
        )

    def __post_init__(self) -> None:
        if self.vector_dim <= 0:
            raise ValueError("query vector_dim must be positive")
        if self.vector_count <= 0:
            raise ValueError("query must contain at least one vector")

        if self.layout == QUERY_LAYOUT_NESTED:
            if self.token_vectors is None or self.token_values is not None:
                raise ValueError("nested queries own only token_vectors")
            if self.token_vectors.shape != (self.vector_count, self.vector_dim):
                raise ValueError("nested query shape does not match metadata")
            return

        if self.layout == QUERY_LAYOUT_FLAT_DIM128:
            if self.vector_dim != FLAT_DIM128_VECTOR_DIM:
                raise ValueError("flat_dim128 query requires vector_dim=128")
            if self.token_values is None or self.token_vectors is not None:
                raise ValueError("flat_dim128 queries own only token_values")
            if self.token_values.size != self.vector_count * self.vector_dim:
                raise ValueError("flat_dim128 query values do not match metadata")
            return

        raise ValueError(f"unsupported query layout: {self.layout}")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.vector_count, self.vector_dim)

    def to_layout(self, layout: str) -> "LateQuery":
        if layout == self.layout:
            return self
        if layout == QUERY_LAYOUT_NESTED:
            return LateQuery.from_vectors(self.as_vector_matrix(), text=self.text)
        if layout == QUERY_LAYOUT_FLAT_DIM128:
            return LateQuery.from_flat_values(
                self.as_flat_values(),
                vector_dim=self.vector_dim,
                text=self.text,
            )
        raise ValueError(f"unsupported query layout: {layout}")

    def with_text(self, text: object | None) -> "LateQuery":
        if self.layout == QUERY_LAYOUT_NESTED:
            return LateQuery.from_vectors(self.as_vector_matrix(), text=text)
        return LateQuery.from_flat_values(
            self.as_flat_values(),
            vector_dim=self.vector_dim,
            text=text,
        )

    def as_vector_matrix(self) -> np.ndarray:
        if self.layout == QUERY_LAYOUT_NESTED:
            assert self.token_vectors is not None
            return self.token_vectors

        assert self.token_values is not None
        return reshape_flat_values(self.token_values, self.vector_dim)

    def as_flat_values(self) -> np.ndarray:
        if self.layout == QUERY_LAYOUT_FLAT_DIM128:
            assert self.token_values is not None
            return self.token_values
        if self.vector_dim != FLAT_DIM128_VECTOR_DIM:
            raise ValueError("only dim128 queries can convert to flat_dim128")

        assert self.token_vectors is not None
        return flatten_vector_matrix(self.token_vectors)
