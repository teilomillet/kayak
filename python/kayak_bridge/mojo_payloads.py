"""Builds explicit Python payloads for Mojo-backed late-interaction calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MojoIndexPayload:
    doc_ids: list[str]
    doc_offsets: list[int]
    packed_vectors: list[list[float]]
    flat_token_values: list[float] | None


def index_payload(index: "LateIndex") -> MojoIndexPayload:
    flat_token_values: list[float] | None = None
    if index.layout == "hybrid_flat_dim128":
        flat_token_values = index.as_flat_token_values().tolist()

    return MojoIndexPayload(
        doc_ids=list(index.doc_ids),
        doc_offsets=[int(offset) for offset in index.doc_offsets],
        packed_vectors=index.as_packed_token_matrix().tolist(),
        flat_token_values=flat_token_values,
    )


def query_payload(query: "LateQuery") -> list[list[float]] | list[float]:
    if query.layout == "flat_dim128":
        return query.as_flat_values().tolist()
    return query.as_vector_matrix().tolist()
