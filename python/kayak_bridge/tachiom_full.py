"""Composes TAC centroid HNSW candidate generation with residual-PQ rerank.

This module owns the paper-shaped Python reference pipeline. It does not own
native kernels or graph construction; those remain in the narrower HNSW and PQ
modules so each stage can be validated in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tachiom_hnsw import TachiomHnswConfig, TachiomTacHnswIndex
from .tachiom_index import TachiomTacIndex
from .tachiom_pq import TachiomResidualPqConfig, TachiomResidualPqIndex


@dataclass(frozen=True, slots=True)
class TachiomTacHnswResidualPqIndex:
    """TAC + centroid HNSW candidates + residual-PQ rerank."""

    hnsw_index: TachiomTacHnswIndex
    pq_index: TachiomResidualPqIndex

    @classmethod
    def from_tac_index(
        cls,
        index: TachiomTacIndex,
        *,
        hnsw_config: TachiomHnswConfig,
        pq_config: TachiomResidualPqConfig,
    ) -> "TachiomTacHnswResidualPqIndex":
        hnsw_index = TachiomTacHnswIndex.from_tac_index(
            index,
            config=hnsw_config,
        )
        pq_index = TachiomResidualPqIndex.from_tac_index(
            index,
            config=pq_config,
        )
        return cls(hnsw_index=hnsw_index, pq_index=pq_index)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self.pq_index.doc_ids

    @property
    def document_count(self) -> int:
        return self.pq_index.document_count

    @property
    def vector_dim(self) -> int:
        return self.pq_index.vector_dim

    @property
    def centroid_count(self) -> int:
        return self.pq_index.centroid_count

    @property
    def posting_count(self) -> int:
        return self.pq_index.posting_count

    @property
    def build_seconds(self) -> float:
        return self.hnsw_index.build_seconds + self.pq_index.build_seconds

    @property
    def index_kind(self) -> str:
        return "token_aware_hnsw_centroid_postings_residual_pq_rerank"

    @property
    def rerank_kind(self) -> str:
        return self.pq_index.rerank_kind

    @property
    def index_bytes(self) -> int:
        return self.pq_index.index_bytes + self.hnsw_index.graph.graph_bytes

    def candidate_positions_batch(
        self,
        queries: np.ndarray,
        *,
        final_k: int | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        return self.hnsw_index.candidate_positions_batch(
            queries,
            final_k=final_k,
        )

    def search_batch_positions(
        self,
        queries: np.ndarray,
        *,
        final_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        candidates = self.candidate_positions_batch(queries, final_k=final_k)
        return self.pq_index.rerank_candidate_positions_batch(
            queries,
            candidates,
            final_k=final_k,
        )
