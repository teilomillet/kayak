"""Owns explicit score-fusion helpers for aligned late-interaction results.

This module owns:
- validation that fused score batches align on queries and document ids
- weighted score summation over exact late-interaction score vectors

This module does not own:
- query encoding
- search execution
- benchmark-specific policy choices
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .late_scores import LateScores


def _validate_aligned_score_batches(
    score_batches: Sequence[Sequence[LateScores]],
) -> None:
    if not score_batches:
        raise ValueError("score fusion requires at least one score batch")

    query_count = len(score_batches[0])
    if query_count == 0:
        raise ValueError("score fusion requires at least one query")

    for score_batch in score_batches[1:]:
        if len(score_batch) != query_count:
            raise ValueError("score fusion batches must have matching query counts")

    reference_doc_ids = tuple(scores.doc_ids for scores in score_batches[0])
    for score_batch in score_batches[1:]:
        for query_index, scores in enumerate(score_batch):
            if scores.doc_ids != reference_doc_ids[query_index]:
                raise ValueError("score fusion batches must align on document ids")


def weighted_sum_score_batches(
    weighted_score_batches: Sequence[tuple[float, Sequence[LateScores]]],
    *,
    backend: str = "weighted_sum_fusion",
) -> tuple[LateScores, ...]:
    if not weighted_score_batches:
        raise ValueError("weighted score fusion requires at least one input")

    weights = tuple(float(weight) for weight, _ in weighted_score_batches)
    score_batches = tuple(score_batch for _, score_batch in weighted_score_batches)
    _validate_aligned_score_batches(score_batches)

    fused_scores: list[LateScores] = []
    for query_index in range(len(score_batches[0])):
        merged_values = np.zeros_like(score_batches[0][query_index].values)
        for weight, score_batch in zip(weights, score_batches, strict=True):
            merged_values = merged_values + (
                np.float32(weight) * score_batch[query_index].values
            )

        fused_scores.append(
            LateScores.from_values(
                backend,
                score_batches[0][query_index].doc_ids,
                merged_values,
            )
        )

    return tuple(fused_scores)
