"""Candidate-window ranking and paper-style pruning for TAC gather scores."""

from __future__ import annotations

import numpy as np

from .dtypes import VECTOR_DTYPE
from .tachiom_arrays import _top_positions


def ranked_candidate_positions_from_scores(
    scores: np.ndarray,
    *,
    candidate_k: int,
    final_k: int | None,
    candidate_pruning_alpha: float | None,
) -> tuple[int, ...]:
    """Rank first-stage scores and optionally apply Tachiom's CP rule.

    Candidate Pruning is defined on the sorted first-stage candidate list. The
    threshold uses the score at the final-rank cutoff. The first ``final_k``
    candidates are always retained so negative synthetic scores cannot make the
    pruning rule return fewer documents than the requested final ranking.
    """

    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    candidate_count = min(candidate_k, int(scores.shape[0]))
    positions = tuple(
        int(position) for position in _top_positions(scores, candidate_count)
    )
    if candidate_pruning_alpha is None:
        return positions
    if final_k is None:
        raise ValueError("final_k is required when candidate pruning is enabled")
    if final_k <= 0:
        raise ValueError("final_k must be positive")
    required_count = min(final_k, len(positions))
    if required_count == 0:
        return ()

    cutoff_score = VECTOR_DTYPE(scores[positions[required_count - 1]])
    threshold = VECTOR_DTYPE((1.0 - candidate_pruning_alpha) * float(cutoff_score))
    kept: list[int] = []
    for offset, position in enumerate(positions):
        if offset >= required_count and scores[position] < threshold:
            break
        kept.append(position)
    return tuple(kept)
