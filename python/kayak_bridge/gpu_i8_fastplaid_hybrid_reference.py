"""CPU validation helpers for the hybrid GPU shortlist-rerank probe."""

from __future__ import annotations

from typing import Sequence


def position_rows(
    positions: Sequence[int],
    *,
    query_count: int,
    row_width: int,
) -> tuple[tuple[int, ...], ...]:
    expected = query_count * row_width
    if len(positions) != expected:
        raise ValueError("flat positions length must match query_count * row_width")
    return tuple(
        tuple(int(position) for position in positions[offset : offset + row_width])
        for offset in range(0, expected, row_width)
    )


def rank_candidate_positions_by_score(
    candidate_positions_by_query: Sequence[Sequence[int]],
    scores_by_query: Sequence[Sequence[float]],
    *,
    final_k: int,
) -> tuple[tuple[int, ...], ...]:
    expected: list[tuple[int, ...]] = []
    for positions, scores in zip(candidate_positions_by_query, scores_by_query):
        ranked_offsets = sorted(
            range(len(positions)),
            key=lambda offset: (-float(scores[offset]), offset),
        )
        expected.append(
            tuple(int(positions[offset]) for offset in ranked_offsets[:final_k])
        )
    return tuple(expected)


def topk_score_delta_max_abs(
    actual_scores: Sequence[float],
    reference_scores_by_query: Sequence[Sequence[float]],
    actual_positions: Sequence[int],
    candidate_positions_by_query: Sequence[Sequence[int]],
    *,
    top_k: int,
) -> float:
    delta_max = 0.0
    position_to_score_by_query = [
        {
            int(position): float(score)
            for position, score in zip(positions, scores)
        }
        for positions, scores in zip(
            candidate_positions_by_query,
            reference_scores_by_query,
        )
    ]
    for index, (actual_score, actual_position) in enumerate(
        zip(actual_scores, actual_positions)
    ):
        query_index = index // top_k
        reference_score = position_to_score_by_query[query_index][
            int(actual_position)
        ]
        delta_max = max(delta_max, abs(float(actual_score) - reference_score))
    return delta_max
