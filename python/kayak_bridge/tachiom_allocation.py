"""Token-aware centroid budget allocation for the TAC probe."""

from __future__ import annotations

import numpy as np

from .tachiom_arrays import _as_flat_token_ids, _as_token_matrix
from .tachiom_types import TachiomTacConfig, TacAllocationSummary


def allocate_tac_centroid_counts(
    *,
    token_values: np.ndarray,
    token_ids: np.ndarray,
    config: TachiomTacConfig,
) -> dict[int, int]:
    """Allocate a global centroid budget across token ids.

    This mirrors Tachiom's first gate: tail handling, damped scoring, bounding,
    and budget reconciliation. The implementation is deterministic so repeated
    benchmark runs can separate algorithm effects from initialization noise.
    """

    values = _as_token_matrix(token_values)
    ids = _as_flat_token_ids(token_ids, expected_length=int(values.shape[0]))
    effective_budget = min(config.centroid_count, int(values.shape[0]))
    unique_ids, counts = np.unique(ids, return_counts=True)
    groups = {int(token_id): np.flatnonzero(ids == token_id) for token_id in unique_ids}
    allocation: dict[int, int] = {}
    active_token_ids: list[int] = []
    weights: dict[int, float] = {}
    caps: dict[int, int] = {}

    for token_id, count in zip(unique_ids, counts, strict=True):
        token = int(token_id)
        token_count = int(count)
        if token_count < config.micro_token_threshold:
            allocation[token] = 1
            continue
        if token_count < config.small_token_threshold:
            allocation[token] = min(2, token_count)
            continue

        active_token_ids.append(token)
        group_values = values[groups[token]]
        center = group_values.mean(axis=0)
        spread = float(np.mean(np.sum((group_values - center) ** 2, axis=1)))
        weights[token] = float(np.sqrt(max(token_count * spread, 1e-12)))
        caps[token] = max(
            config.active_token_floor,
            min(token_count, token_count // config.min_vectors_per_centroid),
        )

    fixed_total = sum(allocation.values())
    token_counts = {
        int(token_id): int(count)
        for token_id, count in zip(unique_ids, counts, strict=True)
    }
    if fixed_total >= effective_budget:
        for token in active_token_ids:
            allocation[token] = min(config.active_token_floor, caps[token])
        return _trim_allocation_to_budget(
            allocation,
            target=effective_budget,
            token_counts=token_counts,
        )

    remaining_budget = effective_budget - fixed_total
    if not active_token_ids:
        return _reconcile_allocations(
            allocation,
            target=effective_budget,
            weights={token: float(len(groups[token])) for token in allocation},
            caps={token: len(groups[token]) for token in allocation},
            floors={token: 1 for token in allocation},
        )

    weight_sum = sum(weights[token] for token in active_token_ids)
    if remaining_budget < len(active_token_ids) * config.active_token_floor:
        for token in sorted(
            active_token_ids,
            key=lambda item: (weights.get(item, 0.0), -item),
            reverse=True,
        )[:remaining_budget]:
            allocation[token] = 1
        return allocation

    fractional: dict[int, float] = {}
    floors: dict[int, int] = {}
    for token in active_token_ids:
        raw = (
            remaining_budget * weights[token] / weight_sum
            if weight_sum > 0.0
            else remaining_budget / len(active_token_ids)
        )
        floor = min(caps[token], max(config.active_token_floor, int(np.floor(raw))))
        allocation[token] = floor
        floors[token] = min(config.active_token_floor, caps[token])
        fractional[token] = raw - np.floor(raw)

    return _reconcile_allocations(
        allocation,
        target=effective_budget,
        weights={token: fractional.get(token, weights.get(token, 0.0)) for token in allocation},
        caps={token: caps.get(token, len(groups[token])) for token in allocation},
        floors={token: floors.get(token, 1) for token in allocation},
    )


def _allocation_summary(
    *,
    allocation: dict[int, int],
    token_ids: np.ndarray,
    requested_centroid_count: int,
    config: TachiomTacConfig,
) -> TacAllocationSummary:
    counts = np.asarray(tuple(allocation.values()), dtype=np.int64)
    frequencies = {
        int(token_id): int(count)
        for token_id, count in zip(*np.unique(token_ids, return_counts=True), strict=True)
    }
    micro_count = sum(
        1 for token in allocation if frequencies[token] < config.micro_token_threshold
    )
    small_count = sum(
        1
        for token in allocation
        if config.micro_token_threshold <= frequencies[token] < config.small_token_threshold
    )
    active_count = len(allocation) - micro_count - small_count
    return TacAllocationSummary(
        requested_centroid_count=requested_centroid_count,
        effective_centroid_count=int(counts.sum()),
        token_type_count=len(allocation),
        micro_token_type_count=micro_count,
        small_token_type_count=small_count,
        active_token_type_count=active_count,
        min_centroids_per_token=int(counts.min()),
        max_centroids_per_token=int(counts.max()),
        mean_centroids_per_token=float(counts.mean()),
    )


def _reconcile_allocations(
    allocation: dict[int, int],
    *,
    target: int,
    weights: dict[int, float],
    caps: dict[int, int],
    floors: dict[int, int],
) -> dict[int, int]:
    current = sum(allocation.values())
    while current < target:
        candidates = [
            token for token, count in allocation.items() if count < caps.get(token, count)
        ]
        if not candidates:
            break
        token = max(candidates, key=lambda item: (weights.get(item, 0.0), -item))
        allocation[token] += 1
        current += 1

    while current > target:
        candidates = [
            token for token, count in allocation.items() if count > floors.get(token, 1)
        ]
        if not candidates:
            break
        token = min(candidates, key=lambda item: (weights.get(item, 0.0), item))
        allocation[token] -= 1
        current -= 1
    return {token: count for token, count in allocation.items() if count > 0}


def _trim_allocation_to_budget(
    allocation: dict[int, int],
    *,
    target: int,
    token_counts: dict[int, int],
) -> dict[int, int]:
    if target <= 0:
        raise ValueError("target must be positive")
    ordered = sorted(
        allocation,
        key=lambda token: (token_counts[token], -token),
        reverse=True,
    )
    trimmed: dict[int, int] = {}
    remaining = target
    for token in ordered:
        if remaining <= 0:
            break
        count = min(allocation[token], remaining)
        trimmed[token] = count
        remaining -= count
    return trimmed
