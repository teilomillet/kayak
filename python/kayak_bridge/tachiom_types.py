"""Shared TAC probe config and allocation report types."""

from __future__ import annotations

from dataclasses import dataclass

from .tachiom_arrays import _require_positive


@dataclass(frozen=True, slots=True)
class TachiomTacConfig:
    """Explicit knobs for token-aware centroid allocation and candidate search."""

    centroid_count: int = 128
    micro_token_threshold: int = 4
    small_token_threshold: int = 16
    active_token_floor: int = 1
    min_vectors_per_centroid: int = 4
    kmeans_iterations: int = 8
    centroids_per_query_vector: int = 8
    candidate_k: int = 64
    candidate_pruning_alpha: float | None = None

    def validate(self, *, final_k: int) -> None:
        _require_positive("centroid_count", self.centroid_count)
        _require_positive("micro_token_threshold", self.micro_token_threshold)
        _require_positive("small_token_threshold", self.small_token_threshold)
        _require_positive("active_token_floor", self.active_token_floor)
        _require_positive("min_vectors_per_centroid", self.min_vectors_per_centroid)
        _require_positive("kmeans_iterations", self.kmeans_iterations)
        _require_positive(
            "centroids_per_query_vector",
            self.centroids_per_query_vector,
        )
        _require_positive("candidate_k", self.candidate_k)
        if self.candidate_pruning_alpha is not None and not (
            0.0 < self.candidate_pruning_alpha < 1.0
        ):
            raise ValueError("candidate_pruning_alpha must be between 0 and 1")
        if self.small_token_threshold <= self.micro_token_threshold:
            raise ValueError(
                "small_token_threshold must be greater than micro_token_threshold"
            )
        if self.candidate_k < final_k:
            raise ValueError("candidate_k must be greater than or equal to final_k")


@dataclass(frozen=True, slots=True)
class TacAllocationSummary:
    """Compact report for the token-aware centroid allocation gate."""

    requested_centroid_count: int
    effective_centroid_count: int
    token_type_count: int
    micro_token_type_count: int
    small_token_type_count: int
    active_token_type_count: int
    min_centroids_per_token: int
    max_centroids_per_token: int
    mean_centroids_per_token: float

    def to_json_ready(self) -> dict[str, int | float]:
        return {
            "requested_centroid_count": self.requested_centroid_count,
            "effective_centroid_count": self.effective_centroid_count,
            "token_type_count": self.token_type_count,
            "micro_token_type_count": self.micro_token_type_count,
            "small_token_type_count": self.small_token_type_count,
            "active_token_type_count": self.active_token_type_count,
            "min_centroids_per_token": self.min_centroids_per_token,
            "max_centroids_per_token": self.max_centroids_per_token,
            "mean_centroids_per_token": self.mean_centroids_per_token,
        }
