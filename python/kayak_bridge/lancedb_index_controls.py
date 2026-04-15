"""Owns explicit LanceDB IVF_PQ build and query controls for benchmarks.

This module does not talk to LanceDB directly. It only validates and
normalizes the benchmark-side knobs so indexed runs can record the exact
configuration that produced a result.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{name} must be positive when provided")
    return int(value)


@dataclass(frozen=True, slots=True)
class LanceDbIndexBuildControls:
    num_partitions: int | None = None
    num_sub_vectors: int | None = None
    target_partition_size: int | None = None

    def validated(self) -> "LanceDbIndexBuildControls":
        return LanceDbIndexBuildControls(
            num_partitions=_require_positive_int(
                "index_num_partitions", self.num_partitions
            ),
            num_sub_vectors=_require_positive_int(
                "index_num_sub_vectors", self.num_sub_vectors
            ),
            target_partition_size=_require_positive_int(
                "index_target_partition_size", self.target_partition_size
            ),
        )

    def has_overrides(self) -> bool:
        return any(
            value is not None
            for value in (
                self.num_partitions,
                self.num_sub_vectors,
                self.target_partition_size,
            )
        )

    def create_index_kwargs(self) -> dict[str, int]:
        kwargs: dict[str, int] = {}
        if self.num_partitions is not None:
            kwargs["num_partitions"] = self.num_partitions
        if self.num_sub_vectors is not None:
            kwargs["num_sub_vectors"] = self.num_sub_vectors
        if self.target_partition_size is not None:
            kwargs["target_partition_size"] = self.target_partition_size
        return kwargs


@dataclass(frozen=True, slots=True)
class LanceDbIndexedQueryControls:
    nprobes: int | None = None
    refine_factor: int | None = None

    def validated(self) -> "LanceDbIndexedQueryControls":
        return LanceDbIndexedQueryControls(
            nprobes=_require_positive_int("indexed_nprobes", self.nprobes),
            refine_factor=_require_positive_int(
                "indexed_refine_factor", self.refine_factor
            ),
        )

    def has_overrides(self) -> bool:
        return any(
            value is not None
            for value in (
                self.nprobes,
                self.refine_factor,
            )
        )


def validate_index_controls(
    *,
    build_index: bool,
    index_build_controls: LanceDbIndexBuildControls | None,
    indexed_query_controls: LanceDbIndexedQueryControls | None,
) -> tuple[LanceDbIndexBuildControls, LanceDbIndexedQueryControls]:
    normalized_build = (
        LanceDbIndexBuildControls()
        if index_build_controls is None
        else index_build_controls
    ).validated()
    normalized_query = (
        LanceDbIndexedQueryControls()
        if indexed_query_controls is None
        else indexed_query_controls
    ).validated()

    if not build_index and (
        normalized_build.has_overrides() or normalized_query.has_overrides()
    ):
        raise ValueError(
            "indexed LanceDB controls require build_index=True; "
            "scan runs should not silently ignore indexed-only knobs"
        )

    return normalized_build, normalized_query
