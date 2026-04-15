"""Shared prepared exact-search types and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS: Final[tuple[str, ...]] = ("process",)


def _require_bool(name: str, value: bool) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _require_non_bool_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    return value


def _require_runtime_backend(name: str, value: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a non-empty string")
    if value not in SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS:
        supported = ", ".join(SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS)
        raise ValueError(
            f"{name} must be one of {supported}; "
            f"got unsupported backend {value!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ExactScoringOptions:
    """Explicit exact-scoring knobs forwarded to the Mojo execution kernel."""

    enable_parallel_scoring: bool = True
    enable_dim128_fast_path: bool = True
    enable_parallel_work_item_oversubscription: bool = True
    parallel_work_item_count_override: int = 0


@dataclass(frozen=True, slots=True)
class PreparedExactSearchRuntimeConfig:
    """Explicit local runtime policy for same-snapshot exact search."""

    execution_backend: str = "process"
    concurrency_lane_count: int = 1
    worker_count: int = 1
    max_batch_size: int = 32
    max_batch_wait_ms: int = 1
    scoring: ExactScoringOptions = field(default_factory=ExactScoringOptions)

    def __post_init__(self) -> None:
        _require_runtime_backend("execution_backend", self.execution_backend)
        concurrency_lane_count = _require_non_bool_int(
            "concurrency_lane_count",
            self.concurrency_lane_count,
        )
        worker_count = _require_non_bool_int("worker_count", self.worker_count)
        max_batch_size = _require_non_bool_int("max_batch_size", self.max_batch_size)
        max_batch_wait_ms = _require_non_bool_int(
            "max_batch_wait_ms",
            self.max_batch_wait_ms,
        )
        if concurrency_lane_count < 1:
            raise ValueError("concurrency_lane_count must be positive")
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be non-negative")
        _normalized_scoring(self.scoring)


@dataclass(frozen=True, slots=True)
class PreparedExactSearchRuntimeStats:
    """Snapshot of runtime work counters and timing totals."""

    submitted_request_count: int = 0
    completed_request_count: int = 0
    failed_request_count: int = 0
    executed_batch_count: int = 0
    last_batch_size: int = 0
    max_observed_batch_size: int = 0
    max_observed_queue_depth: int = 0
    total_queue_wait_seconds: float = 0.0
    total_batch_execution_seconds: float = 0.0

    @property
    def processed_request_count(self) -> int:
        return self.completed_request_count + self.failed_request_count

    @property
    def average_batch_size(self) -> float:
        if self.executed_batch_count == 0:
            return 0.0
        return self.processed_request_count / self.executed_batch_count

    @property
    def average_queue_wait_ms(self) -> float:
        if self.processed_request_count == 0:
            return 0.0
        return (self.total_queue_wait_seconds * 1000.0) / self.processed_request_count

    @property
    def average_batch_execution_ms(self) -> float:
        if self.executed_batch_count == 0:
            return 0.0
        return (
            self.total_batch_execution_seconds * 1000.0
        ) / self.executed_batch_count


def _normalized_scoring(
    scoring: ExactScoringOptions | None,
) -> ExactScoringOptions:
    resolved = ExactScoringOptions() if scoring is None else scoring
    if not isinstance(resolved, ExactScoringOptions):
        raise TypeError("scoring must be an ExactScoringOptions instance")
    _require_bool("enable_parallel_scoring", resolved.enable_parallel_scoring)
    _require_bool("enable_dim128_fast_path", resolved.enable_dim128_fast_path)
    _require_bool(
        "enable_parallel_work_item_oversubscription",
        resolved.enable_parallel_work_item_oversubscription,
    )
    _require_non_bool_int(
        "parallel_work_item_count_override",
        resolved.parallel_work_item_count_override,
    )
    return resolved


def _scoring_payload(scoring: ExactScoringOptions) -> dict[str, object]:
    return {
        "enable_parallel_scoring": scoring.enable_parallel_scoring,
        "enable_dim128_fast_path": scoring.enable_dim128_fast_path,
        "enable_parallel_work_item_oversubscription": (
            scoring.enable_parallel_work_item_oversubscription
        ),
        "parallel_work_item_count_override": (
            scoring.parallel_work_item_count_override
        ),
    }


def _runtime_config(
    config: PreparedExactSearchRuntimeConfig | None,
) -> PreparedExactSearchRuntimeConfig:
    resolved = (
        PreparedExactSearchRuntimeConfig()
        if config is None
        else config
    )
    if not isinstance(resolved, PreparedExactSearchRuntimeConfig):
        raise TypeError(
            "config must be a PreparedExactSearchRuntimeConfig instance"
        )
    return resolved
