"""Compatibility facade for the prepared exact-search Python surface."""

from __future__ import annotations

from .prepared_exact_runtime import (
    PreparedExactSearchRuntime,
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchRuntimeStats,
    PreparedExactSearchScheduler,
    PreparedExactSearchSchedulerConfig,
    PreparedExactSearchSchedulerOverloadedError,
    PreparedExactSearchSchedulerStats,
    prepare_exact_search_runtime,
    prepare_exact_search_scheduler,
)
from .prepared_exact_session import (
    PreparedExactSearchSession,
    prepare_exact_search_session,
)
from .prepared_exact_types import (
    ExactScoringOptions,
    PreparedExactSearchRuntimeOverloadedError,
    SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS,
)

__all__ = [
    "ExactScoringOptions",
    "PreparedExactSearchRuntime",
    "PreparedExactSearchRuntimeConfig",
    "PreparedExactSearchRuntimeOverloadedError",
    "PreparedExactSearchRuntimeStats",
    "PreparedExactSearchScheduler",
    "PreparedExactSearchSchedulerConfig",
    "PreparedExactSearchSchedulerOverloadedError",
    "PreparedExactSearchSchedulerStats",
    "PreparedExactSearchSession",
    "SUPPORTED_PREPARED_EXACT_RUNTIME_BACKENDS",
    "prepare_exact_search_runtime",
    "prepare_exact_search_scheduler",
    "prepare_exact_search_session",
]
