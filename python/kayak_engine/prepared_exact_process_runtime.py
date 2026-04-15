"""Process-backed implementation of the prepared exact-search runtime."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import threading
import time
from typing import Any

from .prepared_exact_session import (
    _normalized_request_for_identity,
    prepare_exact_search_session,
)
from .prepared_exact_types import (
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchRuntimeStats,
    _require_bool,
    _runtime_config,
)


@dataclass(frozen=True, slots=True)
class _RuntimeRequestEnvelope:
    request_id: int
    request: dict[str, Any]
    submitted_at: float


@dataclass(frozen=True, slots=True)
class _RuntimeReadyEnvelope:
    worker_index: int


@dataclass(frozen=True, slots=True)
class _RuntimeSuccessEnvelope:
    request_id: int
    response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _RuntimeErrorEnvelope:
    request_id: int
    error_message: str


@dataclass(frozen=True, slots=True)
class _RuntimeBatchMetricsEnvelope:
    batch_size: int
    queue_wait_seconds: float
    batch_execution_seconds: float
    failed_request_count: int


@dataclass(frozen=True, slots=True)
class _RuntimeStopEnvelope:
    worker_index: int


@dataclass(frozen=True, slots=True)
class _RuntimeFatalEnvelope:
    worker_index: int
    error_message: str


def _collect_process_batch(
    first_request: _RuntimeRequestEnvelope,
    request_queue: Any,
    *,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> tuple[list[_RuntimeRequestEnvelope], bool]:
    batch = [first_request]
    stop_after_batch = False
    deadline = time.perf_counter() + (max_batch_wait_ms / 1000.0)

    while len(batch) < max_batch_size:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        try:
            item = request_queue.get(timeout=remaining)
        except Empty:
            break
        if isinstance(item, _RuntimeStopEnvelope):
            stop_after_batch = True
            break
        batch.append(item)

    return batch, stop_after_batch


def _prepared_exact_search_runtime_worker(
    *,
    worker_index: int,
    request_queue: Any,
    response_queue: Any,
    service_root: str,
    collection_id: str,
    tenant_id: str,
    namespace_id: str,
    snapshot_id: str,
    load_text_corpus: bool,
    config: PreparedExactSearchRuntimeConfig,
) -> None:
    try:
        session = prepare_exact_search_session(
            service_root=service_root,
            collection_id=collection_id,
            tenant_id=tenant_id,
            namespace_id=namespace_id,
            snapshot_id=snapshot_id,
            load_text_corpus=load_text_corpus,
        )
    except Exception as exc:
        response_queue.put(
            _RuntimeFatalEnvelope(
                worker_index=worker_index,
                error_message=f"failed to prepare runtime worker session: {exc}",
            )
        )
        return
    response_queue.put(_RuntimeReadyEnvelope(worker_index=worker_index))

    while True:
        item = request_queue.get()
        if isinstance(item, _RuntimeStopEnvelope):
            response_queue.put(_RuntimeStopEnvelope(worker_index=worker_index))
            return

        batch, stop_after_batch = _collect_process_batch(
            item,
            request_queue,
            max_batch_size=config.max_batch_size,
            max_batch_wait_ms=config.max_batch_wait_ms,
        )
        batch_start = time.perf_counter()
        total_queue_wait_seconds = sum(
            batch_start - entry.submitted_at for entry in batch
        )
        try:
            responses = session._search_batch_normalized(
                [entry.request for entry in batch],
                worker_count=config.worker_count,
                scoring=config.scoring,
            )
        except Exception as exc:
            batch_execution_seconds = time.perf_counter() - batch_start
            message = f"{type(exc).__name__}: {exc}"
            for entry in batch:
                response_queue.put(
                    _RuntimeErrorEnvelope(
                        request_id=entry.request_id,
                        error_message=message,
                    )
                )
            response_queue.put(
                _RuntimeBatchMetricsEnvelope(
                    batch_size=len(batch),
                    queue_wait_seconds=total_queue_wait_seconds,
                    batch_execution_seconds=batch_execution_seconds,
                    failed_request_count=len(batch),
                )
            )
            if stop_after_batch:
                response_queue.put(_RuntimeStopEnvelope(worker_index=worker_index))
                return
            continue

        batch_execution_seconds = time.perf_counter() - batch_start
        for entry, response in zip(batch, responses, strict=True):
            response_queue.put(
                _RuntimeSuccessEnvelope(
                    request_id=entry.request_id,
                    response=response,
                )
            )
        response_queue.put(
            _RuntimeBatchMetricsEnvelope(
                batch_size=len(batch),
                queue_wait_seconds=total_queue_wait_seconds,
                batch_execution_seconds=batch_execution_seconds,
                failed_request_count=0,
            )
        )
        if stop_after_batch:
            response_queue.put(_RuntimeStopEnvelope(worker_index=worker_index))
            return


class PreparedExactProcessRuntime:
    """Explicit process-backed batching for one prepared exact-search snapshot."""

    __slots__ = (
        "service_root",
        "collection_id",
        "tenant_id",
        "namespace_id",
        "snapshot_id",
        "load_text_corpus",
        "_config",
        "_context",
        "_request_queue",
        "_response_queue",
        "_workers",
        "_listener",
        "_lifecycle_lock",
        "_stats_lock",
        "_stats",
        "_closed",
        "_fatal_exception",
        "_ready_event",
        "_ready_worker_count",
        "_stopped_worker_count",
        "_next_request_id",
        "_pending_futures",
        "_pending_request_count",
    )

    def __init__(
        self,
        *,
        service_root: str | Path,
        collection_id: str,
        tenant_id: str,
        namespace_id: str,
        snapshot_id: str,
        load_text_corpus: bool = False,
        config: PreparedExactSearchRuntimeConfig | None = None,
    ) -> None:
        self.service_root = Path(service_root)
        self.collection_id = collection_id
        self.tenant_id = tenant_id
        self.namespace_id = namespace_id
        self.snapshot_id = snapshot_id
        self.load_text_corpus = _require_bool("load_text_corpus", load_text_corpus)
        self._config = _runtime_config(config)
        self._context = mp.get_context("spawn")
        self._request_queue = self._context.Queue()
        self._response_queue = self._context.Queue()
        self._lifecycle_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = PreparedExactSearchRuntimeStats()
        self._closed = False
        self._fatal_exception: BaseException | None = None
        self._ready_event = threading.Event()
        self._ready_worker_count = 0
        self._stopped_worker_count = 0
        self._next_request_id = 0
        self._pending_futures: dict[int, Future[dict[str, Any]]] = {}
        self._pending_request_count = 0
        self._workers = [
            self._context.Process(
                target=_prepared_exact_search_runtime_worker,
                kwargs={
                    "worker_index": worker_index,
                    "request_queue": self._request_queue,
                    "response_queue": self._response_queue,
                    "service_root": str(self.service_root),
                    "collection_id": self.collection_id,
                    "tenant_id": self.tenant_id,
                    "namespace_id": self.namespace_id,
                    "snapshot_id": self.snapshot_id,
                    "load_text_corpus": self.load_text_corpus,
                    "config": self._config,
                },
                name=(
                    "PreparedExactProcessRuntimeWorker"
                    f"[{worker_index}:{self.collection_id}/{self.snapshot_id}]"
                ),
                daemon=True,
            )
            for worker_index in range(self._config.concurrency_lane_count)
        ]
        self._listener = threading.Thread(
            target=self._listener_loop,
            name=(
                "PreparedExactProcessRuntimeListener"
                f"[{self.collection_id}/{self.snapshot_id}]"
            ),
            daemon=True,
        )
        for worker in self._workers:
            worker.start()
        self._listener.start()

    @property
    def config(self) -> PreparedExactSearchRuntimeConfig:
        return self._config

    def stats(self) -> PreparedExactSearchRuntimeStats:
        with self._stats_lock:
            return PreparedExactSearchRuntimeStats(
                submitted_request_count=self._stats.submitted_request_count,
                completed_request_count=self._stats.completed_request_count,
                failed_request_count=self._stats.failed_request_count,
                executed_batch_count=self._stats.executed_batch_count,
                last_batch_size=self._stats.last_batch_size,
                max_observed_batch_size=self._stats.max_observed_batch_size,
                max_observed_queue_depth=self._stats.max_observed_queue_depth,
                total_queue_wait_seconds=self._stats.total_queue_wait_seconds,
                total_batch_execution_seconds=self._stats.total_batch_execution_seconds,
            )

    def wait_until_ready(self, timeout: float | None = None) -> None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        deadline = None if timeout is None else (time.perf_counter() + timeout)
        while True:
            with self._lifecycle_lock:
                if self._fatal_exception is not None:
                    raise RuntimeError(
                        "prepared exact search runtime failed during startup"
                    ) from self._fatal_exception
                if self._ready_event.is_set():
                    return
                if self._closed:
                    raise RuntimeError(
                        "prepared exact search runtime closed before startup completed"
                    )

            wait_timeout = 0.05
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError(
                        "prepared exact search runtime did not become ready in time"
                    )
                wait_timeout = min(wait_timeout, remaining)

            if self._ready_event.wait(timeout=wait_timeout):
                return

    def worker_pids(self) -> tuple[int, ...]:
        return tuple(worker.pid for worker in self._workers if worker.pid is not None)

    def submit(self, payload: dict[str, Any]) -> Future[dict[str, Any]]:
        request = _normalized_request_for_identity(
            payload,
            collection_id=self.collection_id,
            tenant_id=self.tenant_id,
            namespace_id=self.namespace_id,
            snapshot_id=self.snapshot_id,
        )
        future: Future[dict[str, Any]] = Future()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("prepared exact search runtime is closed")
            if self._fatal_exception is not None:
                raise RuntimeError(
                    "prepared exact search runtime failed"
                ) from self._fatal_exception
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending_futures[request_id] = future
            self._pending_request_count += 1
            pending_request_count = self._pending_request_count
        self._request_queue.put(
            _RuntimeRequestEnvelope(
                request_id=request_id,
                request=request,
                submitted_at=time.perf_counter(),
            )
        )
        self._record_submission(pending_request_count)
        return future

    def search(
        self,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.submit(payload).result(timeout=timeout)

    def search_batch(
        self,
        requests: list[dict[str, Any]],
        *,
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        if not isinstance(requests, list):
            raise TypeError("requests must be a list of exact-search payloads")
        if len(requests) == 0:
            return []

        normalized_requests = [
            _normalized_request_for_identity(
                payload,
                collection_id=self.collection_id,
                tenant_id=self.tenant_id,
                namespace_id=self.namespace_id,
                snapshot_id=self.snapshot_id,
            )
            for payload in requests
        ]
        futures = [self.submit(request) for request in normalized_requests]
        if timeout is None:
            return [future.result() for future in futures]

        deadline = time.perf_counter() + timeout
        responses: list[dict[str, Any]] = []
        for future in futures:
            remaining = deadline - time.perf_counter()
            if remaining < 0:
                raise TimeoutError("prepared exact search runtime batch timed out")
            responses.append(future.result(timeout=remaining))
        return responses

    def close(self) -> None:
        with self._lifecycle_lock:
            if not self._closed:
                self._closed = True
                self._broadcast_stop_envelopes_locked()
        for worker in self._workers:
            worker.join(timeout=15.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=15.0)
        self._listener.join(timeout=15.0)
        if self._listener.is_alive():
            raise RuntimeError("prepared exact search runtime listener did not stop")

    def __enter__(self) -> PreparedExactProcessRuntime:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()

    def _record_submission(self, pending_request_count: int) -> None:
        with self._stats_lock:
            self._stats = PreparedExactSearchRuntimeStats(
                submitted_request_count=self._stats.submitted_request_count + 1,
                completed_request_count=self._stats.completed_request_count,
                failed_request_count=self._stats.failed_request_count,
                executed_batch_count=self._stats.executed_batch_count,
                last_batch_size=self._stats.last_batch_size,
                max_observed_batch_size=self._stats.max_observed_batch_size,
                max_observed_queue_depth=max(
                    self._stats.max_observed_queue_depth,
                    pending_request_count,
                ),
                total_queue_wait_seconds=self._stats.total_queue_wait_seconds,
                total_batch_execution_seconds=self._stats.total_batch_execution_seconds,
            )

    def _record_batch(
        self,
        *,
        batch_size: int,
        queue_wait_seconds: float,
        batch_execution_seconds: float,
        failed_request_count: int,
    ) -> None:
        completed_request_count = batch_size - failed_request_count
        with self._stats_lock:
            self._stats = PreparedExactSearchRuntimeStats(
                submitted_request_count=self._stats.submitted_request_count,
                completed_request_count=(
                    self._stats.completed_request_count + completed_request_count
                ),
                failed_request_count=(
                    self._stats.failed_request_count + failed_request_count
                ),
                executed_batch_count=self._stats.executed_batch_count + 1,
                last_batch_size=batch_size,
                max_observed_batch_size=max(
                    self._stats.max_observed_batch_size,
                    batch_size,
                ),
                max_observed_queue_depth=self._stats.max_observed_queue_depth,
                total_queue_wait_seconds=(
                    self._stats.total_queue_wait_seconds + queue_wait_seconds
                ),
                total_batch_execution_seconds=(
                    self._stats.total_batch_execution_seconds
                    + batch_execution_seconds
                ),
            )

    def _resolve_pending_future(
        self, request_id: int
    ) -> Future[dict[str, Any]] | None:
        with self._lifecycle_lock:
            future = self._pending_futures.pop(request_id, None)
            if future is not None:
                self._pending_request_count -= 1
            return future

    def _set_fatal_exception(self, exc: BaseException) -> None:
        with self._lifecycle_lock:
            broadcast_stop = not self._closed
            if self._fatal_exception is None:
                self._fatal_exception = exc
            self._closed = True
            if broadcast_stop:
                self._broadcast_stop_envelopes_locked()

    def _fail_pending_futures(self, exc: BaseException) -> None:
        with self._lifecycle_lock:
            pending_futures = list(self._pending_futures.values())
            self._pending_futures.clear()
            self._pending_request_count = 0
        for future in pending_futures:
            future.set_exception(exc)

    def _broadcast_stop_envelopes_locked(self) -> None:
        for _ in self._workers:
            self._request_queue.put(_RuntimeStopEnvelope(worker_index=-1))

    def _unexpected_worker_exit_exception(self) -> RuntimeError | None:
        for worker in self._workers:
            if worker.is_alive():
                continue
            if worker.exitcode is None:
                continue
            if self._closed and worker.exitcode == 0:
                continue
            return RuntimeError(
                "prepared exact search runtime worker exited unexpectedly"
                f" with code {worker.exitcode}: {worker.name}"
            )
        return None

    def _listener_loop(self) -> None:
        while True:
            try:
                envelope = self._response_queue.get(timeout=0.1)
            except Empty:
                unexpected_worker_exit = self._unexpected_worker_exit_exception()
                if unexpected_worker_exit is not None:
                    self._set_fatal_exception(unexpected_worker_exit)
                    self._fail_pending_futures(unexpected_worker_exit)
                    return
                if any(worker.is_alive() for worker in self._workers):
                    continue
                if self._closed:
                    return
                exc = RuntimeError(
                    "prepared exact search runtime workers exited unexpectedly"
                )
                self._set_fatal_exception(exc)
                self._fail_pending_futures(exc)
                return

            if isinstance(envelope, _RuntimeSuccessEnvelope):
                future = self._resolve_pending_future(envelope.request_id)
                if future is not None:
                    future.set_result(envelope.response)
                continue

            if isinstance(envelope, _RuntimeReadyEnvelope):
                self._ready_worker_count += 1
                if self._ready_worker_count >= len(self._workers):
                    self._ready_event.set()
                continue

            if isinstance(envelope, _RuntimeErrorEnvelope):
                future = self._resolve_pending_future(envelope.request_id)
                if future is not None:
                    future.set_exception(RuntimeError(envelope.error_message))
                continue

            if isinstance(envelope, _RuntimeBatchMetricsEnvelope):
                self._record_batch(
                    batch_size=envelope.batch_size,
                    queue_wait_seconds=envelope.queue_wait_seconds,
                    batch_execution_seconds=envelope.batch_execution_seconds,
                    failed_request_count=envelope.failed_request_count,
                )
                continue

            if isinstance(envelope, _RuntimeFatalEnvelope):
                exc = RuntimeError(
                    f"{envelope.error_message} [worker_index={envelope.worker_index}]"
                )
                self._set_fatal_exception(exc)
                self._fail_pending_futures(exc)
                return

            if isinstance(envelope, _RuntimeStopEnvelope):
                self._stopped_worker_count += 1
                if self._stopped_worker_count >= len(self._workers):
                    return
