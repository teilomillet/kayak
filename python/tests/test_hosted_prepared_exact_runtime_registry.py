from __future__ import annotations

from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from kayak_engine.hosted_prepared_exact_runtime_registry import (
    HostedPreparedExactRuntimeRegistry,
)
from kayak_engine.prepared_exact_types import (
    PreparedExactSearchRuntimeConfig,
    PreparedExactSearchRuntimeStats,
)


class _FakeRuntime:
    def __init__(
        self,
        *,
        snapshot_id: str,
        ready_event: threading.Event,
        created_event: threading.Event,
    ) -> None:
        self.snapshot_id = snapshot_id
        self._ready_event = ready_event
        self.closed = False
        created_event.set()

    def wait_until_ready(self, timeout: float | None = None) -> None:
        if not self._ready_event.wait(timeout=timeout):
            raise TimeoutError(f"fake runtime {self.snapshot_id} did not become ready")

    def search(self, payload: dict[str, object]) -> dict[str, object]:
        return {
            "snapshot_id": self.snapshot_id,
            "payload": payload,
        }

    def search_batch(
        self,
        requests: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        return [self.search(payload) for payload in requests]

    def stats(self) -> PreparedExactSearchRuntimeStats:
        return PreparedExactSearchRuntimeStats()

    def close(self) -> None:
        self.closed = True


class HostedPreparedExactRuntimeRegistryTests(unittest.TestCase):
    def _prepare_config(self) -> PreparedExactSearchRuntimeConfig:
        return PreparedExactSearchRuntimeConfig(
            concurrency_lane_count=1,
            worker_count=1,
            max_batch_size=8,
            max_batch_wait_ms=1,
        )

    def test_prepare_runtime_does_not_block_existing_runtime_search(self) -> None:
        registry = HostedPreparedExactRuntimeRegistry(service_root=Path("/tmp/service-root"))
        ready_events = {
            "snapshot-0001": threading.Event(),
            "snapshot-0002": threading.Event(),
        }
        created_events = {
            "snapshot-0001": threading.Event(),
            "snapshot-0002": threading.Event(),
        }
        ready_events["snapshot-0001"].set()

        def fake_prepare_runtime(**kwargs: object) -> _FakeRuntime:
            snapshot_id = kwargs["snapshot_id"]
            assert isinstance(snapshot_id, str)
            return _FakeRuntime(
                snapshot_id=snapshot_id,
                ready_event=ready_events[snapshot_id],
                created_event=created_events[snapshot_id],
            )

        with patch(
            "kayak_engine.hosted_prepared_exact_runtime_registry.prepare_exact_search_runtime",
            side_effect=fake_prepare_runtime,
        ):
            runtime, reused = registry.prepare_runtime(
                collection_id="news",
                tenant_id="tenant-a",
                namespace_id="search",
                snapshot_id="snapshot-0001",
                load_text_corpus=False,
                config=self._prepare_config(),
            )
            self.assertFalse(reused)
            runtime_id = runtime["runtime_id"]

            prepare_errors: list[BaseException] = []
            second_prepare_done = threading.Event()

            def prepare_second_runtime() -> None:
                try:
                    registry.prepare_runtime(
                        collection_id="news",
                        tenant_id="tenant-a",
                        namespace_id="search",
                        snapshot_id="snapshot-0002",
                        load_text_corpus=False,
                        config=self._prepare_config(),
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    prepare_errors.append(exc)
                finally:
                    second_prepare_done.set()

            prepare_thread = threading.Thread(
                target=prepare_second_runtime,
                daemon=True,
            )
            prepare_thread.start()
            self.assertTrue(created_events["snapshot-0002"].wait(timeout=5.0))

            search_result: dict[str, object] = {}
            search_errors: list[BaseException] = []
            search_done = threading.Event()

            def search_existing_runtime() -> None:
                try:
                    search_result.update(
                        registry.search(runtime_id, {"query_model_name": "colbertv2"})
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    search_errors.append(exc)
                finally:
                    search_done.set()

            search_thread = threading.Thread(target=search_existing_runtime, daemon=True)
            search_thread.start()

            self.assertTrue(search_done.wait(timeout=1.0))
            self.assertFalse(second_prepare_done.is_set())
            self.assertEqual(search_errors, [])
            self.assertEqual(
                search_result,
                {
                    "snapshot_id": "snapshot-0001",
                    "payload": {"query_model_name": "colbertv2"},
                },
            )

            ready_events["snapshot-0002"].set()
            prepare_thread.join(timeout=5.0)
            search_thread.join(timeout=5.0)

            self.assertFalse(prepare_thread.is_alive())
            self.assertFalse(search_thread.is_alive())
            self.assertEqual(prepare_errors, [])

    def test_close_all_finishes_while_runtime_prepare_is_in_flight(self) -> None:
        registry = HostedPreparedExactRuntimeRegistry(service_root=Path("/tmp/service-root"))
        ready_event = threading.Event()
        created_event = threading.Event()
        created_runtimes: list[_FakeRuntime] = []

        def fake_prepare_runtime(**kwargs: object) -> _FakeRuntime:
            snapshot_id = kwargs["snapshot_id"]
            assert isinstance(snapshot_id, str)
            runtime = _FakeRuntime(
                snapshot_id=snapshot_id,
                ready_event=ready_event,
                created_event=created_event,
            )
            created_runtimes.append(runtime)
            return runtime

        with patch(
            "kayak_engine.hosted_prepared_exact_runtime_registry.prepare_exact_search_runtime",
            side_effect=fake_prepare_runtime,
        ):
            prepare_errors: list[BaseException] = []
            prepare_done = threading.Event()

            def prepare_runtime() -> None:
                try:
                    registry.prepare_runtime(
                        collection_id="news",
                        tenant_id="tenant-a",
                        namespace_id="search",
                        snapshot_id="snapshot-0001",
                        load_text_corpus=False,
                        config=self._prepare_config(),
                    )
                except BaseException as exc:  # pragma: no cover - surfaced below
                    prepare_errors.append(exc)
                finally:
                    prepare_done.set()

            prepare_thread = threading.Thread(target=prepare_runtime, daemon=True)
            prepare_thread.start()
            self.assertTrue(created_event.wait(timeout=5.0))

            close_result: list[dict[str, object]] = []
            close_errors: list[BaseException] = []
            close_done = threading.Event()

            def close_registry() -> None:
                try:
                    close_result.extend(registry.close_all())
                except BaseException as exc:  # pragma: no cover - surfaced below
                    close_errors.append(exc)
                finally:
                    close_done.set()

            close_thread = threading.Thread(target=close_registry, daemon=True)
            close_thread.start()

            self.assertTrue(close_done.wait(timeout=1.0))
            self.assertEqual(close_result, [])
            self.assertEqual(close_errors, [])

            ready_event.set()
            prepare_thread.join(timeout=5.0)
            close_thread.join(timeout=5.0)

            self.assertFalse(prepare_thread.is_alive())
            self.assertFalse(close_thread.is_alive())
            self.assertEqual(len(created_runtimes), 1)
            self.assertTrue(created_runtimes[0].closed)
            self.assertEqual(registry.active_runtime_count(), 0)
            self.assertEqual(len(prepare_errors), 1)
            self.assertIn("registry is closed", str(prepare_errors[0]))

