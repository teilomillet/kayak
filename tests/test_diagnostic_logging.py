"""Bounded log effects, including slow/broken sinks and explicit ownership."""

import io
import json
import logging
from threading import Event, Thread

import pytest

from kayak._diagnostics import emit, exception_details, logger
from kayak._logging import BufferedLogHandler, json_logging


def test_configuration_is_scoped_and_off_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)
    original = logger.handlers[:], logger.level, logger.propagate
    with json_logging(False):
        assert (logger.handlers, logger.level, logger.propagate) == original
    assert not stream.getvalue()
    with json_logging(True):
        emit("test.event", request_id="r1")
    assert (logger.handlers, logger.level, logger.propagate) == original
    record = json.loads(stream.getvalue())
    assert record["event"] == "test.event" and record["request_id"] == "r1"


def test_blocked_sink_does_not_block_producers_and_loss_is_visible() -> None:
    class SlowStream(io.StringIO):
        def write(self, text: str) -> int:
            entered.set()
            assert release.wait(5), "test did not release the log sink"
            return super().write(text)

    entered, release, finished = Event(), Event(), Event()
    stream = SlowStream()
    handler = BufferedLogHandler(stream, capacity=2)

    def send(index: int) -> None:
        handler.handle(
            logging.LogRecord("test", logging.INFO, "", 0, json.dumps({"i": index}), (), None)
        )

    def producer() -> None:
        for index in range(1, 10):
            send(index)
        finished.set()

    send(0)
    try:
        assert entered.wait(2)
        thread = Thread(target=producer)
        thread.start()
        assert finished.wait(2), "producer waited for blocked I/O"
        thread.join(timeout=2)
        assert handler._queue.qsize() == 2
        # Shutdown has a deadline even while the OS sink remains blocked.
        closed = Event()

        def close() -> None:
            handler.close()
            closed.set()

        closer = Thread(target=close)
        closer.start()
        assert closed.wait(2)
        closer.join(timeout=2)
    finally:
        release.set()
        handler.close()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert sum(record.get("records", 0) for record in records) == 7
    assert [record["i"] for record in records if "i" in record] == [0, 1, 2]


def test_broken_sink_recovers_and_reports_dropped_record() -> None:
    class BrokenOnce(io.StringIO):
        broken = False

        def write(self, text: str) -> int:
            if not self.broken:
                self.broken = True
                raise OSError("private-sink-failure")
            return super().write(text)

    stream = BrokenOnce()
    handler = BufferedLogHandler(stream)
    try:
        for index in range(2):
            handler.handle(
                logging.LogRecord("test", logging.INFO, "", 0, json.dumps({"i": index}), (), None)
            )
    finally:
        handler.close()
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records == [{"event": "diagnostics.dropped", "records": 1}, {"i": 1}]
    assert "private-sink-failure" not in stream.getvalue()


def test_throwing_application_handler_cannot_replace_the_original_outcome() -> None:
    class BrokenHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            raise RuntimeError("logging failed")

    handler = BrokenHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        emit("test.event")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        handler.close()


def test_exception_cycles_are_bounded_and_messages_are_excluded() -> None:
    first = ValueError("private-first")
    second = RuntimeError("private-second")
    first.__cause__ = second
    second.__cause__ = first
    assert exception_details(first) == [
        {"type": "ValueError", "frames": []},
        {"type": "RuntimeError", "frames": []},
    ]
