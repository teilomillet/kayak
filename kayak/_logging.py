"""CLI-owned JSON log output with bounded buffering and shutdown time."""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import TextIO

from ._diagnostics import logger


class BufferedLogHandler(logging.Handler):
    """A slow stderr must not occupy the model slot; excess records are dropped."""

    def __init__(self, stream: TextIO, *, capacity: int = 1024) -> None:
        super().__init__()
        if capacity < 1:
            raise ValueError("log capacity must be positive")
        self._stream = stream
        self._queue: Queue[str] = Queue(maxsize=capacity)
        self._stopped = Event()
        self._loss_lock = Lock()
        self._lost = 0
        self._worker = Thread(target=self._write_records, name="kayak-diagnostics", daemon=True)
        self._worker.start()

    def emit(self, record: logging.LogRecord) -> None:
        if self._stopped.is_set():
            return
        try:
            self._queue.put_nowait(record.getMessage())
        except Full:
            with self._loss_lock:
                self._lost += 1

    def _write(self, line: str) -> bool:
        try:
            self._stream.write(line + "\n")
            self._stream.flush()
            return True
        except Exception:
            return False

    def _report_loss(self) -> None:
        with self._loss_lock:
            lost = self._lost
        if lost and self._write(json.dumps({"event": "diagnostics.dropped", "records": lost})):
            with self._loss_lock:
                self._lost -= lost

    def _write_records(self) -> None:
        while not self._stopped.is_set() or not self._queue.empty():
            try:
                line = self._queue.get(timeout=0.05)
            except Empty:
                self._report_loss()
                continue
            self._report_loss()
            if not self._write(line):
                with self._loss_lock:
                    self._lost += 1
        self._report_loss()

    def close(self) -> None:
        self._stopped.set()
        # A stuck OS write cannot be cancelled by Python. The daemon cannot
        # prevent process exit, and its pending queue remains bounded.
        self._worker.join(timeout=1.0)
        super().close()


@contextmanager
def json_logging(enabled: bool, *, stream: TextIO | None = None) -> Iterator[None]:
    """Configure only Kayak's event logger for the duration of one CLI service."""
    if not enabled:
        yield
        return
    handler = BufferedLogHandler(sys.stderr if stream is None else stream)
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers = handlers
        logger.setLevel(level)
        logger.propagate = propagate
        handler.close()
