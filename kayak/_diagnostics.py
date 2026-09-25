"""Bounded event data for the named Kayak logger; no logging configuration."""

import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

logger = logging.getLogger("kayak.diagnostics")
VERSION = version("kayak")


def exception_details(error: BaseException) -> list[dict[str, object]]:
    """Keep bounded exception types and frame locations, excluding values and messages."""
    details: list[dict[str, object]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(details) < 4:
        seen.add(id(current))
        frames: list[dict[str, object]] = []
        frame = current.__traceback__
        while frame is not None:
            frames.append(
                {
                    "file": Path(frame.tb_frame.f_code.co_filename).name[:128],
                    "function": frame.tb_frame.f_code.co_name[:128],
                    "line": frame.tb_lineno,
                }
            )
            frames = frames[-8:]
            frame = frame.tb_next
        details.append({"type": type(current).__name__[:128], "frames": frames})
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return details


def emit(
    event: str,
    *,
    request_id: str | None = None,
    duration: float | None = None,
    status_code: int | None = None,
    code: str | None = None,
    route: str | None = None,
    fingerprint: str | None = None,
    error: BaseException | None = None,
) -> None:
    """Emit allowlisted fields. A failing application handler must not change a decision."""
    if not logger.isEnabledFor(logging.INFO):
        return
    fields: dict[str, object] = {
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        "kayak_version": VERSION,
    }
    for name, value in (
        ("request_id", request_id),
        ("duration_seconds", duration),
        ("status_code", status_code),
        ("code", code),
        ("route", route),
        ("model_fingerprint", fingerprint),
    ):
        if value is not None:
            fields[name] = value
    if error is not None:
        fields["exceptions"] = exception_details(error)
    try:
        logger.info(json.dumps(fields, ensure_ascii=True, separators=(",", ":")))
    except Exception:
        # Logging is an optional effect, including application-installed handlers.
        pass
