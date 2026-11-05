from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
STRUCTURED_FIELDS = (
    "event",
    "request_id",
    "method",
    "path",
    "status_code",
    "duration_ms",
    "session_id",
    "owner_id",
    "component",
    "operation",
    "job_id",
    "event_id",
    "memory_type",
    "attempt",
    "outcome",
    "provider",
    "model",
    "latency_ms",
    "error_code",
)


class JsonLogFormatter(logging.Formatter):
    """Serialize stable operational fields without logging request bodies or secrets."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in STRUCTURED_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(level: str, *, json_logs: bool = False) -> None:
    """Configure process logging once.

    Keep this intentionally small: structured enough to be useful in Docker
    logs, but without adding a logging framework before the backend needs one.
    """

    normalized_level = level.upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(normalized_level)

    formatter: logging.Formatter
    if json_logs:
        formatter = JsonLogFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(normalized_level)

    # Wire-level DEBUG logs include HTTP headers and websocket media frames.
    # Keep normal HTTP request/status diagnostics and provider warnings visible.
    # Uvicorn also uses its error logger for inbound websocket wire traffic.
    for logger_name in (
        "httpcore", "websockets.client", "websockets.server", "openai", "uvicorn.error",
    ):
        logging.getLogger(logger_name).setLevel(max(root_logger.level, logging.INFO))
