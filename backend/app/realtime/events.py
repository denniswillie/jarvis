from __future__ import annotations

import json
from typing import Any


class ClientEventError(ValueError):
    pass


def parse_client_event(raw_message: str | bytes) -> dict[str, Any]:
    if isinstance(raw_message, bytes):
        try:
            raw_message = raw_message.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ClientEventError("Client event must be UTF-8 JSON.") from error

    try:
        event = json.loads(raw_message)
    except json.JSONDecodeError as error:
        raise ClientEventError("Client event must be valid JSON.") from error

    if not isinstance(event, dict):
        raise ClientEventError("Client event must be a JSON object.")

    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ClientEventError("Client event type is required.")

    return event


def encode_event(event_type: str, **payload: Any) -> str:
    return json.dumps(
        {"type": event_type, **payload},
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ClientEventError(f"{key} is required.")
    return value


def optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ClientEventError(f"{key} must be a non-empty string.")
    return value


def optional_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClientEventError(f"{key} must be an integer.")
    return value
