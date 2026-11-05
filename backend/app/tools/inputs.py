from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID


class ToolInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def required_string(arguments: dict[str, Any], key: str, *, max_length: int | None = None,
                    preserve_whitespace: bool = False) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolInputError("invalid_tool_input", f"{key} is required.")
    result = value if preserve_whitespace else value.strip()
    if max_length is not None and len(result) > max_length:
        raise ToolInputError("invalid_tool_input", f"{key} must be at most {max_length} characters.")
    return result


def optional_string(arguments: dict[str, Any], key: str, *, max_length: int | None = None) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolInputError("invalid_tool_input", f"{key} must be a string.")
    stripped = value.strip()
    if not stripped:
        return None
    if max_length is not None and len(stripped) > max_length:
        raise ToolInputError("invalid_tool_input", f"{key} must be at most {max_length} characters.")
    return stripped


def required_uuid(arguments: dict[str, Any], key: str) -> UUID:
    value = required_string(arguments, key)
    try:
        return UUID(value)
    except ValueError as error:
        raise ToolInputError("invalid_tool_input", f"{key} must be a UUID.") from error


def optional_uuid(arguments: dict[str, Any], key: str) -> UUID | None:
    value = optional_string(arguments, key)
    if value is None:
        return None
    try:
        return UUID(value)
    except ValueError as error:
        raise ToolInputError("invalid_tool_input", f"{key} must be a UUID.") from error


def optional_positive_int(arguments: dict[str, Any], key: str, *, default: int, max_value: int) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolInputError("invalid_tool_input", f"{key} must be an integer.")
    if value < 1:
        raise ToolInputError("invalid_tool_input", f"{key} must be positive.")
    return min(value, max_value)


def optional_datetime(arguments: dict[str, Any], key: str) -> datetime | None:
    value = optional_string(arguments, key)
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ToolInputError("invalid_tool_input", f"{key} must be ISO-8601.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def success(payload: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", **payload}


def failed(code: str, message: str) -> dict[str, Any]:
    return {"status": "failed", "error": {"code": code, "message": message}}


def input_error_payload(error: ToolInputError) -> dict[str, Any]:
    return failed(error.code, error.message)
