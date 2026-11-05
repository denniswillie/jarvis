"""Pure command identity helpers, not authorization or replay decisions.

Provider call IDs remain opaque and unchanged for wire correlation. Tool names
and arguments belong to a separate digest, never to the operation UUID.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from uuid import NAMESPACE_URL, UUID, uuid5


_COMMAND_NAMESPACE = uuid5(NAMESPACE_URL, "jarvis.command.v1")


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")


def command_operation_id(
    owner_id: str,
    session_id: UUID,
    provider_session_key: str,
    provider_call_id: str,
) -> UUID:
    """Identify a call within an owner, app session, and provider session.

    The owner is an opaque identifier, not necessarily a UUID. Whitespace is
    inspected only to reject blank identifiers; supplied strings are otherwise
    preserved exactly, including case and Unicode representation.
    """
    _validate_identifier(owner_id, "owner_id")
    if not isinstance(session_id, UUID):
        raise TypeError("session_id must be a UUID")
    _validate_identifier(provider_session_key, "provider_session_key")
    _validate_identifier(provider_call_id, "provider_call_id")

    name = json.dumps(
        [owner_id, str(session_id).lower(), provider_session_key, provider_call_id],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return uuid5(_COMMAND_NAMESPACE, name)


def _json_value(value: object, active_containers: set[int]) -> object:
    """Copy JSON containers while rejecting coercions and circular values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("command arguments must contain only finite numbers")
        return value
    if not isinstance(value, (Mapping, list)):
        raise TypeError("command arguments must contain only JSON values")

    container_id = id(value)
    if container_id in active_containers:
        raise ValueError("command arguments must not contain circular values")
    active_containers.add(container_id)
    try:
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError("command argument object keys must be strings")
                result[key] = _json_value(item, active_containers)
            return result
        return [_json_value(item, active_containers) for item in value]
    finally:
        active_containers.remove(container_id)


def command_arguments_digest(
    tool_name: str,
    normalized_arguments: Mapping[str, object],
) -> str:
    """Return a SHA-256 hex digest of the canonical tool/arguments JSON.

    Arguments must already be normalized by the caller. JSON objects may be
    mappings, but arrays must be lists; tuples and other non-JSON values are
    rejected rather than silently coerced. No supplied container is modified.
    """
    _validate_identifier(tool_name, "tool_name")
    if not isinstance(normalized_arguments, Mapping):
        raise TypeError("normalized_arguments must be a mapping")

    payload = {
        "tool": tool_name,
        "arguments": _json_value(normalized_arguments, set()),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
