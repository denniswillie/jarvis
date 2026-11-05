"""Bounded Files browse positions; a cursor never grants access to a file."""
import base64
from dataclasses import dataclass
from datetime import datetime
import json
from uuid import UUID

from app.knowledge.models import SourceError


@dataclass(frozen=True)
class FilePage:
    files: list
    next_cursor: str | None


def encode_cursor(created_at, file_id):
    body = json.dumps([created_at.isoformat(), str(file_id)], separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")


def decode_cursor(cursor):
    if cursor is None:
        return None, None
    try:
        if not isinstance(cursor, str) or not 1 <= len(cursor) <= 200:
            raise ValueError()
        value = json.loads(base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True))
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(x, str) for x in value):
            raise ValueError()
        timestamp, identity = datetime.fromisoformat(value[0]), UUID(value[1])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError()
        return timestamp, identity
    except (ValueError, TypeError, UnicodeError):
        raise SourceError("file_cursor_invalid", "This file-list position is invalid. Refresh the library.", 422) from None
