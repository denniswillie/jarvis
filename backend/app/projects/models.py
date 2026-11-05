"""Version-one connector envelopes. The Mac independently validates every read."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import PurePosixPath
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, Field, JsonValue, SecretStr, field_validator, model_validator

from app.evidence.models import FrozenRecord
from app.knowledge.models import SourceError


WIRE_VERSION = 1
MAX_BODY_BYTES = 2_000_000
MAX_PAYLOAD_BYTES = 230_000
OPERATIONS = {"status", "files", "read", "search", "symbols", "definition", "references", "changes"}
READ_FIELDS = {"operation", "path", "query", "language", "line", "column", "limit", "offset",
               "max_chars", "expected_digest", "expected_snapshot"}


def encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def relative_path(value):
    if (not value or len(value) > 2048 or "\\" in value or
            any(ord(char) < 32 for char in value) or
            PurePosixPath(value).is_absolute() or
            any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Use a canonical relative project path")
    return value


def bounded_payload(value):
    if len(encoded(value).encode("ascii")) > MAX_PAYLOAD_BYTES:
        raise SourceError("project_result_too_large", "The bounded project evidence exceeds source storage limits.")
    return value


class RootOffer(FrozenRecord):
    local_root_id: UUID
    label: str = Field(min_length=1, max_length=100, strict=True)
    fingerprint: str = Field(pattern="^[0-9a-f]{64}$")


class Enrollment(FrozenRecord):
    area_id: UUID
    label: str = Field(min_length=1, max_length=100, strict=True)
    roots: tuple[RootOffer, ...] = Field(min_length=1, max_length=16)
    expires_in_days: int = Field(default=30, ge=1, le=90, strict=True)

    @model_validator(mode="after")
    def unique_roots(self):
        if len({root.local_root_id for root in self.roots}) != len(self.roots):
            raise ValueError("Root identities must be unique")
        return self


class Poll(FrozenRecord):
    protocol_version: Literal[1] = 1
    roots: tuple[RootOffer, ...] = Field(max_length=16)

    @model_validator(mode="after")
    def unique_roots(self):
        if len({root.local_root_id for root in self.roots}) != len(self.roots):
            raise ValueError("Root identities must be unique")
        return self


class Inspection(FrozenRecord):
    root_id: UUID
    request: dict[str, JsonValue]

    @field_validator("request")
    @classmethod
    def validate_read(cls, request):
        if (set(request) - READ_FIELDS or request.get("operation") not in OPERATIONS
                or len(encoded(request)) > 8000):
            raise ValueError("Use a bounded version-one read operation, never a shell command")
        if request.get("path") is not None:
            relative_path(request["path"])
        for field, low, high in (("limit", 1, 30), ("offset", 0, 1000000),
                                 ("max_chars", 1, 32000), ("line", 1, 1000000), ("column", 1, 1000000)):
            if field in request and (type(request[field]) is not int or not low <= request[field] <= high):
                raise ValueError("Read options exceed the network protocol bounds")
        return {"limit": 30, **request}


class Original(FrozenRecord):
    path: str
    text: str = Field(max_length=262144, strict=True)
    content_digest: str = Field(pattern="^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def valid_path(cls, value):
        return relative_path(value)

    @model_validator(mode="after")
    def digest_matches(self):
        if "\x00" in self.text or sha256(self.text.encode("utf-8")).hexdigest() != self.content_digest:
            raise ValueError("Original content must match its exact UTF-8 digest")
        bounded_payload(self.model_dump(mode="json"))
        return self


class Completion(FrozenRecord):
    protocol_version: Literal[1] = 1
    request_id: UUID
    lease_id: UUID
    captured_at: AwareDatetime
    result: dict[str, JsonValue]
    originals: tuple[Original, ...] = Field(default=(), max_length=30)

    @model_validator(mode="after")
    def bounded_result(self):
        bounded_payload(self.result)
        if len({original.path for original in self.originals}) != len(self.originals):
            raise ValueError("Duplicate original paths")
        return self


class ConnectorCredential(FrozenRecord):
    connector_id: UUID
    token: SecretStr
