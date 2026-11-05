"""Phone registration and APNs outcomes; tokens are never public API fields."""
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class DeviceMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID
    expected_revision: int = Field(ge=0, strict=True)


class DeviceRegistration(DeviceMutation):
    environment: Literal["sandbox", "production"]
    topic: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    token: SecretStr | None = None
    permission: Literal["not_determined", "denied", "authorized", "provisional", "ephemeral"]
    enabled: bool = Field(strict=True)
    quiet_until: AwareDatetime | None = None

    @field_validator("token")
    @classmethod
    def valid_token(cls, value):
        if value is None:
            return None
        raw = value.get_secret_value()
        # Token length is deliberately not fixed to a particular device/OS.
        if not 2 <= len(raw) <= 4096 or len(raw) % 2 or any(c not in "0123456789abcdefABCDEF" for c in raw):
            raise ValueError("A hexadecimal APNs device token is required")
        return SecretStr(raw.lower())

    @model_validator(mode="after")
    def enabled_requires_address(self):
        if self.enabled and (self.token is None or self.permission not in {"authorized", "provisional", "ephemeral"}):
            raise ValueError("Enable notifications only with permission and a device token")
        return self

    def token_hash(self):
        return sha256(self.token.get_secret_value().encode()).hexdigest() if self.token else None


@dataclass(frozen=True, slots=True)
class PushClaim:
    owner_id: str
    intent_id: UUID
    attempt_id: UUID
    generation: int
    device_id: UUID
    device_revision: int
    worker_id: str


@dataclass(frozen=True, slots=True)
class PushRequest:
    claim: PushClaim
    token: str = field(repr=False)
    environment: str
    topic: str
    expires_at: datetime | None
    urgency: str


@dataclass(frozen=True, slots=True)
class PushAuthorization:
    """One-use canonical attempt admission, not wire or phone-delivery evidence."""
    claim: PushClaim
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PushOutcome:
    state: Literal["accepted", "failed", "unknown", "revoked"]
    reason: str | None = None
    provider_status: int | None = None
    provider_id: UUID | None = None
    retry_seconds: int | None = None
    invalidate_token: bool = False
    invalidated_at: datetime | None = None
