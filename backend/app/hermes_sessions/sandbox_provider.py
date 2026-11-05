"""Session-owned E2B scope and pause-only lifecycle, reusing the maintained SDK boundary.

Allocation bindings, concurrency and maintenance authorization belong to the
caller. No sandbox is created at construction, and this module owns no scheduler.
Paused retention is not proof that memory or processes survived a provider pause.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.sandbox.e2b_provider import E2BProvider
from app.sandbox.models import SandboxError, SandboxPolicy


class RetainedSandboxScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_id: str = Field(min_length=1, max_length=200, repr=False)
    profile_id: str = Field(min_length=1, max_length=256)
    logical_session_id: str = Field(min_length=1, max_length=256)
    allocation_id: UUID

    @field_validator("owner_id", "profile_id", "logical_session_id")
    @classmethod
    def exact_identity(cls, value):
        if value != value.strip() or "\x00" in value:
            raise ValueError("Retained sandbox identities must be nonblank, exact and NUL-free")
        return value

    def metadata(self, policy: SandboxPolicy) -> dict[str, str]:
        # Check before SDK effects, including accidentally unvalidated model_copy values.
        if (not isinstance(policy, SandboxPolicy) or type(policy.lifetime_seconds) is not int
                or not 60 <= policy.lifetime_seconds <= 3600):
            raise ValueError("Retained sandbox runtime lifetime must be between 60 and 3600 seconds")
        hashed = lambda value: hashlib.sha256(value.encode()).hexdigest()
        return {"jarvis_kind": "hermes-session-sandbox-v1",
                "jarvis_owner": hashed(self.owner_id), "jarvis_profile": hashed(self.profile_id),
                "jarvis_session": hashed(self.logical_session_id),
                "jarvis_allocation": str(self.allocation_id), "jarvis_policy": policy.fingerprint}


class RetainedSandboxInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sandbox_id: str = Field(min_length=1, max_length=200)
    scope: RetainedSandboxScope
    policy: SandboxPolicy
    state: Literal["running", "paused"]
    # Provider runtime deadline, not a deletion deadline for a paused resource.
    expires_at: datetime | None
    template_id: str
    # Compatibility with inherited prepare/resume; retained ownership never moves
    # between attempts, so an alternative provider creation scope is forbidden.
    native_scope: None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def running_deadline(self):
        if self.state == "running" and self.expires_at is None:
            raise ValueError("A running sandbox needs its observed runtime deadline")
        return self

    @property
    def provider_scope(self) -> RetainedSandboxScope:
        return self.scope


class RetainedE2BProvider(E2BProvider):
    """Same SDK operations, session metadata, timeout pause and explicit resume.

    Inherited pause uses the SDK default (memory-preserving) mode, not a
    filesystem-only request. Provider fallback can still lose memory/processes;
    callers must verify their own process receipts after resume.
    """

    _scope_type = RetainedSandboxScope
    _handle_type = RetainedSandboxInfo
    _on_timeout = "pause"

    async def kill(self, sandbox_id, scope, policy):
        raise SandboxError("retained_sandbox_delete_forbidden")

    async def delete(self, sandbox_id, scope, policy):
        raise SandboxError("retained_sandbox_delete_forbidden")
