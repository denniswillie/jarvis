"""Provider identities and bounds, not model-issued authority or a second agent loop."""
from datetime import datetime
import hashlib
import json
from typing import Literal
from uuid import UUID
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_serializer, model_validator


class SandboxError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class SandboxPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    template: Literal["base", "desktop"] = "base"
    lifetime_seconds: int = Field(default=300, ge=60, le=3600, strict=True)
    command_seconds: int = Field(default=60, ge=1, le=300, strict=True)
    output_bytes: int = Field(default=64000, ge=1024, le=256000, strict=True)
    allowed_domains: tuple[str, ...] = Field(default=(), max_length=16)
    internet_access: bool = Field(default=False, strict=True)

    @field_validator("allowed_domains")
    @classmethod
    def exact_domains(cls, values):
        # These are network authority selectors, not language interpretation.
        for value in values:
            labels = value.split(".")
            if (not 3 <= len(value) <= 253 or len(labels) < 2 or value != value.lower()
                    or not value.isascii() or labels[-1].isdigit()
                    or any(not 1 <= len(label) <= 63 or not label[0].isalnum() or not label[-1].isalnum()
                           or any(not (character.isalnum() or character == "-") for character in label)
                           for label in labels)):
                raise ValueError("Use exact lowercase DNS names, not URLs, wildcards, IPs or CIDRs")
        if len(set(values)) != len(values):
            raise ValueError("Duplicate network domain")
        return values

    @model_validator(mode="after")
    def unambiguous_network(self):
        if self.internet_access and self.allowed_domains:
            raise ValueError("Choose general internet access or a domain allowlist, not both")
        return self

    @model_serializer(mode="wrap")
    def compatible_policy(self, handler):
        values = handler(self)
        # Old grants, native metadata and cleanup handles retain their exact
        # policy shape/fingerprint. Missing this field always means false.
        if not self.internet_access:
            values.pop("internet_access", None)
        return values

    @property
    def networked(self):
        return self.internet_access or bool(self.allowed_domains)

    @property
    def required_capabilities(self):
        capabilities = ("sandbox.execute", "sandbox.network") if self.networked else ("sandbox.execute",)
        return capabilities + (("sandbox.desktop",) if self.template == "desktop" else ())

    def within(self, ceiling):
        return ((self.template == "base" or ceiling.template == "desktop")
                and all(getattr(self, name) <= getattr(ceiling, name)
                    for name in ("lifetime_seconds", "command_seconds", "output_bytes"))
                and (ceiling.internet_access or
                     (not self.internet_access and set(self.allowed_domains) <= set(ceiling.allowed_domains))))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()


class SandboxScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    owner_id: str = Field(min_length=1, max_length=200, repr=False)
    task_id: UUID
    task_revision: int = Field(ge=1, strict=True)
    attempt_id: UUID
    allocation_id: UUID

    @field_validator("owner_id")
    @classmethod
    def owner(cls, value):
        if value != value.strip() or "\x00" in value:
            raise ValueError("Invalid sandbox owner")
        return value

    def metadata(self, policy):
        return {"jarvis_kind": "task-sandbox-v1", "jarvis_owner": hashlib.sha256(self.owner_id.encode()).hexdigest(),
            "jarvis_task": str(self.task_id), "jarvis_revision": str(self.task_revision),
            "jarvis_attempt": str(self.attempt_id), "jarvis_allocation": str(self.allocation_id),
            "jarvis_policy": policy.fingerprint}


class E2BConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    api_key: SecretStr = Field(repr=False, exclude=True)
    request_timeout_seconds: float = Field(default=20, ge=1, le=60, allow_inf_nan=False)
    transfer_timeout_seconds: float = Field(default=300, ge=1, le=300, allow_inf_nan=False)

    @field_validator("api_key")
    @classmethod
    def configured_key(cls, value):
        text = value.get_secret_value()
        if not text.strip() or text != text.strip() or text.lower() in {"placeholder", "changeme"}:
            raise ValueError("An explicit E2B credential is required")
        return value


class SandboxInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sandbox_id: str = Field(min_length=1, max_length=200)
    scope: SandboxScope
    policy: SandboxPolicy
    # Current holder scope is distinct from immutable provider creation metadata.
    native_scope: SandboxScope | None = None

    @property
    def provider_scope(self):
        return self.native_scope or self.scope

    state: Literal["running", "paused"]
    expires_at: datetime
    template_id: str


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    pid: int = Field(gt=0)
    exit_code: int
    stdout: str
    stderr: str


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid",frozen=True)
    operation_key: str = Field(min_length=1,max_length=100,
        description="Stable key for this logical command; never use a new key to retry unknown work.")
    command: str = Field(min_length=1,max_length=16000)
    cwd: str = Field(default=".",min_length=1,max_length=1000,
        description="Workspace-relative directory. Use '.' for the initial workspace, not an absolute path or '..'.")

    @field_validator("operation_key","command","cwd")
    @classmethod
    def nonblank(cls,value):
        if not value.strip() or "\x00" in value:
            raise ValueError("Command fields must be nonblank and NUL-free")
        return value

    @field_validator("cwd")
    @classmethod
    def workspace_relative(cls,value):
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Use a workspace-relative working directory")
        return str(path)


class CommandReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid",frozen=True)
    format: Literal["jarvis.sandbox.command.v1"]
    operation_id: UUID
    allocation_id: UUID
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    pid: int = Field(gt=0,strict=True)
    phase: Literal["running","completed"]
    started: bool = Field(strict=True)
    exit_code: int | None = Field(strict=True)
    stdout: str
    stderr: str
    timed_out: bool = Field(strict=True)
    output_limited: bool = Field(strict=True)
    runner_error: str | None = Field(max_length=100)

    @property
    def fingerprint(self):
        # Native bounded receipts can exceed the smaller database-copy limit.
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True,
            ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
