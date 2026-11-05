"""Lazy capability configuration: unused integrations cannot break voice startup.

Configured is deliberately not operational readiness. Credentials are checked
for presence only; database migrations and native provider access need their
own readiness checks when a capability is actually used.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.knowledge.connection import hindsight_origin, private_http_origin
from app.settings import is_configured_secret
from app.sandbox.models import SandboxPolicy
from app.files.generated import ArtifactPolicy
from app.files.transfer import CAPTURE_MAX_BYTES, MP4_OUTPUT_MAX_BYTES


class MemoryAuthority(StrEnum):
    LEGACY = "legacy"
    HINDSIGHT = "hindsight"


class Capability(StrEnum):
    TASKS = "tasks"
    ATTENTION = "attention"
    ATTENTION_VOICE = "attention_voice"
    FRONTIER = "frontier"
    KNOWLEDGE = "knowledge"
    RESPONSIBILITIES = "responsibilities"
    SANDBOX = "sandbox"
    ARTIFACTS = "artifacts"
    RECORDINGS = "recordings"
    PUSH = "push"
    MAIL = "mail"


DEFAULT_FRONTIER_MODEL = "gpt-5.6-sol"

PositiveInt = Annotated[int, Field(gt=0, le=1_000_000_000)]


class TaskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    owner_concurrency: PositiveInt = 2
    global_concurrency: PositiveInt = 4
    owner_queue_limit: PositiveInt = 20
    active_seconds: PositiveInt = 900
    tool_steps: PositiveInt = 40
    request_output_tokens: PositiveInt = 16_000
    task_output_tokens: PositiveInt = 32_000
    task_input_tokens: PositiveInt = 250_000
    owner_daily_tokens: PositiveInt = 2_000_000
    max_attempts: PositiveInt = 8
    memory_edit_grant_seconds: int = Field(default=3600, ge=60, le=86400)
    lease_seconds: PositiveInt = 30
    poll_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 0.5
    shutdown_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 5

    @model_validator(mode="after")
    def validate_limits(self):
        if self.owner_concurrency > self.global_concurrency:
            raise ValueError("owner concurrency cannot exceed global concurrency")
        if self.request_output_tokens > self.task_output_tokens:
            raise ValueError("request output cannot exceed task output budget")
        return self


class AttentionConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    lease_seconds: int = Field(default=30, ge=5, le=300)
    max_attempts: int = Field(default=3, ge=1, le=20)


class AttentionVoiceConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    api_key: SecretStr
    model: str = Field(default="gpt-4o-mini-tts-2025-12-15", min_length=1, max_length=100)
    voice: str = Field(default="cedar", min_length=1, max_length=100)
    poll_seconds: float = Field(default=1, gt=0, le=60, allow_inf_nan=False)
    receipt_seconds: float = Field(default=60, ge=5, le=300, allow_inf_nan=False)


class PushConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    private_key: SecretStr
    key_id: str = Field(pattern=r"^[A-Z0-9]{10}$")
    team_id: str = Field(pattern=r"^[A-Z0-9]{10}$")
    environment: Literal["sandbox", "production"] = "sandbox"
    topic: str = Field(default="com.symbiote.jarvis", min_length=1, max_length=200,
                       pattern=r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
    fallback_seconds: float = Field(default=20, ge=0, le=3600, allow_inf_nan=False)
    poll_seconds: float = Field(default=2, gt=0, le=60, allow_inf_nan=False)
    http_timeout_seconds: float = Field(default=10, gt=0, le=30, allow_inf_nan=False)
    lease_seconds: int = Field(default=30, ge=5, le=120)
    max_attempts: int = Field(default=3, ge=1, le=10)

    @model_validator(mode="after")
    def bounded_request(self):
        if self.http_timeout_seconds + 2 >= self.lease_seconds:
            raise ValueError("The APNs deadline must fit within the delivery lease")
        return self


class FrontierModelConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(min_length=1, max_length=100, pattern=r"^\S+$")
    reasoning_effort: str = Field(pattern=r"^(low|medium|high|xhigh|max|ultra)$")


class FrontierConfiguration(FrontierModelConfiguration):
    api_key: SecretStr


class KnowledgeConfiguration(BaseModel):
    """Connection and lifecycle bounds; configuration is not native readiness."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    private_http_origin: str | None = Field(default=None, repr=False)
    base_url: str = Field(repr=False)
    api_key: SecretStr
    area_id: UUID
    http_timeout_seconds: Annotated[float, Field(gt=0, le=120, allow_inf_nan=False)] = 20
    worker_deadline_seconds: Annotated[float, Field(gt=0, le=180, allow_inf_nan=False)] = 25
    worker_lease_seconds: Annotated[int, Field(gt=0, le=300)] = 60
    native_inflight_limit: Annotated[int, Field(gt=0, le=64)] = 8
    poll_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 1
    shutdown_seconds: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 5

    @field_validator("private_http_origin")
    @classmethod
    def explicit_private_endpoint(cls, value):
        return private_http_origin(value) if value is not None else None

    @field_validator("base_url")
    @classmethod
    def private_endpoint(cls, value, info):
        return hindsight_origin(value, approved_http_origin=info.data.get("private_http_origin"))

    @model_validator(mode="after")
    def bounded_step(self):
        if not is_configured_secret(self.api_key.get_secret_value()):
            raise ValueError("A configured memory-service API key is required")
        if not self.http_timeout_seconds <= self.worker_deadline_seconds < self.worker_lease_seconds:
            raise ValueError("HTTP timeout must fit the step, and the step must fit the lease")
        return self


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    capability: Capability
    enabled: bool
    configured: bool
    configuration: TaskLimits | AttentionConfiguration | AttentionVoiceConfiguration | PushConfiguration | FrontierConfiguration | KnowledgeConfiguration | SandboxConfiguration | ArtifactConfiguration | RecordingConfiguration | None = None
    issue: str | None = None
    invalid_fields: tuple[str, ...] = ()


class SandboxConfiguration(BaseModel):
    """Public computation bounds, not a native credential or readiness claim."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy: SandboxPolicy = Field(default_factory=SandboxPolicy)
    grant_seconds: int = Field(default=3600, ge=60, le=86400)
    cleanup_poll_seconds: float = Field(default=2, gt=0, le=60, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=20, ge=1, le=60, allow_inf_nan=False)
    shutdown_seconds: float = Field(default=10, ge=1, le=20, allow_inf_nan=False)


class MediaInspectorConfiguration(BaseModel):
    """Operator-provisioned media limits and executable pin, not readiness proof."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    storage_max_file_bytes: int = Field(default=CAPTURE_MAX_BYTES, ge=1, le=CAPTURE_MAX_BYTES)
    ffprobe_path: str = Field(min_length=1, max_length=4096)
    ffprobe_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    probe_timeout_seconds: float = Field(default=10, gt=0, le=30, allow_inf_nan=False)

    @field_validator("ffprobe_path")
    @classmethod
    def absolute_probe(cls, value):
        if not Path(value).is_absolute() or "\x00" in value:
            raise ValueError("Provision an absolute media-inspector path")
        return value


class ArtifactMediaConfiguration(MediaInspectorConfiguration):
    storage_max_file_bytes: int = Field(ge=1, le=MP4_OUTPUT_MAX_BYTES)


class RecordingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    media: MediaInspectorConfiguration
    area_id: UUID


def artifact_worker_capabilities(policy: ArtifactPolicy) -> tuple[str, ...]:
    return ("files.publish_generated",) + (("files.publish_generated_mp4",) if policy.supports_media else ())


class ArtifactConfiguration(BaseModel):
    """Public publication policy; a ready storage-capable worker is still required."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)
    area_id: UUID
    grant_seconds: int = Field(default=3600, ge=60, le=86400)
    media: ArtifactMediaConfiguration | None = None

    @property
    def required_capabilities(self):
        return artifact_worker_capabilities(self.policy)


class SecondBrainSettings(BaseSettings):
    # Raw strings intentionally defer parsing until the selected feature is used.
    enabled: str = "false"
    tasks_enabled: str = "false"
    attention_enabled: str = "false"
    attention_voice_enabled: str = "false"
    attention_voice_model: str = "gpt-4o-mini-tts-2025-12-15"
    attention_voice_voice: str = "cedar"
    attention_voice_poll_seconds: str = "1"
    attention_voice_receipt_seconds: str = "60"
    attention_lease_seconds: str = "30"
    attention_max_attempts: str = "3"
    frontier_enabled: str = "false"
    knowledge_enabled: str = "false"
    memory_authority: str = "legacy"
    responsibilities_enabled: str = "false"
    sandbox_enabled: str = "false"
    sandbox_internet_enabled: str = "false"
    sandbox_desktop_enabled: str = "false"
    artifacts_enabled: str = "false"
    artifact_max_files: str = "8"
    artifact_max_file_bytes: str = "8388608"
    artifact_max_total_bytes: str = "33554432"
    artifact_formats: str = "text,csv,json,pdf,xlsx"
    artifact_grant_seconds: str = "3600"
    artifact_mp4_max_file_bytes: str = ""
    artifact_media_storage_max_file_bytes: str = ""
    artifact_media_ffprobe_path: str = ""
    artifact_media_ffprobe_sha256: str = ""
    artifact_media_probe_timeout_seconds: str = "10"
    recordings_enabled: str = "false"
    recording_area_id: str = ""
    recording_storage_max_file_bytes: str = str(CAPTURE_MAX_BYTES)
    recording_ffprobe_path: str = ""
    recording_ffprobe_sha256: str = ""
    recording_probe_timeout_seconds: str = "10"
    sandbox_lifetime_seconds: str = "300"
    sandbox_command_seconds: str = "60"
    sandbox_output_bytes: str = "64000"
    sandbox_grant_seconds: str = "3600"
    sandbox_cleanup_poll_seconds: str = "2"
    sandbox_request_timeout_seconds: str = "20"
    sandbox_shutdown_seconds: str = "10"
    push_enabled: str = "false"
    push_private_key: str = Field(default="", repr=False)
    push_key_id: str = ""
    push_team_id: str = ""
    push_environment: str = "sandbox"
    push_topic: str = "com.symbiote.jarvis"
    push_fallback_seconds: str = "20"
    push_poll_seconds: str = "2"
    push_http_timeout_seconds: str = "10"
    push_lease_seconds: str = "30"
    push_max_attempts: str = "3"
    mail_enabled: str = "false"
    owner_concurrency: str = "2"
    global_concurrency: str = "4"
    owner_queue_limit: str = "20"
    active_seconds: str = "900"
    tool_steps: str = "40"
    request_output_tokens: str = "16000"
    task_output_tokens: str = "32000"
    task_input_tokens: str = "250000"
    owner_daily_tokens: str = "2000000"
    max_attempts: str = "8"
    memory_edit_grant_seconds: str = "3600"
    lease_seconds: str = "30"
    poll_seconds: str = "0.5"
    shutdown_seconds: str = "5"
    frontier_model: str = DEFAULT_FRONTIER_MODEL
    frontier_reasoning_effort: str = "medium"
    hermes_python: str = "/opt/hermes/bin/python"
    hermes_work_root: str = "/var/lib/jarvis/hermes"
    worker_host_id: str = ""
    worker_heartbeat_seconds: str = "5"
    hindsight_url: str = Field(default="", repr=False)
    hindsight_api_key: str = Field(default="", repr=False)
    hindsight_api_key_file: str = Field(default="", repr=False)
    hindsight_private_http_origin: str = Field(default="", repr=False)
    knowledge_area_id: str = ""
    knowledge_http_timeout_seconds: str = "20"
    knowledge_worker_deadline_seconds: str = "25"
    knowledge_worker_lease_seconds: str = "60"
    knowledge_native_inflight_limit: str = "8"
    knowledge_poll_seconds: str = "1"
    knowledge_shutdown_seconds: str = "5"

    model_config = SettingsConfigDict(
        env_prefix="SECOND_BRAIN_", env_file=".env", extra="ignore", frozen=True,
        env_file_encoding="utf-8",
    )

    def selected_memory_authority(self) -> MemoryAuthority:
        """Explicit selection never silently falls back on provider/configuration failure."""
        try:
            return MemoryAuthority(self.memory_authority)
        except ValueError as error:
            raise ValueError("invalid_memory_authority") from error

    def frontier_selection(self):
        """API metadata requires no native interpreter or frontier credential."""
        if not TypeAdapter(bool).validate_python(self.enabled) or not TypeAdapter(bool).validate_python(self.frontier_enabled):
            return None
        return FrontierModelConfiguration(model=self.frontier_model,reasoning_effort=self.frontier_reasoning_effort)

    def resolve(
        self, capability: Capability, *, database_url: str | None = None,
        provider_key: str | None = None,
    ) -> CapabilityResolution:
        try:
            if not TypeAdapter(bool).validate_python(self.enabled):
                return CapabilityResolution(capability, False, False)
            if not TypeAdapter(bool).validate_python(getattr(self, f"{capability.value}_enabled")):
                return CapabilityResolution(capability, False, False)
        except ValidationError:
            return CapabilityResolution(capability, False, False, issue="invalid_feature_flag")
        if not is_configured_secret(database_url):
            return CapabilityResolution(capability, True, False, issue="database_not_configured")
        try:
            if capability == Capability.TASKS:
                config = TaskLimits(**{name: getattr(self, name) for name in TaskLimits.model_fields})
            elif capability == Capability.ATTENTION:
                config = AttentionConfiguration(lease_seconds=self.attention_lease_seconds,
                                                max_attempts=self.attention_max_attempts)
            elif capability == Capability.ATTENTION_VOICE:
                if not is_configured_secret(provider_key):
                    return CapabilityResolution(capability, True, False, issue="provider_key_not_configured")
                config = AttentionVoiceConfiguration(api_key=SecretStr(provider_key),
                    model=self.attention_voice_model, voice=self.attention_voice_voice,
                    poll_seconds=self.attention_voice_poll_seconds, receipt_seconds=self.attention_voice_receipt_seconds)
            elif capability == Capability.FRONTIER:
                if not is_configured_secret(provider_key):
                    return CapabilityResolution(capability, True, False, issue="provider_key_not_configured")
                config = FrontierConfiguration(
                    model=self.frontier_model, reasoning_effort=self.frontier_reasoning_effort,
                    api_key=SecretStr(provider_key),
                )
            elif capability == Capability.KNOWLEDGE:
                key = self.hindsight_api_key
                if self.hindsight_api_key_file:
                    if is_configured_secret(key):
                        return CapabilityResolution(capability, True, False, issue="memory_api_key_configuration_ambiguous")
                    from app.knowledge.service_entrypoint import secret_file
                    from app.knowledge.service_policy import ServicePolicyError
                    try:
                        key = secret_file(self.hindsight_api_key_file)
                    except ServicePolicyError:
                        return CapabilityResolution(capability, True, False, issue="memory_api_key_file_unavailable")
                if not is_configured_secret(key):
                    return CapabilityResolution(capability, True, False, issue="memory_api_key_not_configured")
                config = KnowledgeConfiguration(
                    base_url=self.hindsight_url, api_key=SecretStr(key),
                    private_http_origin=self.hindsight_private_http_origin or None,
                    area_id=self.knowledge_area_id,
                    **{name: getattr(self, "knowledge_" + name) for name in (
                        "http_timeout_seconds", "worker_deadline_seconds", "worker_lease_seconds",
                        "native_inflight_limit", "poll_seconds", "shutdown_seconds",
                    )},
                )
            elif capability == Capability.PUSH:
                if not is_configured_secret(self.push_private_key):
                    return CapabilityResolution(capability, True, False, issue="push_key_not_configured")
                config = PushConfiguration(private_key=SecretStr(self.push_private_key),
                    **{name: getattr(self, "push_" + name) for name in (
                        "key_id", "team_id", "environment", "topic", "fallback_seconds",
                        "poll_seconds", "http_timeout_seconds", "lease_seconds", "max_attempts")})
            elif capability == Capability.SANDBOX:
                # API metadata is deliberately credential-free. A matching
                # authenticated worker readiness lease is still required.
                policy = SandboxPolicy(internet_access=TypeAdapter(bool).validate_python(self.sandbox_internet_enabled),
                    template="desktop" if TypeAdapter(bool).validate_python(self.sandbox_desktop_enabled) else "base",
                    **{name: TypeAdapter(int).validate_python(getattr(self, "sandbox_"+name))
                    for name in ("lifetime_seconds", "command_seconds", "output_bytes")})
                config = SandboxConfiguration(policy=policy, **{name:getattr(self, "sandbox_"+name) for name in
                    ("grant_seconds", "cleanup_poll_seconds", "request_timeout_seconds", "shutdown_seconds")})
            elif capability == Capability.RECORDINGS:
                config = RecordingConfiguration(area_id=self.recording_area_id or self.knowledge_area_id,
                    media=MediaInspectorConfiguration(**{
                    name: getattr(self, "recording_" + name)
                    for name in MediaInspectorConfiguration.model_fields}))
            elif capability == Capability.ARTIFACTS:
                computation = self.resolve(Capability.SANDBOX, database_url=database_url)
                memory = self.resolve(Capability.KNOWLEDGE, database_url=database_url)
                if not computation.configured or not memory.configured:
                    return CapabilityResolution(capability, True, False, issue="artifact_dependencies_unavailable")
                formats = tuple(value.strip() for value in self.artifact_formats.split(","))
                media = None
                media_policy = {}
                if "mp4" in formats:
                    media_policy["mp4_max_file_bytes"] = TypeAdapter(int).validate_python(
                        self.artifact_mp4_max_file_bytes)
                    media = ArtifactMediaConfiguration(**{
                        name: getattr(self, "artifact_media_" + name)
                        for name in ArtifactMediaConfiguration.model_fields})
                policy = ArtifactPolicy(
                    **{name: TypeAdapter(int).validate_python(getattr(self, "artifact_" + name))
                       for name in ("max_files", "max_file_bytes", "max_total_bytes")},
                    formats=formats, **media_policy)
                config = ArtifactConfiguration(policy=policy, area_id=memory.configuration.area_id,
                                               grant_seconds=self.artifact_grant_seconds, media=media)
            elif capability == Capability.MAIL:
                return CapabilityResolution(capability, True, False, issue="integration_not_implemented")
            else:
                config = None
        except ValidationError as error:
            return CapabilityResolution(
                capability, True, False, issue="invalid_configuration",
                invalid_fields=tuple(sorted({str(item["loc"][0]) if item["loc"] else "limits"
                                             for item in error.errors(include_input=False, include_url=False)})),
            )
        return CapabilityResolution(capability, True, True, configuration=config)


@lru_cache(maxsize=1)
def get_second_brain_settings() -> SecondBrainSettings:
    return SecondBrainSettings()
