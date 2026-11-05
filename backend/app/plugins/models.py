"""Trusted inputs and data-only views for the first curated plugin catalog.

Callers scope these observations to an authenticated owner. Constructing a
model does not grant permission or establish an installed software fact.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


PluginId = Literal["video_editor", "browser", "files_data"]
Category = Literal["video_editing", "web_browsing", "files_data"]
Digest = Annotated[str, Field(strict=True, pattern=r"^[0-9a-f]{64}$")]
Reason = Annotated[str, Field(strict=True, min_length=1, max_length=100, pattern=r"^[a-z0-9_]+$")]
Reference = Annotated[str, Field(strict=True, min_length=1, max_length=200, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
BundleVersion = Annotated[str, Field(strict=True, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")]


class FrozenRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class PluginChoice(FrozenRecord):
    plugin_id: PluginId
    desired_enabled: StrictBool
    configuration_id: UUID


class DefaultChoice(FrozenRecord):
    category: Category
    plugin_id: PluginId


class PolicySnapshot(FrozenRecord):
    """Explicit current snapshot; None at the projection boundary means unknown.

    Revision zero may be a Core-established baseline. This module never creates
    that baseline or silently substitutes it for unavailable persisted policy.
    """
    revision: Annotated[int, Field(strict=True, ge=0)]
    entries: tuple[PluginChoice, ...] = Field(max_length=3)
    defaults: tuple[DefaultChoice, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def unique_choices(self):
        if len({item.plugin_id for item in self.entries}) != len(self.entries):
            raise ValueError("Duplicate plugin policy entry")
        if len({item.category for item in self.defaults}) != len(self.defaults):
            raise ValueError("Duplicate default category")
        return self


class TimedObservation(FrozenRecord):
    plugin_id: PluginId
    configuration_id: UUID
    observed_at: AwareDatetime
    expires_at: AwareDatetime

    @field_validator("observed_at", "expires_at", mode="before")
    @classmethod
    def explicit_timestamp(cls, value):
        if not isinstance(value, (datetime, str)):
            raise ValueError("An offset-aware timestamp is required")
        return value

    @model_validator(mode="after")
    def positive_lifetime(self):
        if self.expires_at <= self.observed_at:
            raise ValueError("Observation expiry must follow observation time")
        return self

    def current_at(self, now: datetime) -> bool:
        return self.observed_at <= now < self.expires_at


class InventoryObservation(TimedObservation):
    state: Literal["installed", "absent", "unknown"]
    bundle_version: BundleVersion | None = None
    bundle_sha256: Digest | None = None
    reason_code: Reason | None = None

    @model_validator(mode="after")
    def exact_installed_bundle(self):
        has_bundle = self.bundle_version is not None and self.bundle_sha256 is not None
        if self.state == "installed" and not has_bundle:
            raise ValueError("Installed observations require exact version and fingerprint")
        if self.state != "installed" and (self.bundle_version is not None or self.bundle_sha256 is not None):
            raise ValueError("Absent/unknown observations cannot claim an installed bundle")
        return self


class ReadinessObservation(TimedObservation):
    state: Literal["ready", "unavailable", "unknown"]
    bundle_sha256: Digest | None = None
    reason_code: Reason | None = None

    @model_validator(mode="after")
    def qualified_bundle(self):
        if self.state == "ready" and self.bundle_sha256 is None:
            raise ValueError("Readiness must identify the qualified bundle")
        return self


class RecoveryView(FrozenRecord):
    kind: Literal["retry", "recheck"]
    reference: Reference


class LifecycleObservation(TimedObservation):
    """Projection of an existing owner operation, never an installer scheduler."""
    operation_ref: Reference
    phase: Literal["preparing", "installing", "removing", "failed", "unknown"]
    recovery: RecoveryView | None = None
    reason_code: Reason | None = None

    @model_validator(mode="after")
    def recovery_matches_phase(self):
        if self.recovery is not None:
            if self.phase in {"preparing", "installing", "removing"}:
                raise ValueError("An active operation cannot offer a second recovery action")
            if self.phase == "unknown" and self.recovery.kind != "recheck":
                raise ValueError("An unknown outcome cannot offer a new retry")
        return self


class OperationView(FrozenRecord):
    operation_ref: Reference
    configuration_id: UUID
    phase: Literal["preparing", "installing", "removing", "failed", "unknown"]
    reason_code: Reason | None = None
    recovery: RecoveryView | None = None


class PluginCard(FrozenRecord):
    plugin_id: PluginId
    title: str
    description: str
    category: Category
    descriptor_version: Literal[1] = 1
    selected_default: bool
    desired_enabled: bool | None
    configuration_id: UUID | None
    installed: Literal["installed", "absent", "unknown"]
    installed_version: str | None
    readiness: Literal["ready", "unavailable", "unknown"]
    reason_code: Reason | None
    observations_valid_until: AwareDatetime | None
    operation: OperationView | None
    # An explicit runtime capability; observations alone never activate it.
    can_change: StrictBool = False

    @model_validator(mode="after")
    def change_requires_known_choice(self):
        if self.can_change and (self.desired_enabled is None or self.configuration_id is None):
            raise ValueError("Plugin changes require a known policy choice")
        return self


class PluginCatalog(FrozenRecord):
    schema_version: Literal[1] = 1
    evaluated_at: AwareDatetime
    policy_revision: int | None
    policy_available: bool
    items: tuple[PluginCard, ...]
