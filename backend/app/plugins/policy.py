"""Bounded account-level plugin intent, independent of installer authority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid5

from pydantic import Field, StrictBool, model_validator

from app.errors import JarvisHTTPError
from app.plugins.models import Category, DefaultChoice, FrozenRecord, PluginChoice, PluginId, PolicySnapshot


CAPABILITY = "plugins.curated"
UI_NAMESPACE = "ui:plugins.policy"
CATEGORIES = {"video_editor": "video_editing", "browser": "web_browsing", "files_data": "files_data"}


class EnabledChange(FrozenRecord):
    plugin_id: PluginId
    enabled: StrictBool


class DefaultChange(FrozenRecord):
    category: Category
    plugin_id: PluginId | None

    @model_validator(mode="after")
    def compatible(self):
        if self.plugin_id is not None and CATEGORIES[self.plugin_id] != self.category:
            raise ValueError("The preferred plugin must support this category")
        return self


class PluginPolicyChange(FrozenRecord):
    mutation_id: UUID
    expected_revision: int = Field(strict=True, ge=0, le=2**63 - 2)
    enabled_changes: tuple[EnabledChange, ...] = Field(default=(), max_length=3)
    default_changes: tuple[DefaultChange, ...] = Field(default=(), max_length=3)

    @model_validator(mode="after")
    def bounded_unique_changes(self):
        if not self.enabled_changes and not self.default_changes:
            raise ValueError("At least one explicit plugin choice is required")
        if len({change.plugin_id for change in self.enabled_changes}) != len(self.enabled_changes):
            raise ValueError("A plugin may be changed only once per request")
        if len({change.category for change in self.default_changes}) != len(self.default_changes):
            raise ValueError("A category default may be changed only once per request")
        return self

    def canonical_intent(self) -> dict:
        """Order does not change the meaning of disjoint bounded choices."""
        return self.model_copy(update={
            "enabled_changes": tuple(sorted(self.enabled_changes, key=lambda change: change.plugin_id)),
            "default_changes": tuple(sorted(self.default_changes, key=lambda change: change.category)),
        }).model_dump(mode="json")

    def ingress_payload(self) -> dict:
        return {"plugin_policy": self.canonical_intent()}


class StoredPluginPolicy(FrozenRecord):
    schema_version: Literal[1] = 1
    intent: PluginPolicyChange
    policy: PolicySnapshot


@dataclass(frozen=True)
class PluginPolicyState:
    revision: int
    policy: PolicySnapshot | None
    payload_id: UUID | None = None


def validate_policy(policy: PolicySnapshot) -> PolicySnapshot:
    policy = PolicySnapshot.model_validate(policy)
    entries = {entry.plugin_id: entry for entry in policy.entries}
    if any(choice.plugin_id not in entries or CATEGORIES[choice.plugin_id] != choice.category
           for choice in policy.defaults):
        raise ValueError("A preferred default must name a known compatible choice")
    return policy


def changed_policy(current: PolicySnapshot, command: PluginPolicyChange, admitted_event_id: UUID) -> PolicySnapshot:
    """The event UUID comes from trusted ingress, never a client operation grant.

    Configuration identity changes only for a new enabled/disabled intent. A
    default change or an unrelated card keeps its existing configuration.
    """
    validate_policy(current)
    if current.revision != command.expected_revision:
        raise JarvisHTTPError(409, "plugin_policy_revision_conflict", "Plugin choices changed. Refresh before trying again.")
    entries = {entry.plugin_id: entry for entry in current.entries}
    defaults = {choice.category: choice.plugin_id for choice in current.defaults}
    for change in command.enabled_changes:
        old = entries.get(change.plugin_id)
        if old is None or old.desired_enabled != change.enabled:
            entries[change.plugin_id] = PluginChoice(plugin_id=change.plugin_id, desired_enabled=change.enabled,
                configuration_id=uuid5(admitted_event_id, CAPABILITY + ":" + change.plugin_id))
    for change in command.default_changes:
        if change.plugin_id is None:
            defaults.pop(change.category, None)
        elif change.plugin_id not in entries:
            raise JarvisHTTPError(409, "plugin_choice_unavailable", "This plugin has no confirmed choice yet.")
        else:
            defaults[change.category] = change.plugin_id
    return validate_policy(PolicySnapshot(revision=current.revision + 1,
        entries=tuple(entries[key] for key in sorted(entries)),
        defaults=tuple(DefaultChoice(category=key, plugin_id=defaults[key]) for key in sorted(defaults))))
