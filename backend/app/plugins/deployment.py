"""Explicit deployment defaults, never a synthetic user request or grant.

Keep this identity/content fixed across workers and restarts. Replacing the
revision-zero baseline requires version-aware client CAS before any rollout.
Account policy, including an unavailable head, always supersedes these defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from uuid import UUID, uuid5

from app.plugins.models import DefaultChoice, PluginChoice, PolicySnapshot
from app.plugins.policy import CATEGORIES, PluginPolicyState, validate_policy


@dataclass(frozen=True, slots=True)
class PluginDeploymentBaseline:
    version: str
    policy: PolicySnapshot

    def __post_init__(self):
        if not isinstance(self.version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", self.version):
            raise ValueError("An explicit immutable deployment version is required")
        if not isinstance(self.policy, PolicySnapshot):
            raise TypeError("A validated policy snapshot is required")
        validate_policy(self.policy)
        if self.policy.revision != 0:
            raise ValueError("Deployment defaults must have revision zero")
        entries = {entry.plugin_id: entry for entry in self.policy.entries}
        defaults = {choice.category: choice.plugin_id for choice in self.policy.defaults}
        if (set(entries) != set(CATEGORIES)
                or defaults != {category: plugin for plugin, category in CATEGORIES.items()}
                or any(not entry.desired_enabled or not isinstance(entry.configuration_id, UUID)
                       for entry in entries.values())):
            raise ValueError("The deployment baseline must explicitly name all three enabled bundles/defaults")

    @property
    def identity(self) -> str:
        content = json.dumps(self.policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return self.version + ":" + sha256(content.encode("utf-8")).hexdigest()

    def matches(self, state: PluginPolicyState) -> bool:
        return (isinstance(state, PluginPolicyState) and type(state.revision) is int
                and state.revision == 0 and state.payload_id is None and state.policy == self.policy)


_CONFIGURATION_NAMESPACE = UUID("f0ee01e2-105c-45e6-9ca4-2ee8744078e0")
_VERSION = "jarvis.curated-plugins.v1"
CURATED_DEPLOYMENT_BASELINE_V1 = PluginDeploymentBaseline(
    version=_VERSION,
    policy=PolicySnapshot(revision=0,
        entries=tuple(PluginChoice(plugin_id=plugin, desired_enabled=True,
            configuration_id=uuid5(_CONFIGURATION_NAMESPACE, _VERSION + ":" + plugin))
            for plugin in sorted(CATEGORIES)),
        defaults=tuple(DefaultChoice(category=category, plugin_id=plugin)
            for plugin, category in sorted(CATEGORIES.items(), key=lambda item: item[1]))),
)
