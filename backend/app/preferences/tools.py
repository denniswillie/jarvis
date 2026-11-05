from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from app.preferences.models import JarvisPreference
from app.preferences.repository import PreferenceRepository
from app.tools.inputs import (
    ToolInputError,
    input_error_payload,
    optional_positive_int,
    optional_string,
    required_string,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


@dataclass
class PreferenceToolModule:
    repository: PreferenceRepository
    name: ClassVar[str] = "preferences"
    _attention_service: Any = field(default=None, init=False, repr=False, compare=False)

    def bind_attention_service(self, service) -> None:
        """Runtime delivery dependency only; preferences remain repository-owned."""
        self._attention_service = service

    async def _enforce_owner_quiet(self, owner_id):
        service = self._attention_service
        if service is None or service._closed:
            raise RuntimeError("Attention delivery is not available")
        return await service.enforce_owner_quiet(owner_id)

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        from app.attention.quiet_tools import create_quiet_tools

        return create_preference_tools(
            repository=self.repository,
            owner_id=context.owner_id,
        ) + create_quiet_tools(
            repository=self.repository.attention_quiet(),
            owner_id=context.owner_id,
            session_id=context.session_id,
            # Resolve the runtime at invocation, including registries built
            # before attention readiness. Never capture a stale service value.
            enforce_owner_quiet=self._enforce_owner_quiet,
        )


def create_preference_tools(*, repository: PreferenceRepository, owner_id: str) -> list[ToolDefinition]:
    async def set_preference(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            key = required_string(arguments, "key", max_length=200)
            value = required_string(arguments, "value", max_length=4000)
        except ToolInputError as error:
            return input_error_payload(error)
        preference = await repository.set_preference(owner_id=owner_id, key=key, value=value)
        return success({"preference": _preference_payload(preference)})

    async def get_preferences(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            query = optional_string(arguments, "query", max_length=500)
            limit = optional_positive_int(arguments, "limit", default=20, max_value=100)
        except ToolInputError as error:
            return input_error_payload(error)
        preferences = await repository.search_preferences(owner_id=owner_id, query=query, limit=limit)
        return success({"preferences": [_preference_payload(preference) for preference in preferences]})

    return [
        ToolDefinition(
            name="set_preference",
            mutates_state=True,
            description="Store or update a durable Jarvis user preference or stable personal fact.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "value"],
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                },
            },
            handler=set_preference,
        ),
        ToolDefinition(
            name="get_preferences",
            description="Search Jarvis long-term preferences by key/value substring. Omit query to list recent preferences.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": ["string", "null"]},
                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
                },
            },
            handler=get_preferences,
        ),
    ]


def _preference_payload(preference: JarvisPreference) -> dict[str, Any]:
    return {
        "id": str(preference.id),
        "key": preference.key,
        "value": preference.value,
        "source": preference.source,
        "confidence": preference.confidence,
        "evidence_episode_ids": [
            str(episode_id) for episode_id in preference.evidence_episode_ids
        ],
        "created_at": preference.created_at.isoformat(),
        "updated_at": preference.updated_at.isoformat(),
    }
