from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import ValidationError

from app.assistant.models import (
    MAX_BEHAVIOR_INSTRUCTIONS_LENGTH,
    BehaviorProfileRevisionConflict,
    BehaviorProfileUpdate,
)
from app.assistant.profile_service import BehaviorProfileService
from app.assistant.repository import BehaviorProfileRepository
from app.tools.inputs import success
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class BehaviorToolModule:
    repository: BehaviorProfileRepository
    name: ClassVar[str] = "behavior"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        return create_behavior_tools(
            repository=self.repository,
            owner_id=context.owner_id,
        )


def create_behavior_tools(
    *,
    repository: BehaviorProfileRepository,
    owner_id: str,
) -> list[ToolDefinition]:
    service = BehaviorProfileService(repository)

    async def get_behavior_instructions(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments:
            return _input_error("get_behavior_instructions takes no arguments.")
        profile = await service.get(owner_id=owner_id)
        return success(
            {"behavior_profile": profile, "scope": "persistent_owner"}
        )

    async def set_behavior_instructions(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            update = BehaviorProfileUpdate.model_validate(arguments)
        except ValidationError:
            return _input_error(
                "Provide behavior_instructions (at most 12000 characters) and "
                "expected_revision (the nonnegative integer returned by get_behavior_instructions), "
                "with no other fields."
            )
        try:
            profile = await service.update(owner_id=owner_id, update=update)
        except BehaviorProfileRevisionConflict as error:
            return {"status": "failed", "effect_committed": False, "error": error.error_payload()}
        return success(
            {
                "behavior_profile": profile,
                "scope": "persistent_owner",
                "effect_committed": True,
            }
        )

    return [
        ToolDefinition(
            name="get_behavior_instructions",
            description=(
                "Read the owner's persistent Jarvis behavior instructions and current revision. "
                "Read before editing. A stored profile is not proof that any model consumed it."
            ),
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            handler=get_behavior_instructions,
        ),
        ToolDefinition(
            name="set_behavior_instructions",
            mutates_state=True,
            description=(
                "Replace the owner's persistent Jarvis behavior instructions. Use this when the "
                "user asks Jarvis to change how it should behave in future sessions or how its "
                "visual analyzer sub-agents should proactively help. Pass an empty string to clear "
                "the customization. Use reminder tools for a concrete timed or visual-condition "
                "reminder."
                " Supply expected_revision from get_behavior_instructions. On conflict, read "
                "again and reconcile the requested change; never blindly overwrite a newer edit. "
                "Success confirms storage, not downstream instruction use."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["behavior_instructions", "expected_revision"],
                "properties": {
                    "behavior_instructions": {
                        "type": "string",
                        "maxLength": MAX_BEHAVIOR_INSTRUCTIONS_LENGTH,
                    },
                    "expected_revision": {"type": "integer", "minimum": 0,
                                          "maximum": 9_223_372_036_854_775_806},
                },
            },
            handler=set_behavior_instructions,
        ),
    ]


def _input_error(message: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": {"code": "invalid_tool_arguments", "message": message},
    }


