from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from app.memory.prospective.models import VisualFocus, VisualFocusMode
from app.memory.prospective.service import ProspectiveMemoryService
from app.tools.inputs import (
    ToolInputError,
    input_error_payload,
    optional_string,
    required_string,
    required_uuid,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class ProspectiveToolModule:
    service: ProspectiveMemoryService
    name: ClassVar[str] = "prospective-memory"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        return create_prospective_tools(service=self.service, owner_id=context.owner_id)


def create_prospective_tools(
    *,
    service: ProspectiveMemoryService,
    owner_id: str,
) -> list[ToolDefinition]:
    async def set_visual_focus(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            target = required_string(arguments, "target", max_length=240)
            description = required_string(arguments, "description", max_length=1000)
            purpose = required_string(arguments, "purpose", max_length=500)
            mode = _mode(optional_string(arguments, "mode") or "find_once")
            expires_in_seconds = _optional_duration(arguments)
        except ToolInputError as error:
            return input_error_payload(error)
        focus = await service.set_visual_focus(
            owner_id=owner_id,
            target=target,
            description=description,
            purpose=purpose,
            mode=mode,
            expires_in_seconds=expires_in_seconds,
        )
        return success({"visual_focus": _focus_payload(focus)})

    async def get_visual_focus(_: dict[str, Any]) -> dict[str, Any]:
        focus = await service.get_visual_focus(owner_id=owner_id)
        return success(
            {"visual_focus": _focus_payload(focus) if focus is not None else None}
        )

    async def clear_visual_focus(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            focus_id = required_uuid(arguments, "focus_id")
        except ToolInputError as error:
            return input_error_payload(error)
        focus = await service.clear_visual_focus(owner_id=owner_id, focus_id=focus_id)
        return success(
            {
                "cleared": focus is not None,
                "focus_id": str(focus_id),
                "visual_focus": _focus_payload(focus) if focus is not None else None,
            }
        )

    return [
        ToolDefinition(
            name="set_visual_focus",
            mutates_state=True,
            description=(
                "Ask Jarvis's visual perception to actively find or monitor one visible target. "
                "For lost items such as 'help me find my keys', use find_once: notify once "
                "and finish after successful delivery. Use monitor ONLY for an explicit "
                "request to keep watching or report recurring changes until stopped. "
                "Switching from a box to keys requires calling this tool for keys; "
                "setting a focus replaces the previous one. Do not merely promise to search."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "description", "purpose"],
                "properties": {
                    "target": {"type": "string"},
                    "description": {"type": "string"},
                    "purpose": {"type": "string"},
                    "mode": {
                        "type": ["string", "null"],
                        "enum": ["find_once", "monitor", None],
                    },
                    "expires_in_seconds": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 86400,
                    },
                },
            },
            handler=set_visual_focus,
        ),
        ToolDefinition(
            name="get_visual_focus",
            description=(
                "Return the current active visual find/monitor focus and its latest delivery status. "
                "awaiting_outcome means an alert was dispatched but its outcome is pending or unknown; "
                "do not claim it was heard or automatically restart it. An explicit new search can "
                "replace it. delivered_to_session means audio was forwarded, not verified playback."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
            handler=get_visual_focus,
        ),
        ToolDefinition(
            name="clear_visual_focus",
            mutates_state=True,
            description=(
                "Stop the specified visual find/monitor request using its id from context, "
                "get_visual_focus, or set_visual_focus. Call this when the user "
                "says stop searching, never mind, or I already found it; a verbal "
                "acknowledgement alone does not cancel a search. This is idempotent for "
                "that id: an inactive request does not cancel a replacement with another id."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["focus_id"],
                "properties": {"focus_id": {"type": "string", "description": "The id of the visual focus to stop."}},
            },
            handler=clear_visual_focus,
        ),
    ]


def _mode(value: str) -> VisualFocusMode:
    if value not in {"find_once", "monitor"}:
        raise ToolInputError(
            "invalid_tool_input",
            "mode must be find_once or monitor.",
        )
    return value  # type: ignore[return-value]


def _optional_duration(arguments: dict[str, Any]) -> int | None:
    value = arguments.get("expires_in_seconds")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 86400:
        raise ToolInputError(
            "invalid_tool_input",
            "expires_in_seconds must be an integer between 1 and 86400.",
        )
    return value


def _focus_payload(focus: VisualFocus) -> dict[str, Any]:
    return {
        **focus.as_prompt_payload(),
        "completed_at": focus.completed_at.isoformat() if focus.completed_at else None,
        "cleared_at": focus.cleared_at.isoformat() if focus.cleared_at else None,
    }
