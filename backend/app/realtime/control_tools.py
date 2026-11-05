from __future__ import annotations

from typing import Any

from app.tools.registry import JarvisToolRegistry, ToolDefinition


DISMISS_VISUAL_INTERRUPT = "dismiss_visual_interrupt"


def create_visual_interrupt_control_registry() -> JarvisToolRegistry:
    """Local response controls, exposed only while evaluating visual candidates."""
    registry = JarvisToolRegistry()
    registry.register(ToolDefinition(
        name=DISMISS_VISUAL_INTERRUPT,
        description=(
            "Silently dismiss the current unsolicited visual candidate when it is "
            "unsupported, stale, or not useful to interrupt for. Ends this response "
            "without speech. Do not announce or explain the dismissal."
        ),
        input_schema={
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
        handler=_dismiss_visual_interrupt,
    ))
    return registry


async def _dismiss_visual_interrupt(arguments: dict[str, Any]) -> dict[str, Any]:
    if arguments:
        return {
            "status": "failed",
            "error": {
                "code": "invalid_tool_arguments",
                "message": "dismiss_visual_interrupt takes no arguments.",
            },
        }
    return {"status": "success", "result": {"dismissed": True}}
