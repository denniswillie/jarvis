from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar
from uuid import UUID

from app.reminders.models import JarvisReminder, ReminderStatus, MAX_REMINDER_DELAY_SECONDS
from app.reminders.repository import ReminderRepository
from app.tools.inputs import (
    ToolInputError,
    failed,
    input_error_payload,
    optional_datetime,
    optional_positive_int,
    optional_string,
    required_string,
    required_uuid,
    success,
)
from app.tools.module import ToolBuildContext
from app.tools.registry import ToolDefinition

_ALLOWED_STATUSES: set[ReminderStatus] = {"active", "completed", "failed", "deleted"}


@dataclass(frozen=True)
class ReminderToolModule:
    repository: ReminderRepository
    name: ClassVar[str] = "reminders"

    def build_tools(self, context: ToolBuildContext) -> list[ToolDefinition]:
        if context.session_id is None:
            return []
        return create_reminder_tools(
            repository=self.repository,
            owner_id=context.owner_id,
            session_id=context.session_id,
        )


def create_reminder_tools(*, repository: ReminderRepository, owner_id: str, session_id: UUID) -> list[ToolDefinition]:
    async def create_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            title = required_string(arguments, "title", max_length=200)
            details = optional_string(arguments, "details", max_length=2000) or ""
            delay_seconds = arguments.get("delay_seconds")
            if delay_seconds is not None:
                if type(delay_seconds) is not int or not 1 <= delay_seconds <= MAX_REMINDER_DELAY_SECONDS:
                    return failed("invalid_tool_input",
                        f"delay_seconds must be an integer from 1 to {MAX_REMINDER_DELAY_SECONDS}.")
                if arguments.get("remind_at") is not None or arguments.get("condition") is not None:
                    return failed("invalid_tool_input",
                        "delay_seconds cannot be combined with remind_at or condition.")
            remind_at = optional_datetime(arguments, "remind_at")
            condition = optional_string(arguments, "condition", max_length=1000)
            if delay_seconds is None and remind_at is None and condition is None:
                return failed("invalid_tool_input", "delay_seconds, remind_at or condition is required.")
        except ToolInputError as error:
            return input_error_payload(error)
        reminder = await repository.create_reminder(
            owner_id=owner_id,
            session_id=session_id,
            title=title,
            details=details,
            remind_at=remind_at,
            condition=condition,
            **({"delay_seconds": delay_seconds} if delay_seconds is not None else {}),
        )
        return success({"reminder": _reminder_payload(reminder)})

    async def list_reminders(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_status = optional_string(arguments, "status")
            status = _parse_status(raw_status)
            limit = optional_positive_int(arguments, "limit", default=20, max_value=100)
        except ToolInputError as error:
            return input_error_payload(error)
        reminders = await repository.list_reminders(owner_id=owner_id, status=status, limit=limit)
        return success({"reminders": [_reminder_payload(reminder) for reminder in reminders]})

    async def complete_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            reminder_id = required_uuid(arguments, "reminder_id")
        except ToolInputError as error:
            return input_error_payload(error)
        reminder = await repository.complete_reminder(owner_id=owner_id, reminder_id=reminder_id)
        if reminder is None:
            return failed("reminder_not_found", "Reminder was not found.")
        return success({"reminder": _reminder_payload(reminder)})

    async def delete_reminder(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            reminder_id = required_uuid(arguments, "reminder_id")
        except ToolInputError as error:
            return input_error_payload(error)
        reminder = await repository.delete_reminder(owner_id=owner_id, reminder_id=reminder_id)
        if reminder is None:
            return failed("reminder_not_found", "Reminder was not found.")
        return success({"reminder": _reminder_payload(reminder)})

    return [
        ToolDefinition(
            name="create_reminder",
            mutates_state=True,
            description=(
                "Create a Jarvis reminder. For relative requests such as 'in ten seconds', use delay_seconds; "
                "the backend anchors the delay once when scheduling, not to an earlier context timestamp. "
                "Do not calculate an absolute time for a relative delay. Use remind_at for explicit absolute "
                "dates/times and condition for visual/session reminders such as 'when I am brushing my teeth'. "
                "delay_seconds cannot be combined with remind_at or condition."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["title"],
                "properties": {
                    "title": {"type": "string"},
                    "details": {"type": ["string", "null"]},
                    "delay_seconds": {"type": ["integer", "null"], "minimum": 1,
                        "maximum": MAX_REMINDER_DELAY_SECONDS,
                        "description": "Relative delay from the backend scheduling instant, in whole seconds. Mutually exclusive with remind_at and condition."},
                    "remind_at": {"type": ["string", "null"], "description": "ISO-8601 timestamp for time reminders."},
                    "condition": {"type": ["string", "null"], "description": "Natural-language visual/session condition."},
                },
            },
            handler=create_reminder,
        ),
        ToolDefinition(
            name="list_reminders",
            description="List Jarvis reminders. Defaults to recent reminders; pass status to filter.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "status": {
                        "type": ["string", "null"],
                        "enum": ["active", "completed", "failed", "deleted", None],
                    },
                    "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
                },
            },
            handler=list_reminders,
        ),
        ToolDefinition(
            name="complete_reminder",
            mutates_state=True,
            description="Mark a Jarvis reminder completed.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["reminder_id"],
                "properties": {"reminder_id": {"type": "string"}},
            },
            handler=complete_reminder,
        ),
        ToolDefinition(
            name="delete_reminder",
            mutates_state=True,
            description="Delete a Jarvis reminder by marking it deleted.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["reminder_id"],
                "properties": {"reminder_id": {"type": "string"}},
            },
            handler=delete_reminder,
        ),
    ]


def _parse_status(value: str | None) -> ReminderStatus | None:
    if value is None:
        return None
    if value not in _ALLOWED_STATUSES:
        raise ToolInputError(
            "invalid_tool_input",
            "status must be active, completed, failed, or deleted.",
        )
    return value  # type: ignore[return-value]


def _reminder_payload(reminder: JarvisReminder) -> dict[str, Any]:
    return {
        "id": str(reminder.id),
        "session_id": str(reminder.session_id) if reminder.session_id else None,
        "title": reminder.title,
        "details": reminder.details,
        "remind_at": reminder.remind_at.isoformat() if reminder.remind_at else None,
        "condition": reminder.condition,
        "status": reminder.status,
        "created_at": reminder.created_at.isoformat(),
        "updated_at": reminder.updated_at.isoformat(),
        "completed_at": reminder.completed_at.isoformat() if reminder.completed_at else None,
        "triggered_at": reminder.triggered_at.isoformat() if reminder.triggered_at else None,
        "trigger_attempt_count": reminder.trigger_attempt_count,
        "last_trigger_error": reminder.last_trigger_error,
        "metadata": reminder.metadata,
    }
