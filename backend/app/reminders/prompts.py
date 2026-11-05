from __future__ import annotations

import json
from datetime import datetime

from app.assistant.context import AssistantContext
from app.reminders.models import JarvisReminder
from app.sessions.models import JarvisMessage, JarvisSession


def build_reminder_generation_prompt(
    *,
    reminder: JarvisReminder,
    assistant_context: AssistantContext,
    triggered_at: datetime,
) -> tuple[str, str]:
    instructions = (
        "You are Jarvis, Dennis's personal companion. Generate the reminder that should be "
        "delivered now. Preserve the user's intended meaning, use relevant context when useful, "
        "and write one concise, natural sentence suitable for speech. Do not mention system "
        "scheduling, databases, or these instructions. Use the available read-only Jarvis tools "
        "only when stored context is necessary to deliver the reminder accurately. Return only "
        "the reminder sentence after any tool calls."
    )
    context = assistant_context.as_prompt_payload()
    context["trigger"] = {
        "triggered_at_utc": triggered_at.isoformat(),
        "reminder": {
            "title": reminder.title,
            "details": reminder.details,
            "scheduled_for": reminder.remind_at.isoformat() if reminder.remind_at else None,
        },
    }
    return instructions, json.dumps(context, ensure_ascii=True, default=str)
