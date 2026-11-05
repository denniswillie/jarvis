from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from openai import AsyncOpenAI

from app.assistant.context import AssistantContextBuilder
from app.attention.models import AttentionError
from app.assistant.context_providers import AssistantContextProvider
from app.assistant.tool_limits import ToolCallBudget
from app.realtime.openai_client import safety_identifier_for_owner
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.models import JarvisReminder
from app.reminders.prompts import build_reminder_generation_prompt
from app.reminders.repository import ReminderRepository
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tools.executor import JarvisToolExecutor
from app.tools.module import ToolBuildContext, ToolModule
from app.tools.registry import create_tool_registry


LOGGER = logging.getLogger(__name__)
REMINDER_TOOL_NAMES = frozenset(
    {
        "search_notes",
        "list_reminders",
        "get_preferences",
        "get_behavior_instructions",
        "search_memory",
    }
)
MAX_REMINDER_TOOL_ROUNDS = 2


class ReminderMessageGenerator:
    def __init__(
        self,
        settings: JarvisSettings,
        *,
        tool_modules: Sequence[ToolModule],
        context_providers: Sequence[AssistantContextProvider],
    ) -> None:
        self._settings = settings
        self._model = settings.OPENAI_REMINDER_MODEL
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self._tool_modules = tuple(tool_modules)
        self._context_builder = AssistantContextBuilder(
            providers=context_providers,
        )

    async def close(self) -> None:
        await self._client.close()

    async def generate(
        self,
        *,
        reminder: JarvisReminder,
        owner_id: str,
        session_repository: SessionRepository,
    ) -> str:
        session = None
        recent_messages = []
        if reminder.session_id is not None:
            session = await session_repository.get_session(owner_id, reminder.session_id)
            if session is not None:
                recent_messages = await session_repository.get_recent_messages(
                    owner_id,
                    reminder.session_id,
                    limit=8,
                )
        assistant_context = await self._context_builder.build(
            owner_id=owner_id,
            session=session,
            recent_messages=recent_messages,
        )
        instructions, input_text = build_reminder_generation_prompt(
            reminder=reminder,
            assistant_context=assistant_context,
            triggered_at=datetime.now(timezone.utc),
        )
        if not self._model:
            raise RuntimeError("OpenAI reminder model is not configured.")
        registry = create_tool_registry(
            modules=self._tool_modules,
            context=ToolBuildContext(
                settings=self._settings,
                owner_id=owner_id,
                session_id=reminder.session_id,
            ),
        )
        allowed_tools = [tool for tool in registry.list() if tool.name in REMINDER_TOOL_NAMES]
        executor = JarvisToolExecutor(registry)
        tool_budget = ToolCallBudget(
            max_total_calls=self._settings.JARVIS_MAX_TOOL_CALLS_PER_RESPONSE,
            max_repeated_calls=self._settings.JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE,
        )
        input_items: list[dict[str, Any]] = [{"role": "user", "content": input_text}]
        response = None
        for tool_round in range(MAX_REMINDER_TOOL_ROUNDS + 1):
            tools = (
                [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                    for tool in allowed_tools
                ]
                if tool_round < MAX_REMINDER_TOOL_ROUNDS
                else []
            )
            response = await self._client.responses.create(
                model=self._model,
                instructions=instructions,
                input=input_items,
                tools=tools,
                max_output_tokens=120,
                store=False,
                safety_identifier=safety_identifier_for_owner(owner_id),
            )
            calls = _response_function_calls(response.output)
            if not calls:
                break
            input_items.extend(_response_output_items(response.output))
            for call in calls:
                name = call.get("name")
                call_id = call.get("call_id")
                arguments = _tool_arguments(call.get("arguments"))
                if not isinstance(name, str) or name not in REMINDER_TOOL_NAMES:
                    result = {
                        "status": "failed",
                        "error": {
                            "code": "tool_not_allowed",
                            "message": "Tool is not allowed for reminders.",
                        },
                    }
                else:
                    result = await executor.execute(name, arguments, budget=tool_budget)
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result, ensure_ascii=True, separators=(",", ":")),
                    }
                )
        message = response.output_text.strip() if response is not None else ""
        if not message:
            raise RuntimeError("OpenAI returned an empty reminder message.")
        return message


class ReminderDeliveryService:
    def __init__(
        self,
        *,
        owner_id: str,
        repository: ReminderRepository,
        session_repository: SessionRepository,
        realtime_session_manager: RealtimeSessionManager,
        generator: ReminderMessageGenerator,
    ) -> None:
        self._owner_id = owner_id
        self._repository = repository
        self._session_repository = session_repository
        self._realtime_session_manager = realtime_session_manager
        self._generator = generator

    async def close(self) -> None:
        await self._generator.close()

    async def deliver(self, reminder: JarvisReminder) -> None:
        LOGGER.info(
            "reminder generation started reminder_id=%s attempt=%s",
            reminder.id,
            reminder.trigger_attempt_count,
        )
        attention = None
        selected = getattr(self._realtime_session_manager, "attention_notifications_enabled", False) is True
        if selected:
            attention = self._realtime_session_manager.attention_repository
            if attention is None:
                raise AttentionError("attention_unavailable", "The selected reminder inbox is unavailable.")
            # The repository reads current title/details under its claim lock.
            # No unrelated model retrieval or completion claim enters this text.
            message = ""
        else:
            message = await self._generator.generate(
                reminder=reminder,
                owner_id=self._owner_id,
                session_repository=self._session_repository,
            )
        event = await self._repository.finalize_trigger(
            owner_id=self._owner_id,
            reminder_id=reminder.id,
            claim_token=_required_claim_token(reminder),
            session_id=reminder.session_id,
            message=message,
            metadata={"trigger_attempt": reminder.trigger_attempt_count,
                      **({"delivery_route":"attention", "playback_status":"unconfirmed",
                          "physical_action_status":"not_established"} if selected else {})},
            **({"attention":attention} if selected else {}),
        )
        LOGGER.info(
            "reminder event stored reminder_id=%s reminder_event_id=%s",
            reminder.id,
            event.id,
        )

        if selected:
            # The committed intent owns delivery, including disconnect/restart.
            # Never send this occurrence through the native route as well.
            return

        try:
            LOGGER.info(
                "reminder realtime delivery attempted reminder_id=%s reminder_event_id=%s",
                reminder.id,
                event.id,
            )
            delivered_session_id = await self._realtime_session_manager.speak_reminder(
                owner_id=self._owner_id,
                preferred_jarvis_session_id=reminder.session_id,
                message=message,
                reminder_id=reminder.id,
                reminder_event_id=event.id,
            )
        except Exception:
            LOGGER.exception("reminder realtime delivery failed reminder_id=%s", reminder.id)
            await self._repository.mark_event_delivery(
                owner_id=self._owner_id,
                event_id=event.id,
                event_type="failed",
                metadata={"delivery_route": "realtime", "reason": "dispatch_failed"},
            )
            return

        if delivered_session_id is None:
            await self._repository.mark_event_delivery(
                owner_id=self._owner_id,
                event_id=event.id,
                event_type="missed",
                metadata={"delivery_route": "persisted_event"},
            )
            LOGGER.info("reminder persisted for later display reminder_id=%s", reminder.id)
            return

        LOGGER.info(
            "reminder dispatched reminder_id=%s session_id=%s",
            reminder.id,
            delivered_session_id,
        )


def _required_claim_token(reminder: JarvisReminder):
    if reminder.trigger_claim_token is None:
        raise RuntimeError("Reminder does not have an active trigger claim.")
    return reminder.trigger_claim_token


def _response_output_items(output: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in output or []:
        if isinstance(item, dict):
            items.append(item)
        elif hasattr(item, "model_dump"):
            items.append(item.model_dump(exclude_none=True))
    return items


def _response_function_calls(output: Any) -> list[dict[str, Any]]:
    return [item for item in _response_output_items(output) if item.get("type") == "function_call"]


def _tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if not isinstance(raw_arguments, str):
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
