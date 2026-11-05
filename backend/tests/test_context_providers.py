from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.assistant.context import AssistantContextBuildError, AssistantContextBuilder
from app.assistant.context_providers import (
    ActiveRemindersContextProvider,
    AssistantContextProvider,
    BehaviorContextProvider,
    ContextBuildRequest,
    ContextSection,
    CurrentTimeContextProvider,
    PreferencesContextProvider,
    RecentMessagesContextProvider,
    RecentVisualInterruptsContextProvider,
    SessionContextProvider,
)
from app.sessions.models import JarvisMessage, JarvisSession


NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


class _BehaviorRepository:
    async def get(self, *, owner_id: str):
        return SimpleNamespace(behavior_instructions="Keep answers concise.")


class _ReminderRepository:
    async def list_active_reminders(self, *, owner_id: str):
        return [
            SimpleNamespace(
                id=uuid4(),
                title=f"Reminder {index}",
                details="Details",
                remind_at=None,
                condition=None,
            )
            for index in range(25)
        ]


class _PreferenceRepository:
    async def search_preferences(self, *, owner_id: str, query, limit: int):
        return [
            SimpleNamespace(key="drink", value="tea", source="assistant")
        ]


class _VisualRepository:
    async def list_recent_visual_interrupts(self, **kwargs):
        return [
            SimpleNamespace(
                message="Your keys are still on the desk.",
                reason="leaving_keys",
                confidence=0.98,
                created_at=NOW,
            )
        ]


@dataclass(frozen=True)
class _TestProvider:
    key: str
    priority: int
    payload: object
    critical: bool = False
    fail: bool = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        if self.fail:
            raise RuntimeError("provider failed")
        return ContextSection(
            key=self.key,
            payload=self.payload,
            priority=self.priority,
            freshness=request.requested_at_utc,
            size=1,
        )

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return ContextSection(
            key=self.key,
            payload=[],
            priority=self.priority,
            freshness=request.requested_at_utc,
            size=0,
        )


class ContextProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_context_keeps_legacy_data_out_of_prompt_payload(self) -> None:
        await _check_provider_context_keeps_legacy_data_out_of_prompt_payload()

    async def test_optional_provider_failure_isolated_and_order_is_deterministic(self) -> None:
        await _check_optional_provider_failure_isolated_and_order_is_deterministic()

    async def test_critical_provider_failure_rejects_context(self) -> None:
        await _check_critical_provider_failure_rejects_context()

    async def test_recent_messages_provider_keeps_latest_bounded_items(self) -> None:
        await _check_recent_messages_provider_keeps_latest_bounded_items()


async def _check_provider_context_keeps_legacy_data_out_of_prompt_payload() -> None:
    session = _session()
    message = _message(session.id, "Where are my keys?")
    providers: list[AssistantContextProvider] = [
        CurrentTimeContextProvider(),
        SessionContextProvider(),
        BehaviorContextProvider(_BehaviorRepository()),
        RecentMessagesContextProvider(max_items=12),
        ActiveRemindersContextProvider(_ReminderRepository()),
        PreferencesContextProvider(_PreferenceRepository()),
        RecentVisualInterruptsContextProvider(_VisualRepository()),
    ]

    context = await AssistantContextBuilder(providers=providers).build(
        owner_id="owner",
        session=session,
        recent_messages=[message],
        current_utc_time=NOW,
    )
    payload = context.as_prompt_payload()

    assert list(payload) == [
        "current_utc_time",
        "session",
        "behavior_instructions",
        "recent_messages",
        "active_reminders",
        "preferences",
        "working_memory",
        "recent_visual_interrupts",
        "prospective_memory",
    ]
    assert payload["current_utc_time"] == NOW.isoformat()
    assert payload["session"] == {"id": str(session.id), "title": session.title}
    assert payload["behavior_instructions"]["persistent"] == "Keep answers concise."
    assert len(payload["active_reminders"]) == 20
    assert "recent_visual_context" not in payload


async def _check_optional_provider_failure_isolated_and_order_is_deterministic() -> None:
    providers: list[AssistantContextProvider] = [
        _TestProvider("later", 90, "later"),
        _TestProvider("optional", 50, "ignored", fail=True),
        _TestProvider("earlier", 40, "earlier"),
    ]

    sections = await AssistantContextBuilder(providers=providers).build_sections(
        owner_id="owner",
        session=None,
        recent_messages=[],
        current_utc_time=NOW,
    )

    assert [section.key for section in sections] == ["earlier", "optional", "later"]
    assert sections[1].payload == []


async def _check_critical_provider_failure_rejects_context() -> None:
    provider = _TestProvider("identity", 10, None, critical=True, fail=True)

    try:
        await AssistantContextBuilder(providers=[provider]).build_sections(
            owner_id="owner",
            session=None,
            recent_messages=[],
            current_utc_time=NOW,
        )
    except AssistantContextBuildError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("Critical provider failure did not reject context construction.")


async def _check_recent_messages_provider_keeps_latest_bounded_items() -> None:
    session = _session()
    messages = [_message(session.id, f"message-{index}") for index in range(5)]
    sections = await AssistantContextBuilder(
        providers=[RecentMessagesContextProvider(max_items=2)]
    ).build_sections(
        owner_id="owner",
        session=session,
        recent_messages=messages,
        current_utc_time=NOW,
    )

    assert [message.content for message in sections[0].payload] == ["message-3", "message-4"]
    assert sections[0].size == 2


def _session() -> JarvisSession:
    return JarvisSession(
        id=uuid4(),
        title="Walk home",
        status="active",
        created_at=NOW,
        updated_at=NOW,
        closed_at=None,
        last_activity_at=NOW,
        behavior_instructions="Keep this session quiet.",
        metadata={},
    )


def _message(session_id, content: str) -> JarvisMessage:
    return JarvisMessage(
        id=uuid4(),
        session_id=session_id,
        role="user",
        modality="audio",
        content=content,
        provider_item_id=None,
        provider_response_id=None,
        created_at=NOW,
        metadata={},
    )
