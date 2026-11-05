from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.assistant.context_providers import ContextBuildRequest
from app.assistant.memory_context import (
    MAX_CONTEXT_SEQUENCE_ITEMS,
    MAX_CONTEXT_STRING_CHARACTERS,
    WorkingMemoryContextProvider,
    working_memory_app_event_payload,
)
from app.perception.models import OwnerWorkingMemoryState, SessionWorkingMemoryState
from app.sessions.models import JarvisSession


NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


class _WorkingMemoryService:
    def __init__(self) -> None:
        self.session_id = uuid4()

    async def get_session_working_memory(self, **_):
        return SessionWorkingMemoryState(
            session_id=self.session_id,
            owner_id="owner",
            revision=7,
            last_observation_id=uuid4(),
            last_observation_ended_at=NOW,
            state={
                "scene": {"label": "kitchen"},
                "activity": {"label": "making tea"},
                "summary": "x" * (MAX_CONTEXT_STRING_CHARACTERS + 20),
                "visible_text": list(range(MAX_CONTEXT_SEQUENCE_ITEMS + 5)),
                "image_bytes": b"not prompt-safe",
            },
            expires_at=NOW + timedelta(minutes=2),
            created_at=NOW,
            updated_at=NOW,
        )

    async def get_owner_working_memory(self, **_):
        return OwnerWorkingMemoryState(
            owner_id="owner",
            revision=3,
            state={"active_visual_focus": {"value": "find keys"}},
            expires_at=None,
            created_at=NOW,
            updated_at=NOW,
        )


class MemoryContextProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_is_structured_bounded_and_contains_no_pixels(self) -> None:
        service = _WorkingMemoryService()
        session = _session(service.session_id)
        section = await WorkingMemoryContextProvider(service).build_section(
            ContextBuildRequest(
                owner_id="owner",
                session=session,
                recent_messages=(),
                requested_at_utc=NOW,
            )
        )

        session_payload = section.payload["session"]
        state = session_payload["state"]
        self.assertEqual(session_payload["revision"], 7)
        self.assertEqual(len(state["summary"]), MAX_CONTEXT_STRING_CHARACTERS)
        self.assertEqual(len(state["visible_text"]), MAX_CONTEXT_SEQUENCE_ITEMS)
        self.assertNotIn("image_bytes", state)
        self.assertEqual(section.payload["owner"]["revision"], 3)
        self.assertEqual(
            working_memory_app_event_payload(section.payload),
            {
                "revision": 7,
                "owner_revision": 3,
                "session_available": True,
                "owner_available": True,
                "observed_at": NOW.isoformat(),
                "expires_at": (NOW + timedelta(minutes=2)).isoformat(),
                "scene_summary": "kitchen",
                "activity_summary": "making tea",
            },
        )

    async def test_missing_service_returns_explicit_empty_context(self) -> None:
        section = await WorkingMemoryContextProvider(None).build_section(
            ContextBuildRequest(
                owner_id="owner",
                session=_session(uuid4()),
                recent_messages=(),
                requested_at_utc=NOW,
            )
        )
        self.assertEqual(
            section.payload,
            {"available": False, "session": None, "owner": None},
        )


def _session(session_id) -> JarvisSession:
    return JarvisSession(
        id=session_id,
        title="Test",
        status="active",
        created_at=NOW,
        updated_at=NOW,
        closed_at=None,
        last_activity_at=NOW,
        behavior_instructions="",
        metadata={},
    )
