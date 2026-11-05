from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.models import RealtimeResponseState, RealtimeSessionContext
from app.realtime.output_policy import RealtimeOutputItemState
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings


def _session_for_reminder_test() -> RealtimeAudioSession:
    session = RealtimeAudioSession(
        websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
        context=RealtimeSessionContext(
            audio_session_id="reminder-fixture", owner_id="owner",
            jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace(),
        ),
        settings=JarvisSettings(_env_file=None, OPENAI_API_KEY="offline-fixture-not-a-provider-key"),
        repository=SimpleNamespace(), tool_modules=(), context_providers=(),
        reminder_repository=SimpleNamespace(mark_event_delivery=AsyncMock()),
        provider_connection=SimpleNamespace(close=AsyncMock()),
    )
    session._send_app_event = AsyncMock()
    return session


class RealtimeReminderLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_persistence_failure_does_not_fail_completed_response(self) -> None:
        session = _session_for_reminder_test()
        self.addAsyncCleanup(session.close)
        reminder_id = uuid4()
        event_id = uuid4()
        session._pending_reminder_events[str(event_id)] = str(reminder_id)
        session.reminder_repository.mark_event_delivery.side_effect = RuntimeError("db unavailable")
        state = RealtimeResponseState(
            response_id="response-1",
            kind="reminder",
            output_items={
                "answer": RealtimeOutputItemState(
                    item_id="answer", phase="final_answer",
                    delivery_disposition="forwarded", forwarded_audio_chunks=1,
                ),
            },
            metadata={
                "reminder_id": str(reminder_id),
                "reminder_event_id": str(event_id),
            },
        )

        await session._finish_reminder_response(
            state=state,
            response={"status": "completed"},
            response_id="response-1",
        )

        session._send_app_event.assert_awaited_once_with(
            "reminder.done",
            reminder_id=str(reminder_id),
            reminder_event_id=str(event_id),
            response_id="response-1",
        )
        self.assertIn(str(event_id), session._pending_reminder_events)

    async def test_close_marks_pending_reminder_interrupted(self) -> None:
        session = _session_for_reminder_test()
        self.addAsyncCleanup(session.close)
        reminder_id = uuid4()
        event_id = uuid4()
        session._pending_reminder_events[str(event_id)] = str(reminder_id)
        await session.close("client_disconnect")

        session.reminder_repository.mark_event_delivery.assert_awaited_once_with(
            owner_id="owner",
            event_id=event_id,
            event_type="interrupted",
            session_id=session.context.jarvis_session_id,
            metadata={
                "delivery_route": "realtime",
                "reason": "session_closed",
                "session_close_reason": "client_disconnect",
            },
        )
        self.assertNotIn(str(event_id), session._pending_reminder_events)
        session.openai.close.assert_awaited_once()
        session.websocket.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
