from __future__ import annotations

import unittest
from datetime import UTC, datetime
from uuid import uuid4

from app.realtime.session_manager import RealtimeSessionManager


class _ReminderSession:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def speak_reminder(self, *, message: str, **_: object) -> None:
        self.messages.append(message)


class RealtimeSessionManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_owner_routing_falls_back_after_newest_session_disconnects(self) -> None:
        manager = RealtimeSessionManager()
        first_session_id = uuid4()
        second_session_id = uuid4()
        first = _ReminderSession()
        second = _ReminderSession()

        self.assertTrue(
            await manager.register(
                audio_session_id="audio-1",
                owner_id="owner",
                jarvis_session_id=first_session_id,
                session=first,
            )
        )
        self.assertTrue(
            await manager.register(
                audio_session_id="audio-2",
                owner_id="owner",
                jarvis_session_id=second_session_id,
                session=second,
            )
        )

        selected = await manager.speak_reminder(
            owner_id="owner",
            preferred_jarvis_session_id=None,
            message="newest",
            reminder_id=uuid4(),
            reminder_event_id=uuid4(),
        )
        self.assertEqual(selected, second_session_id)
        self.assertEqual(second.messages, ["newest"])

        await manager.unregister(
            audio_session_id="audio-2",
            owner_id="owner",
            jarvis_session_id=second_session_id,
        )
        selected = await manager.speak_reminder(
            owner_id="owner",
            preferred_jarvis_session_id=None,
            message="fallback",
            reminder_id=uuid4(),
            reminder_event_id=uuid4(),
        )
        self.assertEqual(selected, first_session_id)
        self.assertEqual(first.messages, ["fallback"])

    async def test_session_diagnostics_are_redacted_and_track_connection(self) -> None:
        manager = RealtimeSessionManager()
        session_id = uuid4()
        session = _ReminderSession()

        disconnected = await manager.get_session_diagnostics(session_id)
        self.assertEqual(
            disconnected,
            {"realtime_connected": False, "audio_session_id": None},
        )

        await manager.register(
            audio_session_id="audio-diagnostic",
            owner_id="owner",
            jarvis_session_id=session_id,
            session=session,
        )
        connected = await manager.get_session_diagnostics(session_id)
        self.assertEqual(
            connected,
            {
                "realtime_connected": True,
                "audio_session_id": "audio-diagnostic",
            },
        )




class _VisualSession:
    def __init__(self, accepted: bool) -> None:
        self.accepted = accepted
        self.calls: list[dict[str, object]] = []

    async def speak_visual_interrupt(self, **values: object) -> bool:
        self.calls.append(values)
        return self.accepted


class GenericVisualRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_rejection_is_propagated(self) -> None:
        manager = RealtimeSessionManager()
        session_id = uuid4()
        session = _VisualSession(False)
        await manager.register(
            audio_session_id="audio", owner_id="owner",
            jarvis_session_id=session_id, session=session,
        )
        accepted = await manager.speak_visual_interrupt(
            owner_id="owner", audio_session_id="audio", jarvis_session_id=session_id,
            source_received_at=datetime.now(UTC), message="A new observation.",
            interrupt_id=uuid4(),
        )
        self.assertFalse(accepted)
        self.assertEqual(len(session.calls), 1)

    async def test_fallback_is_limited_to_the_same_owner_and_conversation(self) -> None:
        manager = RealtimeSessionManager()
        target_id, other_id = uuid4(), uuid4()
        target, other = _VisualSession(True), _VisualSession(True)
        await manager.register(
            audio_session_id="target-audio", owner_id="owner",
            jarvis_session_id=target_id, session=target,
        )
        await manager.register(
            audio_session_id="other-audio", owner_id="other-owner",
            jarvis_session_id=other_id, session=other,
        )
        for supplied_id in ("disconnected-audio", "other-audio"):
            accepted = await manager.speak_visual_interrupt(
                owner_id="owner", audio_session_id=supplied_id,
                jarvis_session_id=target_id, source_received_at=datetime.now(UTC),
                message="Only for the requested conversation.", interrupt_id=uuid4(),
            )
            self.assertTrue(accepted)
        self.assertEqual(len(target.calls), 2)
        self.assertEqual(other.calls, [])
        for owner, target_session in (("other-owner", target_id), ("owner", other_id)):
            self.assertFalse(await manager.speak_visual_interrupt(
                owner_id=owner, audio_session_id="target-audio",
                jarvis_session_id=target_session, source_received_at=datetime.now(UTC),
                message="Do not redirect.", interrupt_id=uuid4(),
            ))
        self.assertEqual(len(target.calls), 2)
        self.assertEqual(other.calls, [])

if __name__ == "__main__":
    unittest.main()
