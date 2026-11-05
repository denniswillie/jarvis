from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.companion import CompanionState
from app.live_state.companion_notices import CompanionNotices
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.live_session import LiveAudioSession
from app.attention.quiet_delivery import QuietDeliverySuppressed
from test_companion_state import Memory, Reminders
from test_live_voice_adapter import LiveFixture, NOW


class CompanionNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 11, 4, tzinfo=timezone.utc)
        self.repo = Reminders()
        self.state = CompanionState(owner_id="fixture", working_memory=Memory(),
                                    reminders=self.repo, clock=lambda: self.now)
        await self.state.initialize()
        self.row = await self.state.manage_reminder(action="create", title="Take umbrella", condition="Leaving")
        self.repo.rows[self.row["id"]]["updated_at"] = self.now.isoformat()
        await self.state.refresh_reminders()
        self.match = NS(reminder_id=self.row["id"], reminder_revision=self.now.isoformat(),
                        evidence_at=self.now, description="The user is leaving.")
        self.manager = NS(send_companion_notice=AsyncMock(return_value={"status": "provider_received"}))
        self.notices = CompanionNotices(self.state, self.manager)

    async def test_one_notice_does_not_complete_reminder(self):
        await self.notices.on_reminder(self.match, (), 0)
        self.now += timedelta(minutes=1)
        await self.notices.on_reminder(self.match, (), 0)
        self.manager.send_companion_notice.assert_awaited_once()
        self.assertEqual(self.state.read()["system"]["reminders"][0]["status"], "active")

    async def test_visual_notice_contains_detection_not_just_watch_request(self):
        self.repo.rows[self.row["id"]]["title"] = "Alert me when the Tin Man statue is in view"
        await self.state.refresh_reminders()
        self.match.description = "A silver Tin Man statue is visible beside the wall."
        await self.notices.on_reminder(self.match, (), 0)
        notice = self.manager.send_companion_notice.await_args.args[1]
        self.assertIn("Reminder condition matched:", notice["text"])
        self.assertIn(self.match.description, notice["text"])
        self.assertIn(self.now.isoformat(), notice["text"])
        self.assertEqual(notice["evidence_description"], self.match.description)

    async def test_detection_text_is_bounded(self):
        self.match.description = "x" * 10000
        await self.notices.on_reminder(self.match, (), 0)
        notice = self.manager.send_companion_notice.await_args.args[1]
        self.assertEqual(len(notice["evidence_description"]), 600)

    async def test_unready_state_does_not_send_or_replay_expired_evidence(self):
        self.state._ready = False
        await self.notices.on_reminder(self.match, (), 0)
        self.now += timedelta(seconds=61)
        self.state._ready = True
        await self.notices.on_reminder(self.match, (), 0)
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_replaced_or_forgotten_evidence_cannot_send(self):
        self.match.reminder_revision = "obsolete"
        await self.notices.on_reminder(self.match, (), 0)
        self.match.reminder_revision = self.now.isoformat()
        await self.notices.on_reminder(self.match, (), 2)
        self.manager.send_companion_notice.assert_not_awaited()

    async def test_manager_does_not_start_voice_or_send_to_another_owner(self):
        manager = RealtimeSessionManager()
        session = NS(ready=True, speak_companion_notice=AsyncMock(return_value={"status": "provider_received"}))
        await manager.register(audio_session_id="one", owner_id="other", jarvis_session_id=uuid4(), session=session)
        self.assertEqual((await manager.send_companion_notice("fixture", {}))["status"], "unavailable")
        session.speak_companion_notice.assert_not_awaited()

    async def test_current_session_uses_notice_adapter_and_readiness_expiry_guards(self):
        class Session(LiveAudioSession):
            @property
            def ready(self):
                return True
        session = object.__new__(Session)
        session.services = NS(companion_state=self.state)
        session.context = NS(owner_id="fixture")
        session.adapter = NS(submit_notice=AsyncMock(return_value={"status": "provider_received"}))
        notice = await self.state.reserve_reminder_notice(reminder_id=self.match.reminder_id,
            revision=self.match.reminder_revision, evidence_at=self.now, generation=0)
        result = await session.speak_companion_notice(notice)
        self.assertEqual(result["status"], "provider_received")
        guard = session.adapter.submit_notice.await_args.kwargs["guard"]
        async with guard():
            pass
        self.state._ready = False
        with self.assertRaises(QuietDeliverySuppressed):
            async with guard():
                self.fail("unready notice reached adapter")
        self.state._ready = True
        self.now += timedelta(seconds=31)
        with self.assertRaises(QuietDeliverySuppressed):
            async with guard():
                self.fail("expired notice reached adapter")


class LiveNoticeWireTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = LiveFixture()
        await self.fixture.start()

    async def asyncTearDown(self):
        await self.fixture.close()

    async def test_notice_uses_data_only_commentary_without_faking_a_delegation(self):
        @asynccontextmanager
        async def allowed():
            yield
        args = dict(notice_id=uuid4(), text="Reminder: Take umbrella",
                    expires_at=NOW + timedelta(seconds=30), guard=allowed)
        result = await self.fixture.route.submit_notice(**args)
        self.assertTrue(result["provider_received"])
        self.assertEqual(result, await self.fixture.route.submit_notice(**args))
        sent = [event for event in self.fixture.wire.socket.sent
                if event["type"] == "session.commentary.append"]
        self.assertEqual(len(sent), 1)
        self.assertIsNone(sent[0]["delegation_id"])
        self.assertEqual(sent[0]["content"],
                         '{"kind":"authorized_reminder","text":"Reminder: Take umbrella"}')

    async def test_notice_guard_refusal_does_not_close_normal_conversation(self):
        @asynccontextmanager
        async def unavailable():
            raise QuietDeliverySuppressed("stale_or_unavailable_notice", "live_hermes_notice")
            yield
        result = await self.fixture.route.submit_notice(notice_id=uuid4(), text="Reminder",
            expires_at=NOW + timedelta(seconds=30), guard=unavailable)
        self.assertFalse(result["provider_received"])
        self.assertFalse(self.fixture.route.requires_fenced_replacement)
