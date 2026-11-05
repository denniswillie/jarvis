"""Live invalidation, explicit clearing and canonical reconnect evidence."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from types import SimpleNamespace
import unittest
from uuid import uuid4

from app.assistant.context import AssistantContext
from app.assistant.context_providers import merge_behavior_instructions
from app.realtime.continuity import recover_voice_context
from app.realtime.events import ClientEventError
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from tests.test_live_response_source_fence import LiveResponseSourceDatabaseTests


def context(messages=()):
    return AssistantContext(current_utc_time=datetime.now(UTC), session=None,
        behavior=merge_behavior_instructions(persistent="", session_override=""),
        recent_messages=list(messages), active_reminders=[], preferences=[], recent_visual_interrupts=[])


async def until(predicate, timeout=5):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class ContextRecoveryUnitTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        async def send_text(text):
            self.events.append(json.loads(text))
        async def close(**kwargs):
            pass
        self.voice = RealtimeAudioSession(websocket=SimpleNamespace(send_text=send_text, close=close),
            context=SimpleNamespace(owner_id="fixture", jarvis_session_id=uuid4(), audio_session_id="fixture-audio"),
            settings=JarvisSettings(_env_file=None), repository=SimpleNamespace(),
            tool_modules=(), context_providers=())
        self.addAsyncCleanup(self.voice.close, "fixture_complete")

    async def test_clear_ack_is_exact_and_retirement_is_single_owned_task(self):
        self.voice._begin_context_rebuild()
        task = self.voice._context_retirement_task
        identity = self.voice._context_reset_id
        self.voice._begin_context_rebuild()
        self.assertIs(self.voice._context_retirement_task, task)
        await until(lambda: self.events)
        self.assertEqual(self.events[0]["type"], "session.context_invalidated")
        self.assertFalse(self.voice._closed.is_set())
        valid = {"type":"session.context_cleared", "protocol_version":1,
            "reset_id":identity, "audio_session_id":"fixture-audio"}
        for change in [{"protocol_version":True}, {"reset_id":str(uuid4())}, {"audio_session_id":"other"}]:
            with self.assertRaises(ClientEventError):
                await self.voice._handle_client_event(valid | change)
        self.assertFalse(self.voice._context_reset_ack.is_set())
        await self.voice._handle_client_event(valid)
        await asyncio.wait_for(task, 1)
        self.assertTrue(self.voice._context_reset_ack.is_set())
        self.assertTrue(self.voice._closed.is_set())

    async def test_queued_media_does_not_cross_invalidation_notice(self):
        sender = self.voice._app_sender
        await sender._lock.acquire()
        media = asyncio.create_task(self.voice._send_app_event("audio.output.delta",
            response_id="old", item_id="old", audio_base64="AAAA"))
        await asyncio.sleep(0)
        self.voice._begin_context_rebuild()
        sender._lock.release()
        self.assertFalse(await media)
        await until(lambda: self.events)
        self.assertEqual([e["type"] for e in self.events], ["session.context_invalidated"])

    async def test_capture_failure_cannot_prevent_retirement(self):
        class BrokenCapture:
            def context_reset(self, *args, **kwargs):
                raise RuntimeError("Synthetic journal failure")
            async def close(self):
                pass
        self.voice._context_capture = BrokenCapture()
        self.voice._begin_context_rebuild()
        self.voice._context_reset_ack.set()
        await asyncio.wait_for(self.voice._context_retirement_task, 1)
        self.assertTrue(self.voice._closed.is_set())

    async def test_no_source_watch_has_no_database_activity_and_wakes_on_first_source(self):
        sources = LiveResponseSources(None, owner_id="fixture", session_id=uuid4(), audio_session_id="a")
        task = asyncio.create_task(sources.wait_for_invalidation(interval=10))
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        sources.observe([uuid4()])
        with self.assertRaises(LiveContextError):
            await asyncio.wait_for(task, 1)

    async def test_new_source_wakeup_does_not_wait_for_poll_interval(self):
        sources = LiveResponseSources(None, owner_id="fixture", session_id=uuid4(), audio_session_id="a")
        checks = 0
        entered = asyncio.Event()
        @asynccontextmanager
        async def guard():
            nonlocal checks
            checks += 1
            entered.set()
            if checks > 1:
                raise LiveContextError()
            yield
        sources.context_guard = guard
        sources.observe([uuid4()])
        task = asyncio.create_task(sources.wait_for_invalidation(interval=10))
        await entered.wait()
        sources.observe([uuid4()])
        with self.assertRaises(LiveContextError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(checks, 2)

    async def test_empty_recovery_preserves_context_without_inventing_history(self):
        rebuilt = recover_voice_context(context(), {"total":0, "records":[]})
        self.assertEqual(rebuilt.voice_continuity["records"], [])
        self.assertEqual(rebuilt.source_payload_ids, ())


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class ContextRecoveryDatabaseTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = LiveResponseSourceDatabaseTests.asyncSetUp
    voice = LiveResponseSourceDatabaseTests.voice
    remove = LiveResponseSourceDatabaseTests.remove

    async def test_running_session_detects_removal_accepts_clear_ack_and_closes_transport(self):
        voice, _, phone = self.voice()
        inbox = asyncio.Queue()
        async def incoming():
            while True:
                yield await inbox.get()
        async def initialize():
            voice._realtime_context_ready.set()
        async def provider_events():
            await asyncio.Event().wait()
        voice.websocket.iter_text = incoming
        voice._initialize_openai_session = initialize
        voice.openai.receive_event = provider_events
        running = asyncio.create_task(voice.run())
        try:
            await until(lambda: any(e["type"]=="session.created" for e in phone))
            await self.remove()
            await until(lambda: any(e["type"]=="session.context_invalidated" for e in phone), 8)
            self.assertTrue(voice._context_revoked)
            notice = next(e for e in phone if e["type"]=="session.context_invalidated")
            await inbox.put(json.dumps({"type":"session.context_cleared", "protocol_version":1,
                "reset_id":notice["context_reset"]["id"], "audio_session_id":"fixture-audio"}))
            await asyncio.wait_for(running, 5)
            self.assertIsNone(voice.openai._websocket)
            closed = next(e for e in phone if e["type"]=="session.closed")
            self.assertEqual(closed["context_reset"], notice["context_reset"])
            self.assertEqual((await self.messages.get_session(self.f.owner, self.session.id)).status, "active")
            history = await voice._context_capture.repository.voice_history(self.f.owner, self.session.id)
            resets = [r for r in history["records"] if (r.content or {}).get("reason")=="source_context_rebuild"]
            self.assertEqual(len(resets), 1)
            self.assertTrue(resets[0].content["client_reported_audio_cleared"])
        finally:
            if not running.done():
                running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    async def test_missing_ack_is_not_reported_as_client_clearing(self):
        voice, _, _ = self.voice()
        voice._begin_context_rebuild()
        await asyncio.wait_for(voice._context_retirement_task, 3)
        self.assertFalse(voice._context_reset_ack.is_set())
        await voice.close("fixture_complete")
        history = await voice._context_capture.repository.voice_history(self.f.owner, self.session.id)
        reset = next(r for r in history["records"] if (r.content or {}).get("reason")=="source_context_rebuild")
        self.assertFalse(reset.content["client_reported_audio_cleared"])

    async def test_recovery_retains_interrupted_words_missing_input_and_exact_sources(self):
        voice, _, _ = self.voice()
        capture = voice._context_capture
        capture.input_admitted("unfinished")
        capture.observe_output({"type":"transcript.delta", "item_id":"independent", "response_id":"earlier",
            "text":"Independent partial answer."}, forwarded=True)
        await capture.close()
        history = await capture.repository.voice_history(self.f.owner, self.session.id)
        rebuilt = recover_voice_context(context(), history)
        records = rebuilt.voice_continuity["records"]
        output = next(r for r in records if r["kind"]=="assistant_output")
        missing = next(r for r in records if r["kind"]=="user_transcript")
        self.assertEqual(output["content"]["completion_state"], "interrupted")
        self.assertIsNone(output["content"]["heard_transcript"])
        self.assertEqual(missing["content"]["transcript_status"], "session_closed_without_transcript")
        self.assertIsNone(missing["content"]["text"])
        self.assertEqual(set(rebuilt.source_payload_ids), {r.payload_id for r in history["records"]})
        self.assertTrue(rebuilt.voice_continuity["meaning"])

    async def test_deleted_records_and_character_budget_do_not_reconstruct_content(self):
        voice, _, _ = self.voice()
        capture = voice._context_capture
        capture.observe_output({"type":"transcript.done", "item_id":"secret", "response_id":"earlier",
            "text":"Synthetic removable words."}, forwarded=True)
        await capture.close()
        history = await capture.repository.voice_history(self.f.owner, self.session.id)
        row = history["records"][0]
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_context_payloads set availability='deleted',content=null where owner_id=$1 and id=$2",
                            self.f.owner, row.payload_id)
        rebuilt = recover_voice_context(context(), await capture.repository.voice_history(self.f.owner, self.session.id))
        self.assertNotIn("Synthetic removable words.", json.dumps(rebuilt.voice_continuity))
        self.assertEqual(rebuilt.source_payload_ids, ())
        tiny = recover_voice_context(context(), history, max_characters=1)
        self.assertEqual(tiny.voice_continuity["records"], [])
        self.assertEqual(tiny.voice_continuity["omitted_records"], 1)

    async def test_camera_observations_do_not_displace_voice_history(self):
        voice, _, _ = self.voice()
        repo = voice._context_capture.repository
        for i in range(3):
            await repo.append(self.f.owner, ContextIngress(session_id=self.session.id, source_namespace="voice:fixture",
                source_event_id=str(i), receipt_id=str(i), kind="observation", trust="sensor", payload={"frame":i}))
        voice._context_capture.input_final("speech", "Synthetic conversational history.")
        await voice._context_capture.close()
        history = await repo.voice_history(self.f.owner, self.session.id, limit=1)
        self.assertEqual(history["total"], 1)
        self.assertEqual(history["records"][0].kind, "user_transcript")
        with self.assertRaises(Exception):
            await repo.voice_history(self.f.other, self.session.id)

