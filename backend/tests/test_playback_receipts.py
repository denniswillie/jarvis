"""Client playback telemetry is scoped evidence, never inferred human hearing."""
import asyncio
import base64
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.realtime.events import ClientEventError
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from tests import test_attention_voice as voice_tests


class NativePlaybackReceiptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.capture_store = SimpleNamespace(append=AsyncMock())
        session_id = uuid4()
        self.capture = SessionContextCapture(self.capture_store, owner_id="fixture-owner", session_id=session_id,
                                             provider="openai", audio_session_id="fixture-audio")
        self.session = RealtimeAudioSession(websocket=SimpleNamespace(send_text=AsyncMock(), close=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="fixture-audio", owner_id="fixture-owner",
                jarvis_session_id=session_id, client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None), repository=SimpleNamespace(create_message=AsyncMock()),
            tool_modules=(), context_providers=(), context_capture=self.capture)
        self.session.openai = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.addAsyncCleanup(self.session.close, "fixture_cleanup")

    async def media(self, *, item="item", response="response", complete=True):
        event = {"type":"audio.output.delta", "item_id":item, "response_id":response,
                 "audio_base64":base64.b64encode(bytes(960)).decode("ascii")}
        self.capture.observe_output(event, forwarded=True)
        await self.session._send_app_event(event.pop("type"), **event)
        if complete:
            await self.session._send_app_event("audio.output.done", item_id=item, response_id=response)

    async def receipt(self, *, state="completed", item="item", response="response", samples=480, **extra):
        await self.session._handle_client_event({"type":"playback."+state, "item_id":item,
            "response_id":response, "played_samples":samples, "audio_session_id":"fixture-audio",
            "protocol_version":1, "sample_rate":24000, **extra})

    async def test_progress_does_not_release_voice_but_completion_and_duplicate_are_safe(self):
        await self.media()
        await self.receipt(state="started", samples=240)
        self.assertIn("item", self.session._native_playback)
        await self.receipt(state="progress", samples=480)
        self.assertIn("item", self.session._native_playback)
        await self.receipt()
        self.assertNotIn("item", self.session._native_playback)
        await self.media(item="new", response="new-response")
        await self.receipt()
        self.assertIn("new", self.session._native_playback)
        self.session.openai.send_event.assert_not_awaited()

    async def test_session_sample_and_response_validation(self):
        await self.media()
        for changes in ({"audio_session_id":"other"}, {"sample_rate":48000}, {"protocol_version":True},
                        {"response":"other"}, {"samples":481}, {"samples":True}, {"content_index":False}):
            with self.subTest(changes=changes), self.assertRaises(ClientEventError):
                await self.receipt(**changes)
        self.assertIn("item", self.session._native_playback)

    async def test_played_samples_without_generation_done_cannot_complete(self):
        await self.media(complete=False)
        with self.assertRaises(ClientEventError):
            await self.receipt()
        await self.receipt(state="progress")
        self.assertIn("item", self.session._native_playback)

    async def test_cancelled_output_keeps_receipt_identity_without_blocking_new_output(self):
        await self.media()
        await self.session._handle_client_event({"type":"response.cancel"})
        await self.media(item="new", response="new-response")
        await self.receipt(state="interrupted", samples=240)
        self.assertIn("new", self.session._native_playback)
        self.assertNotIn("item", self.session._native_playback)

    async def test_actual_receipt_updates_context_without_inventing_heard_words(self):
        await self.media()
        await self.receipt()
        await self.capture.flush()
        events = [call.args[1] for call in self.capture_store.append.await_args_list]
        boundary = next(e for e in events if e.kind == "played_output_boundary")
        self.assertEqual(boundary.payload["played_samples"], 480)
        self.assertEqual(boundary.payload["source"], "client_av_audio_data_played_back")
        self.assertIsNone(boundary.payload["heard_transcript"])
        outputs = [e for e in events if e.kind == "assistant_output"]
        self.assertEqual(outputs[-1].payload["playback_status"], "client_reported_completed")
        await self.receipt(state="progress")
        self.assertEqual(self.capture._outputs["item"].playback_state, "client_reported_completed")

    async def test_done_without_forwarded_audio_creates_no_playback_debt(self):
        await self.session._send_app_event("audio.output.done", item_id="empty", response_id="r")
        self.assertNotIn("empty", self.session._native_playback)


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"), "Explicit manifested database required")
class ExpertPlaybackReceiptTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = voice_tests.AttentionVoiceDatabaseTests.asyncSetUp
    complete = voice_tests.AttentionVoiceDatabaseTests.complete
    deliver = voice_tests.AttentionVoiceDatabaseTests.deliver

    async def test_source_deletion_does_not_deadlock_with_delivery_entry(self):
        task, identity = await self.complete()
        async with self.f.pool.acquire() as deletion, self.f.pool.acquire() as reader:
            pid = await reader.fetchval("select pg_backend_pid()")
            async def enter():
                async with reader.transaction():
                    return await self.inbox._entry(reader, self.f.owner, identity, lock=True)
            pending = None
            try:
                async with deletion.transaction():
                    await deletion.execute("select id from jarvis_context_payloads where id=$1 for update", task.result_payload_id)
                    pending = asyncio.create_task(enter())
                    async with asyncio.timeout(5):
                        while not await deletion.fetchval("select exists(select 1 from pg_locks where pid=$1 and not granted)", pid):
                            await asyncio.sleep(0.02)
                    await deletion.execute("set local lock_timeout='500ms'")
                    await deletion.execute("update jarvis_context_payloads set content=null,availability='deleted' where id=$1", task.result_payload_id)
                row, _ = await asyncio.wait_for(pending, 5)
                self.assertEqual(row["state"], "revoked")
            finally:
                if pending is not None and not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)

    async def test_progress_preserves_expert_speech_and_completion_records_the_exact_items(self):
        _, identity = await self.complete()
        delivery, _ = await self.deliver()
        await asyncio.wait_for(self.generated.wait(), 10)
        for state in ("started", "progress"):
            await self.voice._handle_client_event({"type":"playback."+state,
                "response_id":delivery.response_id, "item_id":delivery.item_id(0), "played_samples":480})
            self.assertFalse(delivery.revoked)
            self.assertIs(self.voice._attention_delivery, delivery)
        for index in (0, 1):
            await self.voice._handle_client_event({"type":"playback.completed",
                "response_id":delivery.response_id, "item_id":delivery.item_id(index), "played_samples":480})
        await asyncio.wait_for(delivery.task, 10)
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "played")
        self.assertEqual(detail["attempts"][0]["played_samples"], 960)
