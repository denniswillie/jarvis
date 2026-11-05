"""Real session/writer and manifested storage; synthetic PCM, not live glasses."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.attention.models import AttentionError
from app.attention.tts import SpeechFrame
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.socket_sender import WebSocketEventSender
from app.settings import JarvisSettings
from tests import test_attention_inbox as inbox_tests


class SyntheticSpeech:
    def __init__(self, preparation, *, gate=None):
        self.prepare = preparation.prepare
        self.gate = gate
        self.started = asyncio.Event()
        self.closed = 0
        self.requests = []

    @asynccontextmanager
    async def open_stream(self, request, *, authorize_submission, is_current):
        assert await authorize_submission(request)
        self.requests.append(request)
        self.started.set()
        async def frames():
            if self.gate is not None:
                await self.gate.wait()
            for start in (0, 240):
                if not await is_current():
                    raise AttentionError("attention_source_unavailable", "Fixture source revoked")
                yield SpeechFrame(b"\x00\x00" * 240, start)
        try:
            yield SimpleNamespace(frames=frames, samples_yielded=480, first_bytes_ms=1.0,
                                  provider_request_id="synthetic-tts-request")
        finally:
            self.closed += 1


class AttentionWriterTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_settings_are_independent_and_lazy(self):
        from app.runtime.feature_config import Capability, SecondBrainSettings
        settings = SecondBrainSettings(_env_file=None, enabled="true", attention_enabled="true",
                                      attention_voice_poll_seconds="invalid")
        self.assertTrue(settings.resolve(Capability.ATTENTION, database_url="fixture").configured)
        self.assertFalse(settings.resolve(Capability.ATTENTION_VOICE, database_url="fixture").enabled)

    async def test_empty_queue_still_recovers_crashed_claims(self):
        from app.attention.service import AttentionService
        recovered = asyncio.Event()
        async def recover(connection, owner):
            self.assertEqual(owner, "fixture")
            recovered.set()
        @asynccontextmanager
        async def connection(*args):
            yield SimpleNamespace(fetchval=AsyncMock(return_value=False))
        repository = SimpleNamespace(tasks=SimpleNamespace(transaction=connection),
                                     pool=SimpleNamespace(acquire=connection), _recover=recover)
        service = AttentionService(repository)
        self.addAsyncCleanup(service.close)
        with patch.object(service, "start_voice") as start_voice:
            service.start_maintenance(owner="fixture", poll_seconds=60)
            await asyncio.wait_for(recovered.wait(), 1)
            start_voice.assert_not_called()

    async def test_expert_context_keeps_script_and_playback_uncertainty_distinct(self):
        from uuid import uuid4
        from app.tasks.context_capture import SessionContextCapture
        repository = SimpleNamespace(append=AsyncMock())
        capture = SessionContextCapture(repository, owner_id="fixture-owner", session_id=uuid4(),
                                        provider="openai", audio_session_id="fixture-audio")
        self.addAsyncCleanup(capture.close)
        capture.expert_speech(response_id="tts-response", item_id="tts-item", text="Check directly.",
            samples=480, complete=False, interrupted=True, provenance={"chunk_index":0})
        self.assertTrue(await capture.flush())
        event = repository.append.await_args.args[1]
        self.assertEqual(event.trust, "assistant_output")
        self.assertEqual(event.payload["audio_provider"], "openai_tts")
        self.assertEqual(event.payload["forwarded_audio_bytes"], 960)
        self.assertIsNone(event.payload["heard_transcript"])
        self.assertEqual(event.payload["text_represents"], "expert_script_not_a_verified_heard_transcript")

    async def test_guard_is_acquired_after_writer_and_rechecks_local_cancellation(self):
        events = []
        current = True
        socket = SimpleNamespace(send_text=AsyncMock())
        writer = WebSocketEventSender(socket)
        @asynccontextmanager
        async def guard():
            nonlocal current
            events.append("guard")
            current = False
            yield lambda: events.append("sent")
        await writer._lock.acquire()
        task = asyncio.create_task(writer.send_text("{}", should_send=lambda: current, write_guard=guard))
        await asyncio.sleep(0)
        self.assertEqual(events, [])
        writer._lock.release()
        self.assertFalse(await task)
        self.assertEqual(events, ["guard"])
        socket.send_text.assert_not_awaited()


    async def test_expert_history_ids_fit_native_limit_and_keep_chunk_identity(self):
        from uuid import uuid4
        voice = object.__new__(RealtimeAudioSession)
        voice.context = SimpleNamespace(owner_id="fixture-owner", jarvis_session_id=uuid4(),
                                        audio_session_id="fixture-audio")
        voice._source_context = SimpleNamespace(observe=lambda _sources: None)
        voice.repository = SimpleNamespace(create_message=AsyncMock())
        voice.openai = SimpleNamespace(send_event=AsyncMock(return_value=True))
        @asynccontextmanager
        async def guard(_claim):
            yield object(), None, None, None
        delivery = SimpleNamespace(
            repository=SimpleNamespace(guard=guard), current=lambda: True,
            claim=SimpleNamespace(attempt_id=uuid4(), source_payload_id=uuid4(), intent_id=uuid4()),
            item_id=lambda index: f"fixture-item-{index}", response_id="fixture-response")
        for index in (0, 63, 0):
            await voice.record_expert_speech(delivery, index, "Exact expert words.")
        items = [call.args[0]["item"] for call in voice.openai.send_event.await_args_list]
        self.assertTrue(all(len(item["id"]) == 32 for item in items))
        self.assertEqual(items[0]["id"], items[2]["id"])
        self.assertNotEqual(items[0]["id"], items[1]["id"])
        self.assertTrue(all(item["content"][0]["text"].endswith("Exact expert words.") for item in items))


    async def test_manager_does_not_dispatch_to_a_foreign_owner(self):
        from uuid import uuid4
        manager = RealtimeSessionManager()
        session = SimpleNamespace(try_attention=AsyncMock())
        await manager.register(audio_session_id="foreign", owner_id="other", jarvis_session_id=uuid4(), session=session)
        self.assertIsNone(await manager.deliver_attention(owner_id="owner", repository=None, speech=None,
                                                          worker_id="test", receipt_seconds=5))
        session.try_attention.assert_not_awaited()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class AttentionVoiceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    complete = inbox_tests.AttentionDatabaseTests.complete

    async def asyncSetUp(self):
        await inbox_tests.AttentionDatabaseTests.asyncSetUp(self)
        self.events = []
        self.generated = asyncio.Event()
        async def send(text):
            self.events.append(json.loads(text))
            if len([e for e in self.events if e["type"] == "audio.output.done"]) == 2:
                self.generated.set()
        self.voice = RealtimeAudioSession(
            websocket=SimpleNamespace(send_text=send, close=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="fixture-audio", owner_id=self.f.owner,
                jarvis_session_id=self.session, client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=())
        self.voice.openai = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.voice._realtime_context_ready.set()
        self.addAsyncCleanup(self.voice.close, "fixture_cleanup")

    async def deliver(self, *, speech=None, receipt_seconds=5):
        speech = speech or SyntheticSpeech(self.tts)
        delivery = await self.voice.try_attention(self.inbox, speech, worker_id="fixture-delivery",
                                                   receipt_seconds=receipt_seconds)
        self.assertIsNotNone(delivery)
        return delivery, speech

    async def test_exact_expert_result_reaches_common_writer_and_only_receipts_finish_delivery(self):
        task, identity = await self.complete()
        delivery, speech = await self.deliver()
        await asyncio.wait_for(self.generated.wait(), 10)
        self.assertIs(self.voice._attention_delivery, delivery)
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "awaiting_receipt")
        self.assertIsNone(detail["attempts"][0]["played_samples"])
        for index in range(2):
            await self.voice._handle_client_event({"type":"playback.completed", "item_id":delivery.item_id(index),
                "response_id":delivery.response_id, "played_samples":480, "content_index":0})
        await asyncio.wait_for(delivery.task, 10)
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "played")
        self.assertEqual(detail["attempts"][0]["played_samples"], 960)
        self.assertEqual((await self.tasks.get(self.f.owner, task.id)).result_payload_id, task.result_payload_id)
        self.assertEqual([r.text for r in speech.requests], ["I cannot confirm the switch is off.", "Please check it directly."])
        self.assertEqual([e["text"] for e in self.events if e["type"] == "transcript.done"], [r.text for r in speech.requests])
        self.assertEqual(speech.closed, 2)
        self.assertEqual(len([e for e in self.events if e["type"] == "response.done"]), 1)
        self.assertTrue(all(e["provider"] == "openai" and e["audio_origin"] == "expert_tts"
                            for e in self.events if e["type"].startswith("audio.output")))
        self.assertFalse(any(c.args[0]["type"] == "response.create" for c in self.voice.openai.send_event.await_args_list))

    async def test_revocation_while_waiting_for_common_writer_suppresses_audio(self):
        task, identity = await self.complete()
        await self.voice._app_sender._lock.acquire()
        delivery, speech = await self.deliver()
        try:
            await asyncio.wait_for(speech.started.wait(), 10)
            async with self.f.pool.acquire() as connection:
                await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2", self.f.owner, task.result_payload_id)
        finally:
            self.voice._app_sender._lock.release()
        await asyncio.wait_for(delivery.task, 10)
        self.assertFalse(any(e["type"] == "audio.output.delta" for e in self.events))
        self.assertEqual((await self.inbox.detail(self.f.owner, identity))["state"], "revoked")

    async def test_barge_in_cancels_tts_http_not_the_expert_and_retains_truncate_tombstone(self):
        task, identity = await self.complete()
        speech = SyntheticSpeech(self.tts, gate=asyncio.Event())
        delivery, _ = await self.deliver(speech=speech)
        await asyncio.wait_for(speech.started.wait(), 10)
        await self.voice._handle_client_event({"type":"response.cancel"})
        await asyncio.wait_for(delivery.task, 10)
        await self.voice._handle_client_event({"type":"playback.truncated", "item_id":delivery.item_id(0),
                                             "content_index":0, "audio_end_ms":0})
        self.assertEqual(speech.closed, 1)
        self.assertEqual((await self.tasks.get(self.f.owner, task.id)).status.value, "succeeded")
        self.assertEqual((await self.inbox.detail(self.f.owner, identity))["attempts"][0]["phase"], "interrupted")
        self.assertFalse(any(c.args[0]["type"] == "conversation.item.truncate" for c in self.voice.openai.send_event.await_args_list))

    async def test_missing_receipts_clear_playback_and_leave_hearing_unknown(self):
        _, identity = await self.complete()
        delivery, _ = await self.deliver(receipt_seconds=0.01)
        await asyncio.wait_for(delivery.task, 10)
        detail = await self.inbox.detail(self.f.owner, identity)
        self.assertEqual(detail["state"], "unknown")
        self.assertIsNone(detail["attempts"][0]["played_samples"])
        self.assertEqual(len([e for e in self.events if e["type"] == "audio.output.clear"]), 2)
        self.assertTrue(all(e["provider"] == "openai" for e in self.events if e["type"] == "audio.output.clear"))

    async def test_chunk_identity_is_exact_and_single_use(self):
        await self.complete()
        claim = await self.inbox.claim_next(self.f.owner, self.session, "fixture")
        requests = await self.inbox.prepare(claim, self.tts)
        with self.assertRaises(AttentionError):
            await self.inbox.submit_chunk(claim, 0, replace(requests[0], text="Different text"))
        self.assertTrue(await self.inbox.submit_chunk(claim, 0, requests[0]))
        with self.assertRaises(AttentionError):
            await self.inbox.submit_chunk(claim, 0, requests[0])

    async def test_wrong_response_or_oversized_receipt_does_not_mark_played(self):
        _, identity = await self.complete()
        delivery, _ = await self.deliver()
        await asyncio.wait_for(self.generated.wait(), 10)
        for response, samples in (("wrong", 480), (delivery.response_id, 481)):
            with self.assertRaises(AttentionError):
                await self.voice._handle_client_event({"type":"playback.completed", "item_id":delivery.item_id(0),
                                                     "response_id":response, "played_samples":samples})
        self.assertEqual((await self.inbox.detail(self.f.owner, identity))["state"], "awaiting_receipt")
        await delivery.stop("fixture_complete")
        await delivery.task

    async def test_native_unconfirmed_playback_prevents_expert_overlap(self):
        await self.complete()
        await self.voice._send_app_event("audio.output.delta", item_id="native", response_id="native-response", audio_base64="AAA=")
        await self.voice._send_app_event("audio.output.done", item_id="native", response_id="native-response")
        self.assertIsNone(await self.voice.try_attention(self.inbox, SyntheticSpeech(self.tts),
                                                        worker_id="fixture", receipt_seconds=5))
        await self.voice._handle_client_event({"type":"playback.completed", "item_id":"native",
                                             "response_id":"native-response", "played_samples":1})
        delivery, _ = await self.deliver()
        await delivery.stop("fixture_complete")
        await delivery.task

    async def test_cancellation_before_coroutine_start_settles_without_network(self):
        _, identity = await self.complete()
        delivery, speech = await self.deliver()
        await delivery.stop("user_interrupted")
        await asyncio.wait_for(delivery.task, 10)
        self.assertEqual(speech.requests, [])
        self.assertIsNone(self.voice._attention_delivery)
        self.assertNotEqual((await self.inbox.detail(self.f.owner, identity))["state"], "claimed")

    async def test_history_write_rejects_a_revoked_source(self):
        task, _ = await self.complete()
        delivery, speech = await self.deliver(speech=SyntheticSpeech(self.tts, gate=asyncio.Event()))
        await asyncio.wait_for(speech.started.wait(), 10)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_context_payloads set content=null,availability='deleted' where owner_id=$1 and id=$2", self.f.owner, task.result_payload_id)
        with self.assertRaises(AttentionError):
            await self.voice.record_expert_speech(delivery, 0, "Do not release this revoked text.")
        self.voice.repository.create_message.assert_not_awaited()
        self.voice.openai.send_event.assert_not_awaited()
        await delivery.stop("fixture_complete")
        await delivery.task

    async def test_delivery_guard_locks_the_published_expert_context_manifest(self):
        import asyncpg
        task, _ = await self.complete()
        async with self.f.pool.acquire() as connection:
            payload = await self.tasks.payload(connection, self.f.owner, {"fixture":"expert context"})
            manifest = await connection.fetchval("""insert into jarvis_task_context_manifests
                (owner_id,task_id,task_revision,context_watermark,payload_id) values($1,$2,$3,0,$4) returning id""",
                self.f.owner, task.id, task.revision, payload)
            await connection.execute("""update jarvis_task_attempts set context_manifest_id=$3
                where owner_id=$1 and task_id=$2""", self.f.owner, task.id, manifest)
        claim = await self.inbox.claim_next(self.f.owner, self.session, "fixture")
        async with self.inbox.guard(claim):
            async with self.f.pool.acquire() as connection:
                with self.assertRaises(asyncpg.LockNotAvailableError):
                    async with connection.transaction():
                        await connection.execute("set local lock_timeout='50ms'")
                        await connection.execute("update jarvis_task_context_manifests set status=status where id=$1", manifest)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update jarvis_task_context_manifests set status=status where id=$1", manifest)
