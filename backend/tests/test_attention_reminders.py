"""Scheduled reminders on the real inbox/writer; synthetic PCM, no real notifications."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import json
import os

import asyncpg
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.attention.models import AttentionError, InboxMutation
from app.attention.notifications import reminder_text, speech_chunks
from app.attention.repository import AttentionRepository
from app.realtime.session_manager import RealtimeSessionManager
from app.reminders.delivery import ReminderDeliveryService
from app.reminders.repository import ReminderRepository
from tests import test_attention_inbox as inbox_tests
from tests import test_attention_voice as voice_tests


class NotificationTextTests(unittest.IsolatedAsyncioTestCase):
    def test_literal_instruction_does_not_assert_completion(self):
        text = reminder_text({"title":"Check the stove", "details":"I do not know whether it is off."})
        self.assertEqual(text, "Reminder: Check the stove. I do not know whether it is off.")

    def test_utf8_splitting_is_lossless_and_bounded(self):
        for text in ("exact words " * 1000, "\u6e2c\u8a66\U0001f642 " * 1300, "x" * 1601):
            with self.subTest(length=len(text)):
                chunks = speech_chunks(text)
                self.assertEqual("".join(chunks), text)
                self.assertTrue(all(0 < len(chunk.encode("utf-8")) <= 1600 for chunk in chunks))
                self.assertLessEqual(len(chunks), 64)
        self.assertEqual(speech_chunks("x" * 102401), ())
        self.assertEqual(speech_chunks(" "), ())
        self.assertEqual(speech_chunks("bad\x00input"), ())

    async def test_selected_route_never_calls_generator_or_native_speech(self):
        inbox = object()
        manager = SimpleNamespace(attention_notifications_enabled=True, attention_repository=inbox,
                                  speak_reminder=AsyncMock())
        repository = SimpleNamespace(finalize_trigger=AsyncMock(return_value=SimpleNamespace(id=uuid4())))
        generator = SimpleNamespace(generate=AsyncMock())
        service = ReminderDeliveryService(owner_id="owner", repository=repository, session_repository=None,
                                         realtime_session_manager=manager, generator=generator)
        reminder = SimpleNamespace(id=uuid4(), session_id=None, trigger_claim_token=uuid4(), trigger_attempt_count=1)
        await service.deliver(reminder)
        args = repository.finalize_trigger.await_args.kwargs
        self.assertIs(args["attention"], inbox)
        self.assertEqual(args["metadata"]["physical_action_status"], "not_established")
        generator.generate.assert_not_awaited()
        manager.speak_reminder.assert_not_awaited()

    async def test_unavailable_selected_inbox_does_not_fall_back(self):
        manager = SimpleNamespace(attention_notifications_enabled=True, attention_repository=None,
                                  speak_reminder=AsyncMock())
        generator = SimpleNamespace(generate=AsyncMock())
        repository = SimpleNamespace(finalize_trigger=AsyncMock())
        service = ReminderDeliveryService(owner_id="owner", repository=repository, session_repository=None,
                                         realtime_session_manager=manager, generator=generator)
        with self.assertRaises(AttentionError):
            await service.deliver(SimpleNamespace(id=uuid4(), trigger_attempt_count=1))
        generator.generate.assert_not_awaited()
        repository.finalize_trigger.assert_not_awaited()
        manager.speak_reminder.assert_not_awaited()

    async def test_disabled_route_preserves_legacy_generation_and_dispatch(self):
        manager = SimpleNamespace(attention_notifications_enabled=False, speak_reminder=AsyncMock(return_value=uuid4()))
        event = SimpleNamespace(id=uuid4())
        repository = SimpleNamespace(finalize_trigger=AsyncMock(return_value=event))
        generator = SimpleNamespace(generate=AsyncMock(return_value="Legacy generated text."))
        service = ReminderDeliveryService(owner_id="owner", repository=repository, session_repository=None,
                                         realtime_session_manager=manager, generator=generator)
        reminder = SimpleNamespace(id=uuid4(), session_id=None, trigger_claim_token=uuid4(), trigger_attempt_count=1)
        await service.deliver(reminder)
        generator.generate.assert_awaited_once()
        manager.speak_reminder.assert_awaited_once()
        self.assertNotIn("attention", repository.finalize_trigger.await_args.kwargs)

    async def test_notification_capture_is_not_expert_or_a_heard_transcript(self):
        from app.tasks.context_capture import SessionContextCapture
        repository = SimpleNamespace(append=AsyncMock())
        capture = SessionContextCapture(repository, owner_id="fixture", session_id=uuid4(),
                                        provider="openai", audio_session_id="fixture")
        self.addAsyncCleanup(capture.close)
        capture.tts_speech(response_id="tts", item_id="item", text="Reminder: check directly.",
            samples=480, complete=True, interrupted=False, provenance={}, authored_by="stored_reminder")
        self.assertTrue(await capture.flush())
        payload = repository.append.await_args.args[1].payload
        self.assertEqual(payload["authored_by"], "stored_reminder")
        self.assertEqual(payload["text_represents"], "notification_script_not_a_verified_heard_transcript")
        self.assertIsNone(payload["heard_transcript"])


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class ReminderAttentionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    complete = inbox_tests.AttentionDatabaseTests.complete

    async def asyncSetUp(self):
        await inbox_tests.AttentionDatabaseTests.asyncSetUp(self)
        self.reminders = ReminderRepository(self.f.pool)
        self.manager = RealtimeSessionManager()
        self.manager.attention_notifications_enabled = True
        self.manager.attention_repository = self.inbox
        self.generator = SimpleNamespace(generate=AsyncMock(side_effect=AssertionError("Unexpected model generation")))
        self.delivery = ReminderDeliveryService(owner_id=self.f.owner, repository=self.reminders,
            session_repository=None, realtime_session_manager=self.manager, generator=self.generator)

    async def due(self, *, title="Check the stove", details="Do not assume it is off."):
        result = await self.reminders.create_reminder(owner_id=self.f.owner, session_id=self.session,
            title=title, details=details, remind_at=datetime.now(UTC)-timedelta(seconds=10), condition=None)
        claimed = await self.reminders.claim_due_reminders(owner_id=self.f.owner, claim_timeout_seconds=120,
                                                         max_attempts=3)
        return next(item for item in claimed if item.id == result.id)

    async def enqueue(self, **kwargs):
        reminder = await self.due(**kwargs)
        await self.delivery.deliver(reminder)
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow(
                """select i.*,b.reminder_id from jarvis_attention_intents i
                   join jarvis_reminder_attention_sources b on b.owner_id=i.owner_id and b.event_id=i.source_id
                   where i.owner_id=$1 and b.reminder_id=$2""", self.f.owner, reminder.id)
        return reminder, row

    async def claim(self):
        claim = await self.inbox.claim_next(self.f.owner, self.session, "fixture-reminder")
        self.assertIsNotNone(claim)
        return claim

    async def test_atomic_finalize_and_restarted_inbox_not_expert_result(self):
        reminder, row = await self.enqueue()
        restarted = AttentionRepository(self.tasks)
        detail = await restarted.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "pending")
        self.assertEqual(detail["privacy"], "sensitive")
        self.assertIsNone(detail["expert_result"])
        self.assertFalse(detail["expert_speech_prepared"])
        self.assertTrue(detail["speech_prepared"])
        self.assertEqual(detail["notification"]["authored_by"], "stored_reminder")
        self.assertEqual(detail["notification"]["message"], "Reminder: Check the stove. Do not assume it is off.")
        self.assertEqual(detail["attempts"], [])
        self.generator.generate.assert_not_awaited()
        with self.assertRaises(RuntimeError):
            await self.delivery.deliver(reminder)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_attention_intents"), 1)
            self.assertEqual(await c.fetchval("select count(*) from jarvis_reminder_events"), 1)
        with self.assertRaises(AttentionError):
            await restarted.detail(self.f.other, row["id"])

    async def test_enqueue_failure_rolls_back_reminder_and_event(self):
        reminder = await self.due()
        bad = SimpleNamespace(enqueue_reminder=AsyncMock(side_effect=RuntimeError("fixture enqueue failure")))
        self.manager.attention_repository = bad
        with self.assertRaises(RuntimeError):
            await self.delivery.deliver(reminder)
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select status,trigger_claim_token,triggered_at from jarvis_reminders where id=$1", reminder.id)
            self.assertEqual(row["status"], "active")
            self.assertEqual(row["trigger_claim_token"], reminder.trigger_claim_token)
            self.assertIsNone(row["triggered_at"])
            self.assertEqual(await c.fetchval("select count(*) from jarvis_reminder_events"), 0)
            self.assertEqual(await c.fetchval("select count(*) from jarvis_attention_intents"), 0)

    async def test_deletion_revokes_prepared_speech_and_derived_copies(self):
        reminder, row = await self.enqueue()
        claim = await self.claim()
        await self.inbox.prepare(claim, self.tts)
        async with self.tasks.transaction(self.f.owner) as c:
            copy = await self.tasks.payload(c, self.f.owner, {"text":"Reminder copied into context"})
            await c.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                self.f.owner, row["source_payload_id"], copy)
        await self.reminders.delete_reminder(owner_id=self.f.owner, reminder_id=reminder.id)
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(claim, self.tts)
        detail = await self.inbox.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "revoked")
        self.assertIsNone(detail["notification"])
        async with self.f.pool.acquire() as c:
            self.assertIsNone(await c.fetchval("select content from jarvis_context_payloads where id=$1", copy))
            self.assertEqual(await c.fetchval("select message from jarvis_reminder_events where id=$1", row["source_id"]),
                             "[source unavailable]")

    async def test_current_edit_before_finalize_uses_locked_instruction_then_later_edit_revokes(self):
        reminder = await self.due()
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_reminders set details='Check it directly first.' where id=$1", reminder.id)
        await self.delivery.deliver(reminder)
        row = (await self.inbox.list(self.f.owner))[0]
        detail = await self.inbox.detail(self.f.owner, row["id"])
        self.assertEqual(detail["notification"]["message"], "Reminder: Check the stove. Check it directly first.")
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_reminders set title='Check the lights' where id=$1", reminder.id)
        self.assertIsNone((await self.inbox.detail(self.f.owner, row["id"]))["notification"])

    async def test_explicit_completion_cancels_already_generated_notification(self):
        reminder, row = await self.enqueue()
        claim = await self.claim()
        await self.inbox.prepare(claim, self.tts)
        await self.reminders.complete_reminder(owner_id=self.f.owner, reminder_id=reminder.id)
        detail = await self.inbox.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "revoked")
        self.assertIsNone(detail["notification"])
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(claim, self.tts)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_reminder_events"), 1)

    async def test_wrong_owner_binding_is_rejected(self):
        reminder, row = await self.enqueue()
        async with self.f.pool.acquire() as c, c.transaction():
            other_payload = await self.tasks.payload(c, self.f.other, {"format":"jarvis.attention.notification.v1"})
            with self.assertRaises(asyncpg.CheckViolationError):
                async with c.transaction():
                    await c.execute("""insert into jarvis_reminder_attention_sources
                        (owner_id,reminder_id,event_id,payload_id) values($1,$2,$3,$4)""",
                        self.f.other, reminder.id, row["source_id"], other_payload)

    async def test_unknown_delivery_replay_uses_same_occurrence_without_new_generation(self):
        reminder, row = await self.enqueue()
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        await self.inbox.settle(claim, reason="fixture_disconnect")
        detail = await self.inbox.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "unknown")
        await self.inbox.mutate(self.f.owner, row["id"], "replay",
            InboxMutation(request_id=uuid4(), expected_generation=detail["generation"]))
        replay = await self.claim()
        self.assertNotEqual(replay.attempt_id, claim.attempt_id)
        self.assertEqual(replay.intent_id, claim.intent_id)
        self.assertEqual((await self.inbox.prepare(replay, self.tts))[0].text, request.text)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_reminder_events"), 1)
            self.assertEqual(await c.fetchval("select trigger_attempt_count from jarvis_reminders where id=$1", reminder.id), 1)
        self.generator.generate.assert_not_awaited()

    async def test_mixed_experts_and_reminder_use_one_owner_voice_claim(self):
        first, first_intent = await self.complete()
        reminder, row = await self.enqueue()
        second, second_intent = await self.complete()
        expected = [(first_intent,"task_result"), (row["id"],"reminder_event"), (second_intent,"task_result")]
        for identity, namespace in expected:
            claim = await self.claim()
            self.assertEqual((claim.intent_id, claim.source_namespace), (identity, namespace))
            self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "competitor"))
            await self.inbox.prepare(claim, self.tts)
            await self.inbox.settle(claim, reason="fixture_pre_submission_failure")
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "finished"))

    async def test_reminder_speech_uses_real_writer_and_receipts_without_expert_attribution(self):
        await voice_tests.AttentionVoiceDatabaseTests.asyncSetUp(self)
        # The voice setup creates a fresh manifested owner/pool. Bind the producer to it.
        self.reminders = ReminderRepository(self.f.pool)
        self.manager.attention_repository = self.inbox
        self.delivery._repository = self.reminders
        self.delivery._owner_id = self.f.owner
        self.generated = asyncio.Event()
        async def send(text):
            event = json.loads(text)
            self.events.append(event)
            if event["type"] == "audio.output.done":
                self.generated.set()
        # Install the same production writer over an explicit synthetic phone.
        from app.realtime.socket_sender import WebSocketEventSender
        self.voice._app_sender = WebSocketEventSender(SimpleNamespace(send_text=send))
        reminder, row = await self.enqueue()
        self.voice._user_audio_pending = True
        speech = voice_tests.SyntheticSpeech(self.tts)
        self.assertIsNone(await self.voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=10))
        self.voice._user_audio_pending = False
        delivery = await self.voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=10)
        self.assertIsNotNone(delivery)
        await asyncio.wait_for(self.generated.wait(), 15)
        detail = await self.inbox.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "awaiting_receipt")
        self.assertEqual(speech.requests[0].text, detail["notification"]["message"])
        self.assertTrue(all(e["audio_origin"] == "notification_tts" for e in self.events))
        self.assertEqual(self.voice.repository.create_message.await_args.kwargs["metadata"]["authored_by"], "stored_reminder")
        self.assertFalse(any(c.args[0]["type"] == "response.create" for c in self.voice.openai.send_event.await_args_list))
        await self.voice._handle_client_event({"type":"playback.completed", "response_id":delivery.response_id,
            "item_id":delivery.item_id(0), "played_samples":480, "content_index":0})
        await asyncio.wait_for(delivery.task, 10)
        self.assertEqual((await self.inbox.detail(self.f.owner, row["id"]))["state"], "played")
        async with self.f.pool.acquire() as c:
            event = await c.fetchrow("select event_type,metadata from jarvis_reminder_events where id=$1", row["source_id"])
            self.assertEqual(event["event_type"], "generated")
            metadata = json.loads(event["metadata"]) if isinstance(event["metadata"], str) else event["metadata"]
            self.assertEqual(metadata["physical_action_status"], "not_established")

    async def test_source_deletion_under_socket_lock_sends_no_stale_audio(self):
        reminder, row = await self.enqueue()
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        from app.realtime.socket_sender import WebSocketEventSender
        socket = SimpleNamespace(send_text=AsyncMock())
        sender = WebSocketEventSender(socket)
        await sender._lock.acquire()
        sending = asyncio.create_task(sender.send_text("synthetic PCM", write_guard=lambda:
            self.inbox.write_guard(claim, 0, start_sample=0, samples=240)))
        try:
            await asyncio.sleep(0)
            await self.reminders.delete_reminder(owner_id=self.f.owner, reminder_id=reminder.id)
            sender._lock.release()
            with self.assertRaises(AttentionError):
                await asyncio.wait_for(sending, 10)
            socket.send_text.assert_not_awaited()
        finally:
            if sender._lock.locked():
                sender._lock.release()
            if not sending.done():
                sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
