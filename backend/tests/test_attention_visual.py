"""Real visual producers, durable inbox, and phone writer with synthetic PCM."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from types import SimpleNamespace
from typing import get_args
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.attention.models import AttentionError, InboxMutation
from app.attention.notifications import selected_notification_repository, speech_attribution
from app.attention.repository import AttentionRepository
from app.memory.prospective.models import VisualFocusMode
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.contracts import FastInterruptCandidate, FastPerceptionOutput, FastProspectiveMatch
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.realtime.socket_sender import WebSocketEventSender
from app.reminders.repository import ReminderRepository
from app.settings import JarvisSettings
from app.attention.tts import OpenAISpeechProvider
from app.visual_memory.repository import VisualFrameRepository
from evals.second_brain.visual_observation_fixture import run_visual_fixture
from tests import test_attention_inbox as inbox_tests
from tests import test_attention_voice as voice_tests
from tests import test_second_brain_observation_publication as observations


class VisualAttentionUnitTests(unittest.TestCase):
    def test_selected_unavailable_is_not_legacy_fallback(self):
        self.assertIsNone(selected_notification_repository(SimpleNamespace()))
        with self.assertRaises(AttentionError):
            selected_notification_repository(SimpleNamespace(
                attention_notifications_enabled=True, attention_repository=None))

    def test_attribution_is_not_expert_or_physical_completion(self):
        for namespace in ("visual_alert", "prospective_event"):
            origin, author, label = speech_attribution(namespace)
            self.assertEqual((origin, author), ("notification_tts", "visual_analyzer"))
            self.assertIn("Observation-backed", label)
        with self.assertRaises(AttentionError):
            speech_attribution("untrusted_model_choice")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class VisualAttentionDatabaseTests(unittest.IsolatedAsyncioTestCase):
    observation = observations.ObservationPublicationDatabaseTests.observation
    insert = observations.ObservationPublicationDatabaseTests.insert
    source = observations.ObservationPublicationDatabaseTests.source
    complete = inbox_tests.AttentionDatabaseTests.complete

    async def frame_for(self, owner, session, *, sequence=0):
        self.when = datetime.now(UTC)
        return await observations.ObservationPublicationDatabaseTests.frame_for(
            self, owner, session, sequence=sequence)

    async def asyncSetUp(self):
        await observations.ObservationPublicationDatabaseTests.asyncSetUp(self)
        self.inbox = AttentionRepository(self.tasks)
        self.tts = OpenAISpeechProvider(SimpleNamespace(with_options=lambda **kwargs: SimpleNamespace()))
        self.prospective = ProspectiveRepository(self.f.pool)
        self.reminders = ReminderRepository(self.f.pool)
        self.manager = RealtimeSessionManager()
        self.manager.attention_notifications_enabled = True
        self.manager.attention_repository = self.inbox
        self.search = ProspectiveMemoryService(repository=self.prospective,
            realtime_session_manager=self.manager, settings=JarvisSettings(_env_file=None))

    async def focus(self, *, mode="find_once"):
        return await self.prospective.set_visual_focus(owner_id=self.f.owner,
            target="fixture prop", description="Synthetic harmless target", purpose="test", mode=mode,
            expires_at=datetime.now(UTC)+timedelta(minutes=10))

    async def visual(self, source, **changes):
        return await VisualFrameRepository(self.f.pool).create_visual_interrupt(**(dict(
            owner_id=self.f.owner, session_id=self.session, visual_frame_id=self.frame,
            message="Possible fixture match.", reason="Synthetic evidence", confidence=0.99,
            dedupe_bucket="fixture", source_payload_id=source.payload_id,
            attention=self.inbox, max_age_seconds=300) | changes))

    async def tracked(self, source, *, subject=None, kind="visual_focus", max_age=300):
        subject = subject or await self.focus()
        token = await self.prospective.try_claim_subject(owner_id=self.f.owner, subject_type=kind,
            subject_id=subject.id, claim_timeout_seconds=60)
        self.assertIsNotNone(token)
        event = await self.prospective.reserve_delivery_event(owner_id=self.f.owner, session_id=self.session,
            subject_type=kind, subject_id=subject.id, claim_token=token, occurrence_key=uuid4().hex,
            message="Possible fixture match.", evidence={"observed_condition":"Synthetic match",
                "supporting_frame_ids":[str(self.frame)], "source_payload_id":str(source.payload_id)},
            dedupe_window_seconds=0, source_payload_id=source.payload_id,
            attention=self.inbox, max_age_seconds=max_age)
        self.assertIsNotNone(event)
        return subject, event

    async def intent(self, event):
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select * from jarvis_attention_intents where owner_id=$1 and source_id=$2",
                                  self.f.owner, event.id)
        self.assertIsNotNone(row)
        return row

    async def claim(self):
        result = await self.inbox.claim_next(self.f.owner, self.session, "fixture-visual")
        self.assertIsNotNone(result)
        return result

    async def generated(self, claim):
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        async with self.inbox.write_guard(claim, 0, start_sample=0, samples=480) as mark_sent:
            mark_sent()
        await self.inbox.chunk_generated(claim, 0, samples=480, first_bytes_ms=1,
                                         provider_request_id="synthetic")
        await self.inbox.generation_finished(claim)

    async def receipt(self, claim):
        return await self.inbox.playback_receipt(claim, item_id=f"tts_{claim.attempt_id.hex}_0",
            response_id=f"tts_{claim.attempt_id.hex}", samples=480, completed=True)

    async def remove(self, source):
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)

    async def recover(self):
        async with self.tasks.transaction(self.f.owner) as c:
            await self.inbox._recover(c, self.f.owner)
        await self.inbox.reconcile_notifications(self.f.owner)

    async def make_voice(self):
        self.events, self.done = [], asyncio.Event()
        async def send(text):
            value = json.loads(text)
            self.events.append(value)
            if value["type"] == "audio.output.done":
                self.done.set()
        self.voice = RealtimeAudioSession(
            websocket=SimpleNamespace(send_text=send, close=AsyncMock()),
            context=SimpleNamespace(audio_session_id="fixture", owner_id=self.f.owner,
                jarvis_session_id=self.session, client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=())
        self.voice.openai = SimpleNamespace(send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.voice._realtime_context_ready.set()
        self.addAsyncCleanup(self.voice.close, "fixture_cleanup")
        return self.voice

    async def test_atomic_generic_dedupe_restart_and_owner_scope(self):
        source = await self.source(await self.insert())
        first = await self.visual(source)
        second = await self.visual(source)
        self.assertEqual(first.id, second.id)
        row = await self.intent(first)
        restarted = AttentionRepository(self.tasks)
        detail = await restarted.detail(self.f.owner, row["id"])
        self.assertEqual(detail["state"], "pending")
        self.assertIsNone(detail["expert_result"])
        self.assertEqual(detail["notification"]["message"], "Possible fixture match.")
        self.assertEqual(detail["notification"]["authored_by"], "visual_analyzer")
        self.assertEqual(detail["notification"]["observation_payload_id"], str(source.payload_id))
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from visual_interrupts"), 1)
            self.assertEqual(await c.fetchval("select count(*) from jarvis_attention_intents"), 1)
            self.assertTrue(await c.fetchval("""select exists(select 1 from jarvis_context_payload_dependencies
                where source_payload_id=$1 and dependent_payload_id=$2)""", source.payload_id, row["source_payload_id"]))
        with self.assertRaises(AttentionError):
            await restarted.detail(self.f.other, row["id"])

    async def test_failed_enqueue_rolls_back_raw_record_binding_and_intent(self):
        source = await self.source(await self.insert())
        bad = SimpleNamespace(enqueue_visual=AsyncMock(side_effect=RuntimeError("fixture")))
        with self.assertRaises(RuntimeError):
            await self.visual(source, attention=bad)
        with self.assertRaises(AttentionError):
            await self.visual(source, source_payload_id=None)
        async with self.f.pool.acquire() as c:
            for table in ("visual_interrupts", "jarvis_visual_delivery_sources", "jarvis_attention_intents"):
                self.assertEqual(await c.fetchval("select count(*) from "+table), 0)

    async def test_pipeline_general_routes_to_inbox_without_native_speech(self):
        output = FastPerceptionOutput(summary="Synthetic possible match.", confidence=.9, starts_new_event=False,
            interrupt_candidate=FastInterruptCandidate(message="Possible fixture match.",
                reason="Synthetic evidence", confidence=.99, scope="general"))
        result = await run_visual_fixture(self.f, self.publisher, self.frame, output.summary,
                                         analyzer_output=output, attention_repository=self.inbox)
        self.assertEqual(result["notifications"], [])
        entries = await self.inbox.list(self.f.owner)
        self.assertEqual([entry["source_namespace"] for entry in entries], ["visual_alert"])

    async def test_pipeline_tracked_candidate_owns_notification_without_generic_competition(self):
        focus = await self.focus()
        output = FastPerceptionOutput(summary="Synthetic possible match.", confidence=.9, starts_new_event=False,
            interrupt_candidate=FastInterruptCandidate(message="Do not send competing generic alert.",
                reason="Synthetic evidence", confidence=.99, scope="general"),
            prospective_matches=[FastProspectiveMatch(subject_type="visual_focus", subject_id=focus.id,
                confidence=.99, observed_condition="Synthetic match", supporting_frame_ids=[self.frame],
                suggested_message="Possible fixture match.")])
        result = await run_visual_fixture(self.f, self.publisher, self.frame, output.summary,
                                         analyzer_output=output, attention_repository=self.inbox)
        self.assertEqual(result["notifications"], [])
        self.assertEqual([entry["source_namespace"] for entry in await self.inbox.list(self.f.owner)],
                         ["prospective_event"])
        self.assertIsNone((await self.prospective.get_active_visual_focus(owner_id=self.f.owner)).completed_at)

    async def test_find_once_completes_only_after_exact_client_receipt(self):
        source = await self.source(await self.insert())
        focus, event = await self.tracked(source)
        claim = await self.claim()
        await self.generated(claim)
        self.assertIsNotNone(await self.prospective.get_active_visual_focus(owner_id=self.f.owner))
        self.assertTrue(await self.receipt(claim))
        self.assertIsNone(await self.prospective.get_active_visual_focus(owner_id=self.f.owner))
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select * from jarvis_visual_focuses where id=$1", focus.id)
            self.assertIsNotNone(row["completed_at"])
            evidence = json.loads(row["completion_evidence"])
            self.assertEqual(evidence["physical_action_status"], "not_established")
            self.assertEqual(evidence["playback_status"], "client_confirmed")
            self.assertEqual(await c.fetchval("select event_type from jarvis_prospective_delivery_events where id=$1",
                                             event.id), "delivered_to_session")

    async def test_monitoring_stays_active_after_receipt(self):
        source = await self.source(await self.insert())
        focus, _ = await self.tracked(source, subject=await self.focus(mode=next(mode for mode in get_args(VisualFocusMode) if mode != "find_once")))
        claim = await self.claim()
        await self.generated(claim)
        self.assertTrue(await self.receipt(claim))
        current = await self.prospective.get_active_visual_focus(owner_id=self.f.owner)
        self.assertEqual(current.id, focus.id)
        self.assertIsNone(current.completed_at)
        self.assertIsNone(current.trigger_claim_token)

    async def test_conditional_reminder_excludes_timed_claim_and_does_not_assert_action(self):
        reminder = await self.reminders.create_reminder(owner_id=self.f.owner, session_id=self.session,
            title="Check the fixture", details="No physical action is assumed.", condition="When fixture visible",
            remind_at=datetime.now(UTC)-timedelta(seconds=1))
        source = await self.source(await self.insert())
        _, event = await self.tracked(source, subject=reminder, kind="conditional_reminder")
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_reminders set trigger_claimed_at=clock_timestamp()-interval '1 hour' where id=$1", reminder.id)
        self.assertEqual(await self.reminders.claim_due_reminders(owner_id=self.f.owner,
            claim_timeout_seconds=60, max_attempts=3), [])
        claim = await self.claim()
        await self.generated(claim)
        self.assertTrue(await self.receipt(claim))
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select * from jarvis_reminders where id=$1", reminder.id)
            self.assertEqual(row["status"], "completed")
            self.assertEqual(json.loads(row["metadata"])["physical_action_status"], "not_established")
            self.assertEqual(json.loads(row["metadata"])["attention_completed_event_id"], str(event.id))

    async def test_attention_ownership_survives_lease_and_rejects_legacy_callbacks(self):
        source = await self.source(await self.insert())
        focus, event = await self.tracked(source)
        async with self.f.pool.acquire() as c:
            await c.execute("update jarvis_visual_focuses set trigger_claimed_at=clock_timestamp()-interval '1 hour' where id=$1", focus.id)
        restarted = ProspectiveRepository(self.f.pool)
        self.assertIsNone(await restarted.try_claim_subject(owner_id=self.f.owner, subject_type="visual_focus",
            subject_id=focus.id, claim_timeout_seconds=60))
        self.assertFalse(await restarted.begin_delivery(owner_id=self.f.owner, event_id=event.id,
            session_id=self.session, source_payload_id=source.payload_id))
        await restarted.resolve_delivery_event(owner_id=self.f.owner, event_id=event.id,
            outcome="delivered", provider_response_id="stale-native", source_payload_id=source.payload_id)
        self.assertIsNotNone(await restarted.get_active_visual_focus(owner_id=self.f.owner))
        self.assertEqual((await self.inbox.detail(self.f.owner, (await self.intent(event))["id"]))["state"], "pending")

    async def test_unknown_requires_explicit_replay_same_occurrence_then_dismiss_releases(self):
        source = await self.source(await self.insert())
        focus, event = await self.tracked(source)
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        await self.inbox.settle(claim, reason="fixture_disconnect")
        self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "restart"))
        detail = await self.inbox.detail(self.f.owner, claim.intent_id)
        self.assertEqual(detail["state"], "unknown")
        await self.inbox.mutate(self.f.owner, claim.intent_id, "replay",
            InboxMutation(request_id=uuid4(), expected_generation=detail["generation"]))
        replay = await self.claim()
        self.assertEqual(replay.intent_id, claim.intent_id)
        self.assertEqual((await self.inbox.prepare(replay, self.tts))[0].text, request.text)
        detail = await self.inbox.detail(self.f.owner, claim.intent_id)
        await self.inbox.mutate(self.f.owner, claim.intent_id, "dismiss",
            InboxMutation(request_id=uuid4(), expected_generation=detail["generation"]))
        current = await self.prospective.get_active_visual_focus(owner_id=self.f.owner)
        self.assertEqual(current.id, focus.id)
        self.assertIsNone(current.trigger_claim_token)
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select count(*) from jarvis_prospective_delivery_events"), 1)

    async def test_expired_unsent_releases_but_unknown_does_not_reissue(self):
        source = await self.source(await self.insert())
        focus, event = await self.tracked(source, max_age=1)
        await asyncio.sleep(1.1)
        await self.recover()
        self.assertEqual((await self.inbox.detail(self.f.owner, (await self.intent(event))["id"]))["state"], "expired")
        self.assertIsNone((await self.prospective.get_active_visual_focus(owner_id=self.f.owner)).trigger_claim_token)
        self.frame = await self.frame_for(self.f.owner, self.session, sequence=1)
        source = await self.source(await self.insert())
        _, event = await self.tracked(source, subject=focus, max_age=15)
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        await self.inbox.settle(claim, reason="fixture_disconnect")
        await asyncio.sleep(15.1)
        await self.recover()
        self.assertEqual((await self.inbox.detail(self.f.owner, claim.intent_id))["state"], "unknown")
        self.assertIsNotNone((await self.prospective.get_active_visual_focus(owner_id=self.f.owner)).trigger_claim_token)

    async def test_clear_after_generation_blocks_late_completion(self):
        source = await self.source(await self.insert())
        focus, _ = await self.tracked(source)
        claim = await self.claim()
        await self.generated(claim)
        await self.search.clear_visual_focus(owner_id=self.f.owner, focus_id=focus.id)
        self.assertFalse(await self.receipt(claim))
        async with self.f.pool.acquire() as c:
            row = await c.fetchrow("select cleared_at,completed_at from jarvis_visual_focuses where id=$1", focus.id)
            self.assertIsNotNone(row["cleared_at"])
            self.assertIsNone(row["completed_at"])

    async def test_queued_phone_write_is_fenced_by_subject_replacement(self):
        source = await self.source(await self.insert())
        _, _ = await self.tracked(source)
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        socket = SimpleNamespace(send_text=AsyncMock())
        writer = WebSocketEventSender(socket)
        await writer._lock.acquire()
        pending = asyncio.create_task(writer.send_text("{}", write_guard=lambda:
            self.inbox.write_guard(claim, 0, start_sample=0, samples=480)))
        try:
            await asyncio.sleep(0)
            await self.focus()
        finally:
            writer._lock.release()
        with self.assertRaises(AttentionError):
            await asyncio.wait_for(pending, 5)
        socket.send_text.assert_not_awaited()

    async def test_source_deletion_waits_for_inflight_write_then_scrubs_copies(self):
        source = await self.source(await self.insert())
        _, event = await self.tracked(source)
        claim = await self.claim()
        request = (await self.inbox.prepare(claim, self.tts))[0]
        await self.inbox.submit_chunk(claim, 0, request)
        started, release = asyncio.Event(), asyncio.Event()
        async def send(_):
            started.set()
            await release.wait()
        writer = WebSocketEventSender(SimpleNamespace(send_text=send))
        pending = asyncio.create_task(writer.send_text("{}", write_guard=lambda:
            self.inbox.write_guard(claim, 0, start_sample=0, samples=480)))
        await asyncio.wait_for(started.wait(), 5)
        deletion = asyncio.create_task(self.remove(source))
        try:
            await asyncio.sleep(.05)
            self.assertFalse(deletion.done())
        finally:
            release.set()
        self.assertTrue(await asyncio.wait_for(pending, 10))
        await asyncio.wait_for(deletion, 10)
        with self.assertRaises(AttentionError):
            await self.inbox.prepare(claim, self.tts)
        self.assertIsNone((await self.inbox.detail(self.f.owner, claim.intent_id))["notification"])
        async with self.f.pool.acquire() as c:
            self.assertEqual(await c.fetchval("select message from jarvis_prospective_delivery_events where id=$1",
                                             event.id), "[source unavailable]")

    async def test_real_writer_uses_visual_attribution_and_receipts_not_native_generation(self):
        source = await self.source(await self.insert())
        focus, _ = await self.tracked(source)
        voice = await self.make_voice()
        speech = voice_tests.SyntheticSpeech(self.tts)
        delivery = await voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=10)
        self.assertIsNotNone(delivery)
        await asyncio.wait_for(self.done.wait(), 10)
        await voice._handle_client_event({"type":"playback.completed", "response_id":delivery.response_id,
            "item_id":delivery.item_id(0), "played_samples":480})
        await asyncio.wait_for(delivery.task, 10)
        self.assertIsNone(await self.prospective.get_active_visual_focus(owner_id=self.f.owner))
        metadata = voice.repository.create_message.await_args.kwargs["metadata"]
        self.assertEqual(metadata["response_kind"], "notification_tts")
        self.assertEqual(metadata["authored_by"], "visual_analyzer")
        self.assertFalse(any(call.args[0]["type"] == "response.create"
                             for call in voice.openai.send_event.await_args_list))
        self.assertEqual(speech.requests[0].text, "Possible fixture match.")

    async def test_generic_quiet_interval_and_proactive_cap_do_not_consume_attempt(self):
        source = await self.source(await self.insert())
        event = await self.visual(source)
        voice = await self.make_voice()
        voice._last_user_turn_settled_at = datetime.now(UTC)
        voice._last_user_turn_settled_monotonic = voice._monotonic()
        speech = voice_tests.SyntheticSpeech(self.tts)
        self.assertIsNone(await voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=5))
        row = await self.intent(event)
        self.assertEqual(row["attempt_count"], 0)
        voice._last_user_turn_settled_monotonic -= 10
        self.assertIsNone(await voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=5))
        self.assertEqual((await self.intent(event))["state"], "expired")
        _, tracked = await self.tracked(source)
        while voice._proactive_interrupt_gate.try_begin():
            pass
        self.assertIsNone(await voice.try_attention(self.inbox, speech, worker_id="fixture", receipt_seconds=5))
        self.assertEqual((await self.intent(tracked))["attempt_count"], 0)

    async def test_mixed_expert_and_visual_share_one_voice_claim(self):
        _, expert = await self.complete()
        source = await self.source(await self.insert())
        event = await self.visual(source)
        visual = (await self.intent(event))["id"]
        for expected in (expert, visual):
            claim = await self.claim()
            self.assertEqual(claim.intent_id, expected)
            self.assertIsNone(await self.inbox.claim_next(self.f.owner, self.session, "competitor"))
            await self.inbox.settle(claim, reason="fixture_no_submission")
