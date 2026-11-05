"""Real notification storage and phone-writer fences; no external speech."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg

from app.knowledge.models import SourceError
from app.attention.models import AttentionError
from app.knowledge.visual_delivery_sources import (
    bind_visual_delivery_source, lock_visual_delivery_source, observation_delivery_source,
)
from app.memory.prospective.repository import ProspectiveRepository
from app.memory.prospective.service import ProspectiveMemoryService
from app.perception.contracts import (
    FastInterruptCandidate, FastPerceptionFrame, FastPerceptionOutput, FastPerceptionRequest, FastProspectiveMatch,
)
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.settings import JarvisSettings
from app.visual_memory.interrupts import VisualInterruptService
from app.visual_memory.repository import VisualFrameRepository
from tests import test_second_brain_observation_publication as observations
from evals.second_brain.visual_observation_fixture import run_visual_fixture
from test_attention_quiet import LedgerFixture


def voice(owner, session_id, pool=None):
    events = []
    async def send_text(text):
        events.append(json.loads(text))
    instance = RealtimeAudioSession(
        websocket=SimpleNamespace(send_text=send_text, close=AsyncMock()),
        context=SimpleNamespace(owner_id=owner, jarvis_session_id=session_id, audio_session_id="fixture-audio"),
        settings=JarvisSettings(_env_file=None), repository=SimpleNamespace(_pool=pool),
        tool_modules=(), context_providers=())
    instance._realtime_context_ready.set()
    return instance, events


async def close_voice(instance):
    instance._context_reset_ack.set()
    await instance.close("fixture_complete")


class VisualDeliverySourceUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_unbound_legacy_helpers_do_not_access_source_storage(self):
        connection = SimpleNamespace(fetchval=AsyncMock(), execute=AsyncMock())
        await lock_visual_delivery_source(connection, owner_id="fixture", session_id=uuid4(),
            payload_id=None, frame_ids=())
        await bind_visual_delivery_source(connection, owner_id="fixture", session_id=uuid4(),
            kind="visual", delivery_id=uuid4(), payload_id=None)
        connection.fetchval.assert_not_awaited()
        connection.execute.assert_not_awaited()

    async def test_visual_producer_passes_same_parent_to_storage_and_speech(self):
        parent = uuid4()
        record = SimpleNamespace(id=uuid4())
        repository = SimpleNamespace(recent_visual_interrupt_exists=AsyncMock(return_value=False),
            create_visual_interrupt=AsyncMock(return_value=record))
        manager = SimpleNamespace(speak_visual_interrupt=AsyncMock(return_value=True))
        service = VisualInterruptService(repository=repository, realtime_session_manager=manager, dedupe_seconds=10)
        await service.maybe_interrupt(owner_id="fixture", session_id=uuid4(), audio_session_id="a",
            visual_frame_id=uuid4(), source_received_at=datetime.now(UTC),
            message="Possible fixture match.", reason="Synthetic observation", confidence=0.9,
            source_payload_id=parent)
        self.assertEqual(repository.create_visual_interrupt.await_args.kwargs["source_payload_id"], parent)
        self.assertEqual(manager.speak_visual_interrupt.await_args.kwargs["source_payload_id"], parent)

    async def test_generic_native_admission_observes_the_parent(self):
        quiet_pool = LedgerFixture()  # Explicit canonical inactive owner policy.
        instance, _ = voice("fixture", uuid4(), quiet_pool)
        self.addAsyncCleanup(close_voice, instance)
        instance._send_reserved_response = AsyncMock(return_value=True)
        instance._send_speech_notification = AsyncMock()
        parent = uuid4()
        self.assertTrue(await instance.speak_visual_interrupt(owner_id="fixture",
            jarvis_session_id=instance.context.jarvis_session_id, source_received_at=datetime.now(UTC),
            message="Synthetic possible match.", interrupt_id=uuid4(), source_payload_id=parent))
        self.assertIn(parent, instance._source_context._sources)
        instance._send_reserved_response.assert_awaited_once()
        self.assertTrue(any("attention_quiet:current" in query for query, _ in quiet_pool.calls))

    async def test_generic_native_admission_without_quiet_authority_fails_closed(self):
        instance, _ = voice("fixture", uuid4())
        self.addAsyncCleanup(close_voice, instance)
        instance._send_reserved_response = AsyncMock(return_value=True)
        with self.assertRaises(AttentionError) as error:
            await instance.speak_visual_interrupt(owner_id="fixture",
                jarvis_session_id=instance.context.jarvis_session_id, source_received_at=datetime.now(UTC),
                message="Synthetic possible match.", interrupt_id=uuid4(), source_payload_id=uuid4())
        self.assertEqual(error.exception.code, "owner_quiet_unavailable")
        instance._send_reserved_response.assert_not_awaited()

    async def test_tracked_native_admission_observes_parent_without_new_speech_loop(self):
        instance, _ = voice("fixture", uuid4())
        self.addAsyncCleanup(close_voice, instance)
        instance._send_reserved_response = AsyncMock(return_value=True)
        instance._send_speech_notification = AsyncMock()
        parent = uuid4()
        self.assertTrue(await instance.speak_prospective_interrupt(message="Synthetic possible match.",
            event_id=uuid4(), subject_type="visual_focus", subject_id=uuid4(),
            on_outcome=AsyncMock(), source_payload_id=parent))
        self.assertIn(parent, instance._source_context._sources)

    async def test_unbound_native_media_does_not_require_a_database(self):
        instance, events = voice("fixture", uuid4())
        self.addAsyncCleanup(close_voice, instance)
        self.assertTrue(await instance._send_app_event("audio.output.delta",
            response_id="response", item_id="item", audio_base64="AAAA"))
        self.assertEqual(events[0]["type"], "audio.output.delta")

    async def test_manager_preserves_parent_on_both_owned_routes(self):
        instance = SimpleNamespace(speak_visual_interrupt=AsyncMock(return_value=True),
            speak_prospective_interrupt=AsyncMock(return_value=True))
        manager = RealtimeSessionManager()
        session_id, parent = uuid4(), uuid4()
        manager._sessions_by_audio_id["a"] = instance
        manager._audio_ids_by_owner_id["fixture"] = ["a"]
        manager._audio_ids_by_jarvis_session_id[session_id] = "a"
        manager._jarvis_session_ids_by_audio_id["a"] = session_id
        self.assertTrue(await manager.speak_visual_interrupt(owner_id="fixture", audio_session_id="a",
            jarvis_session_id=session_id, source_received_at=datetime.now(UTC), message="Fixture.",
            interrupt_id=uuid4(), source_payload_id=parent))
        self.assertEqual(await manager.speak_prospective_interrupt(owner_id="fixture",
            preferred_jarvis_session_id=session_id, message="Fixture.", event_id=uuid4(),
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=AsyncMock(),
            source_payload_id=parent), session_id)
        self.assertEqual(instance.speak_visual_interrupt.await_args.kwargs["source_payload_id"], parent)
        self.assertEqual(instance.speak_prospective_interrupt.await_args.kwargs["source_payload_id"], parent)
        self.assertFalse(await manager.speak_visual_interrupt(owner_id="other", audio_session_id="a",
            jarvis_session_id=session_id, source_received_at=datetime.now(UTC), message="Foreign.",
            interrupt_id=uuid4(), source_payload_id=parent))
        self.assertIsNone(await manager.speak_prospective_interrupt(owner_id="other",
            preferred_jarvis_session_id=session_id, message="Foreign.", event_id=uuid4(),
            subject_type="visual_focus", subject_id=uuid4(), on_outcome=AsyncMock(), source_payload_id=parent))
        self.assertEqual(instance.speak_visual_interrupt.await_count, 1)
        self.assertEqual(instance.speak_prospective_interrupt.await_count, 1)

    async def test_failed_source_binding_releases_unsubmitted_search_claim(self):
        now, frame, subject, parent, token = datetime.now(UTC), uuid4(), uuid4(), uuid4(), uuid4()
        request = FastPerceptionRequest(owner_id="fixture", session_id=uuid4(), window_id=uuid4(),
            frames=(FastPerceptionFrame(frame, 0, now, "glasses", "image/jpeg", b"fixture"),),
            previous_working_state={}, existing_entity_ids=frozenset(), recent_audio_transcripts=(),
            behavior_instructions="", active_reminders=(), active_focus=None, recent_interrupts=(), requested_at=now)
        match = FastProspectiveMatch(subject_type="visual_focus", subject_id=subject, confidence=0.99,
            observed_condition="Synthetic match", supporting_frame_ids=[frame], suggested_message="Possible match.")
        repository = SimpleNamespace(try_claim_subject=AsyncMock(return_value=token),
            reserve_delivery_event=AsyncMock(side_effect=SourceError("fixture", "Source changed")),
            release_subject_claim=AsyncMock())
        manager = SimpleNamespace(speak_prospective_interrupt=AsyncMock())
        service = ProspectiveMemoryService(repository=repository,
            realtime_session_manager=manager, settings=JarvisSettings(_env_file=None))
        with self.assertRaises(SourceError):
            await service.process_matches(request=request, matches=[match], source_payload_id=parent)
        repository.release_subject_claim.assert_awaited_once_with(owner_id="fixture",
            subject_type="visual_focus", subject_id=subject, claim_token=token)
        manager.speak_prospective_interrupt.assert_not_awaited()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires manifested isolated database fixtures")
class VisualDeliverySourceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = observations.ObservationPublicationDatabaseTests.asyncSetUp
    frame_for = observations.ObservationPublicationDatabaseTests.frame_for
    observation = observations.ObservationPublicationDatabaseTests.observation
    insert = observations.ObservationPublicationDatabaseTests.insert
    source = observations.ObservationPublicationDatabaseTests.source

    async def visual(self, source, **changes):
        return await VisualFrameRepository(self.f.pool).create_visual_interrupt(**(dict(
            owner_id=self.f.owner, session_id=self.session, visual_frame_id=self.frame,
            message="Possible fixture match.", reason="Synthetic evidence", confidence=0.9,
            dedupe_bucket="fixture", source_payload_id=source.payload_id) | changes))

    async def tracked(self, source):
        repository = ProspectiveRepository(self.f.pool)
        focus = await repository.set_visual_focus(owner_id=self.f.owner, target="fixture prop",
            description="Synthetic harmless target", purpose="test", mode="find_once",
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        token = await repository.try_claim_subject(owner_id=self.f.owner, subject_type="visual_focus",
            subject_id=focus.id, claim_timeout_seconds=60)
        event = await repository.reserve_delivery_event(owner_id=self.f.owner, session_id=self.session,
            subject_type="visual_focus", subject_id=focus.id, claim_token=token, occurrence_key=uuid4().hex,
            message="Possible fixture match.", evidence={"observed_condition":"Synthetic match",
                "supporting_frame_ids":[str(self.frame)], "source_payload_id":str(source.payload_id)},
            dedupe_window_seconds=0, source_payload_id=source.payload_id)
        self.assertIsNotNone(event)
        return repository, focus, event

    async def remove(self, source):
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id, expected_revision=1)

    async def test_original_identity_is_owner_and_session_bound(self):
        observation = await self.insert()
        source = await self.source(observation)
        self.assertEqual(await observation_delivery_source(self.publisher, owner_id=self.f.owner,
            session_id=self.session, observation_id=observation.id), source.payload_id)
        for owner, session in [(self.f.other, self.session), (self.f.owner, uuid4())]:
            with self.assertRaises(SourceError):
                await observation_delivery_source(self.publisher, owner_id=owner,
                    session_id=session, observation_id=observation.id)

    async def test_delete_scrubs_only_bound_visual_copy_and_blocks_a_late_copy(self):
        source = await self.source(await self.insert())
        bound = await self.visual(source)
        legacy = await self.visual(source, source_payload_id=None, message="Independent legacy fixture.")
        await self.remove(source)
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow("select * from visual_interrupts where id=$1", bound.id)
            self.assertEqual((row["message"], row["reason"], row["dedupe_bucket"]), ("", "source_unavailable", ""))
            self.assertEqual(await connection.fetchval("select message from visual_interrupts where id=$1", legacy.id),
                             "Independent legacy fixture.")
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_visual_delivery_sources"), 1)
        with self.assertRaises(SourceError):
            await self.visual(source)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from visual_interrupts"), 2)

    async def test_foreign_frame_or_session_rejects_whole_visual_write(self):
        source = await self.source(await self.insert())
        for changes in [{"visual_frame_id":uuid4()}, {"session_id":uuid4()}, {"owner_id":self.f.other}]:
            with self.assertRaises(SourceError):
                await self.visual(source, **changes)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from visual_interrupts"), 0)
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_visual_delivery_sources"), 0)

    async def test_binding_guard_rejects_an_unrelated_notification_identity(self):
        source = await self.source(await self.insert())
        async with self.f.pool.acquire() as connection:
            with self.assertRaises(asyncpg.CheckViolationError):
                await connection.execute("""insert into jarvis_visual_delivery_sources
                    (owner_id,delivery_kind,delivery_id,session_id,payload_id) values($1,'visual',$2,$3,$4)""",
                    self.f.owner, uuid4(), self.session, source.payload_id)

    async def test_deleted_tracked_alert_cannot_complete_find_once_from_late_callback(self):
        source = await self.source(await self.insert())
        repository, focus, event = await self.tracked(source)
        self.assertTrue(await repository.begin_delivery(owner_id=self.f.owner, event_id=event.id,
            session_id=self.session, source_payload_id=source.payload_id))
        await self.remove(source)
        settled = await repository.resolve_delivery_event(owner_id=self.f.owner, event_id=event.id,
            outcome="delivered", provider_response_id="late-fixture", source_payload_id=source.payload_id)
        self.assertEqual(settled.event_type, "failed")
        self.assertEqual(settled.message, "[source unavailable]")
        self.assertEqual(settled.evidence, {})
        current = await repository.get_active_visual_focus(owner_id=self.f.owner)
        self.assertEqual(current.id, focus.id)
        self.assertIsNone(current.completed_at)
        self.assertIsNone(current.trigger_claim_token)
        self.assertFalse(await repository.begin_delivery(owner_id=self.f.owner, event_id=event.id,
            session_id=self.session, source_payload_id=source.payload_id))

    async def test_completed_search_keeps_historical_outcome_but_not_revoked_evidence(self):
        source = await self.source(await self.insert())
        repository, focus, event = await self.tracked(source)
        await repository.begin_delivery(owner_id=self.f.owner, event_id=event.id,
            session_id=self.session, source_payload_id=source.payload_id)
        await repository.resolve_delivery_event(owner_id=self.f.owner, event_id=event.id,
            outcome="delivered", provider_response_id="fixture", source_payload_id=source.payload_id)
        await self.remove(source)
        async with self.f.pool.acquire() as connection:
            row = await connection.fetchrow("select completed_at,completion_evidence from jarvis_visual_focuses where id=$1", focus.id)
            self.assertIsNotNone(row["completed_at"])
            self.assertEqual(json.loads(row["completion_evidence"])["status"], "source_unavailable")
            event_row = await connection.fetchrow("select event_type,evidence from jarvis_prospective_delivery_events where id=$1", event.id)
            self.assertEqual(event_row["event_type"], "delivered_to_session")
            self.assertEqual(json.loads(event_row["evidence"]), {})

    async def test_logical_forgetting_invalidates_notification_copies_without_erasing_original(self):
        source = await self.source(await self.insert())
        notification = await self.visual(source)
        repository, _, event = await self.tracked(source)
        async with self.f.sources.transaction(self.f.owner) as connection:
            await connection.execute("update jarvis_knowledge_sources set status='forgotten',generation=generation+1 where id=$1", source.source_id)
        async with self.f.pool.acquire() as connection:
            self.assertIsNotNone(await connection.fetchval("select content from jarvis_context_payloads where id=$1", source.payload_id))
            self.assertEqual(await connection.fetchval("select message from visual_interrupts where id=$1", notification.id), "")
        self.assertFalse(await repository.begin_delivery(owner_id=self.f.owner, event_id=event.id,
            session_id=self.session, source_payload_id=source.payload_id))

    async def test_queued_native_media_checks_source_inside_the_common_writer(self):
        source = await self.source(await self.insert())
        instance, events = voice(self.f.owner, self.session, self.f.pool)
        self.addAsyncCleanup(close_voice, instance)
        instance._source_context.observe((source.payload_id,))
        writer = instance._app_sender
        await writer._lock.acquire()
        pending = asyncio.create_task(instance._send_app_event("audio.output.delta",
            response_id="fixture", item_id="fixture", audio_base64="AAAA"))
        try:
            await asyncio.sleep(0)
            await self.remove(source)
        finally:
            writer._lock.release()
        self.assertFalse(await pending)
        self.assertTrue(instance._context_revoked)
        self.assertFalse(any(event["type"] == "audio.output.delta" for event in events))
        self.assertEqual(instance._native_playback, {})

    async def test_first_written_chunk_is_not_retracted_but_next_chunk_is_fenced(self):
        source = await self.source(await self.insert())
        instance, events = voice(self.f.owner, self.session, self.f.pool)
        self.addAsyncCleanup(close_voice, instance)
        instance._source_context.observe((source.payload_id,))
        self.assertTrue(await instance._send_app_event("audio.output.delta",
            response_id="fixture", item_id="fixture", audio_base64="AAAA"))
        await self.remove(source)
        self.assertFalse(await instance._send_app_event("audio.output.delta",
            response_id="fixture", item_id="fixture", audio_base64="AAAA"))
        self.assertEqual(len([e for e in events if e["type"]=="audio.output.delta"]), 1)

    async def test_stale_started_hint_does_not_leak_its_message(self):
        source = await self.source(await self.insert())
        instance, events = voice(self.f.owner, self.session, self.f.pool)
        self.addAsyncCleanup(close_voice, instance)
        instance._source_context.observe((source.payload_id,))
        await self.remove(source)
        await instance._send_speech_notification("visual.interrupt.started", message="Revoked fixture words.")
        self.assertFalse(any(e.get("message") == "Revoked fixture words." for e in events))

    async def current_fixture_frame(self):
        self.when = datetime.now(UTC)
        async with self.f.pool.acquire() as connection:
            await connection.execute("update visual_frames set captured_at=$2 where id=$1", self.frame, self.when)

    async def test_actual_pipeline_binds_generic_candidate_before_notification_admission(self):
        await self.current_fixture_frame()
        output = FastPerceptionOutput(summary="Synthetic possible match.", confidence=0.9, starts_new_event=False,
            interrupt_candidate=FastInterruptCandidate(message="Possible fixture match.",
                reason="Synthetic image interpretation", confidence=0.99, scope="general"))
        result = await run_visual_fixture(self.f, self.publisher, self.frame, output.summary,
            analyzer_output=output, notification_observer=AsyncMock())
        self.assertEqual([n["kind"] for n in result["notifications"]], ["visual"])
        candidate = result["notifications"][0]
        self.assertIsInstance(candidate["source_payload_id"], UUID)
        async with self.f.pool.acquire() as connection:
            binding = await connection.fetchrow(
                "select * from jarvis_visual_delivery_sources where delivery_id=$1", candidate["interrupt_id"])
            observation_id = await connection.fetchval(
                "select observation_id from jarvis_observation_sources where payload_id=$1", binding["payload_id"])
        self.assertEqual(binding["payload_id"], candidate["source_payload_id"])
        source = await self.source(SimpleNamespace(id=observation_id))
        await self.remove(source)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select message from visual_interrupts where id=$1",
                                                      candidate["interrupt_id"]), "")

    async def test_actual_pipeline_tracked_match_owns_the_candidate_without_generic_fallback(self):
        await self.current_fixture_frame()
        repository = ProspectiveRepository(self.f.pool)
        focus = await repository.set_visual_focus(owner_id=self.f.owner, target="fixture prop",
            description="Synthetic harmless target", purpose="test", mode="find_once",
            expires_at=datetime.now(UTC)+timedelta(minutes=10))
        output = FastPerceptionOutput(summary="Synthetic possible match.", confidence=0.9, starts_new_event=False,
            interrupt_candidate=FastInterruptCandidate(message="Duplicate generic candidate.",
                reason="Must not bypass tracked search", confidence=0.99, scope="general"),
            prospective_matches=[FastProspectiveMatch(subject_type="visual_focus", subject_id=focus.id,
                confidence=0.99, observed_condition="Synthetic match", supporting_frame_ids=[self.frame],
                suggested_message="Possible fixture match.")])
        result = await run_visual_fixture(self.f, self.publisher, self.frame, output.summary,
            analyzer_output=output, notification_observer=AsyncMock())
        self.assertEqual(str(result["requests"][0].active_focus["id"]), str(focus.id))
        self.assertEqual([n["kind"] for n in result["notifications"]], ["prospective"])
        candidate = result["notifications"][0]
        async with self.f.pool.acquire() as connection:
            binding = await connection.fetchrow(
                "select * from jarvis_visual_delivery_sources where delivery_id=$1", candidate["event_id"])
            self.assertEqual(await connection.fetchval("select count(*) from visual_interrupts"), 0)
            event = await connection.fetchrow(
                "select event_type,evidence from jarvis_prospective_delivery_events where id=$1", candidate["event_id"])
        self.assertEqual(binding["payload_id"], candidate["source_payload_id"])
        self.assertEqual(json.loads(event["evidence"])["source_payload_id"], str(candidate["source_payload_id"]))
        self.assertEqual(event["event_type"], "missed")
        self.assertIsNone((await repository.get_active_visual_focus(owner_id=self.f.owner)).completed_at)
