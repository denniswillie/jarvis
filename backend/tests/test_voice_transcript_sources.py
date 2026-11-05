"""Actual voice ingress and isolated DB evidence for source-bound user copies."""
import asyncio
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.publication import CanonicalSourcePublication
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskError
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


def voice_session(owner, session, repository, capture):
    phone = SimpleNamespace(send_text=AsyncMock(), close=AsyncMock())
    voice = RealtimeAudioSession(
        websocket=phone,
        context=RealtimeSessionContext(audio_session_id=capture.audio_session_id if capture else "legacy",
            owner_id=owner, jarvis_session_id=session, client_kind="phone_audio", session=SimpleNamespace()),
        settings=JarvisSettings(_env_file=None), repository=repository,
        tool_modules=(), context_providers=(), context_capture=capture,
    )
    voice.openai._websocket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    return voice, phone


class VoiceTranscriptSourceUnitTests(unittest.IsolatedAsyncioTestCase):
    def voice(self, *, capture_enabled=True, append=None):
        owner, session = "fixture", uuid4()
        history = SimpleNamespace(append=append or AsyncMock())
        capture = SessionContextCapture(history, owner_id=owner, session_id=session,
            provider="openai", audio_session_id="fixture") if capture_enabled else None
        messages = SimpleNamespace(create_message=AsyncMock())
        voice, phone = voice_session(owner, session, messages, capture)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        return voice, phone, messages, capture, history

    async def transcript(self, voice, item="input", text="Synthetic spoken input"):
        await voice._handle_openai_event({"type":"conversation.item.input_audio_transcription.completed",
            "item_id":item, "transcript":text})

    async def test_capture_never_also_writes_an_independent_user_message(self):
        voice, _, messages, capture, history = self.voice()
        await self.transcript(voice)
        self.assertTrue(await capture.flush())
        messages.create_message.assert_not_awaited()
        event = history.append.await_args.args[1]
        self.assertEqual(event.kind, "user_transcript")
        self.assertEqual(event.payload["text"], "Synthetic spoken input")

    async def test_capture_write_does_not_block_transcript_ingress(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def append(owner, event):
            entered.set()
            await release.wait()
        voice, phone, messages, capture, _ = self.voice(append=append)
        try:
            await asyncio.wait_for(self.transcript(voice), 1)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(any(json.loads(call.args[0])["type"] == "user.transcript.done"
                                for call in phone.send_text.await_args_list))
            messages.create_message.assert_not_awaited()
        finally:
            release.set()
        self.assertTrue(await capture.flush())

    async def test_capture_failure_has_no_legacy_persistence_fallback(self):
        async def append(owner, event):
            if event.kind == "user_transcript":
                raise RuntimeError("Synthetic database failure")
        voice, _, messages, capture, _ = self.voice(append=append)
        await self.transcript(voice)
        self.assertFalse(await capture.flush())
        self.assertEqual(capture.issue, "context_capture_persistence_failed")
        messages.create_message.assert_not_awaited()

    async def test_missing_provider_identity_records_gap_not_untracked_text(self):
        voice, _, messages, capture, history = self.voice()
        await self.transcript(voice, item=None)
        self.assertFalse(await capture.flush())
        self.assertEqual(capture.issue, "context_transcript_identity_missing")
        messages.create_message.assert_not_awaited()
        self.assertEqual([c.args[1].kind for c in history.append.await_args_list], ["context_gap"])
        self.assertNotIn("Synthetic spoken input", repr(history.append.await_args_list))

    async def test_disabled_capture_preserves_existing_legacy_path(self):
        voice, _, messages, _, _ = self.voice(capture_enabled=False)
        await self.transcript(voice)
        messages.create_message.assert_awaited_once()
        self.assertEqual(messages.create_message.await_args.kwargs["content"], "Synthetic spoken input")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class VoiceTranscriptSourceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        manager = source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.f = await manager.__aenter__()
        self.addAsyncCleanup(manager.__aexit__, None, None, None)
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic spoken source")
        self.history = ContextRepository(TaskRepository(self.f.pool, TaskLimits()),
            publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.voice, self.capture = self.make_voice()

    def make_voice(self, *, audio_session_id="fixture-audio", history=None):
        capture = SessionContextCapture(history or self.history, owner_id=self.f.owner,
            session_id=self.session.id, provider="openai", audio_session_id=audio_session_id, write_timeout=10)
        voice, _ = voice_session(self.f.owner, self.session.id, self.messages, capture)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        return voice, capture

    async def say(self, item="first", text="Synthetic first report", *, voice=None, capture=None):
        voice, capture = voice or self.voice, capture or self.capture
        await voice._handle_openai_event({"type":"input_audio_buffer.committed", "item_id":item})
        await voice._handle_openai_event({"type":"conversation.item.input_audio_transcription.completed",
            "item_id":item, "transcript":text})
        self.assertTrue(await capture.flush(), capture.diagnostics())

    async def messages_by_id(self):
        return {m.id:m for m in await self.messages.get_recent_messages(self.f.owner, self.session.id)}

    async def source_for(self, payload):
        async with self.f.pool.acquire() as connection:
            source = await connection.fetchval("""select source_id from jarvis_knowledge_source_versions
                where owner_id=$1 and area_id=$2 and payload_id=$3""", self.f.owner, self.f.area, payload)
        self.assertIsNotNone(source)
        return await self.f.sources.read(self.f.owner, self.f.area, source)

    def ingress(self, *, receipt="first", text="Synthetic first report", **changes):
        return ContextIngress(**(dict(session_id=self.session.id, source_namespace="voice:fixture-replay",
            source_event_id="input:fixture", receipt_id=receipt, kind="user_transcript", trust="user_input",
            source_time=datetime(2026, 9, 17, 12, tzinfo=UTC), payload={"audio_provider":"openai",
                "audio_session_id":"fixture-replay", "provider_item_id":"item", "text":text}) | changes))

    async def test_actual_spoken_source_deletion_scrubs_only_its_message(self):
        await self.say()
        await self.say("second", "Synthetic unrelated report")
        before = await self.messages_by_id()
        by_item = {m.provider_item_id:m for m in before.values()}
        self.assertEqual(len(before), 2)
        first, control = by_item["first"], by_item["second"]
        self.assertEqual(first.content, "Synthetic first report")
        self.assertTrue(all(m.source_payload_id is not None for m in before.values()))
        source = await self.source_for(first.source_payload_id)
        self.assertEqual(source.content["text"], first.content)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id,
            expected_revision=source.revision)
        after = await self.messages_by_id()
        self.assertEqual(after[first.id].content, "")
        self.assertEqual(after[first.id].source_payload_id, first.source_payload_id)
        self.assertEqual(after[control.id].content, control.content)
        event = await self.history.get(self.f.owner, source.context_event_id)
        self.assertEqual((event.availability, event.content), ("deleted", None))

    async def test_duplicate_callbacks_and_concurrent_receipt_replay_make_one_copy(self):
        await self.say()
        await self.say()
        self.assertEqual(len(await self.messages_by_id()), 1)
        event = self.ingress()
        first, second = await asyncio.gather(self.history.append(self.f.owner, event),
                                             self.history.append(self.f.owner, event))
        self.assertEqual(first.id, second.id)
        rows = [m for m in (await self.messages_by_id()).values() if m.source_payload_id == first.payload_id]
        self.assertEqual(len(rows), 1)

    async def test_revisions_and_old_receipt_replay_do_not_restore_obsolete_message(self):
        first_request = self.ingress()
        first = await self.history.append(self.f.owner, first_request)
        second = await self.history.append(self.f.owner, self.ingress(receipt="second",
            text="Synthetic corrected report", finality="revised"))
        third = await self.history.append(self.f.owner, self.ingress(receipt="third", finality="revised"))
        replay = await self.history.append(self.f.owner, first_request)
        self.assertEqual(replay.id, first.id)
        messages = {m.source_payload_id:m for m in (await self.messages_by_id()).values()}
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[first.payload_id].content, "")
        self.assertEqual(messages[second.payload_id].content, "")
        self.assertEqual(messages[third.payload_id].content, first_request.payload["text"])
        self.assertEqual((await self.history.get(self.f.owner, first.id)).content["text"], first_request.payload["text"])
        source = await self.source_for(third.payload_id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id,
            expected_revision=source.revision)
        self.assertTrue(all(not m.content for m in (await self.messages_by_id()).values()))

    async def test_late_transcript_keeps_exact_commit_and_provider_item_identity(self):
        self.capture.admit_openai_commit("request-first", 1)
        self.capture.admit_openai_commit("request-second", 2)
        for item in ("provider-first", "provider-second"):
            await self.voice._handle_openai_event({"type":"input_audio_buffer.committed", "item_id":item})
        await self.say("provider-second", "Later request arrives first")
        await self.say("provider-first", "Earlier request arrives late")
        messages = {m.provider_item_id:m for m in (await self.messages_by_id()).values()}
        self.assertEqual(set(messages), {"provider-first", "provider-second"})
        for item, request in (("provider-first", "request-first"), ("provider-second", "request-second")):
            source = await self.source_for(messages[item].source_payload_id)
            event = await self.history.get(self.f.owner, source.context_event_id)
            self.assertEqual(event.source_event_id, self.capture._source("input", "commit:" + request))
            self.assertEqual(event.content["text"], messages[item].content)

    async def test_archived_area_keeps_bound_conversation_without_memory_publication(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_knowledge_areas set status='archived',revision=revision+1
                where owner_id=$1 and id=$2""", self.f.owner, self.f.area)
        await self.say()
        message, = (await self.messages_by_id()).values()
        self.assertIsNotNone(message.source_payload_id)
        self.assertEqual(message.content, "Synthetic first report")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_knowledge_sources
                where owner_id=$1""", self.f.owner), 0)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, message.source_payload_id)
        self.assertEqual((await self.messages_by_id())[message.id].content, "")

    async def test_message_write_failure_rolls_back_transcript_and_publication(self):
        marker = "Synthetic rollback-only private transcript"
        with patch.object(SessionRepository, "create_message", new=AsyncMock(side_effect=RuntimeError("fixture failure"))):
            await self.voice._handle_openai_event({"type":"input_audio_transcription.completed",
                "item_id":"failed", "transcript":marker})
            self.assertFalse(await self.capture.flush())
        self.assertEqual(await self.messages_by_id(), {})
        self.assertEqual(self.capture.issue, "context_capture_persistence_failed")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payloads
                where owner_id=$1 and content->>'text'=$2""", self.f.owner, marker), 0)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_events
                where owner_id=$1 and kind='user_transcript'""", self.f.owner), 0)
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_knowledge_source_versions
                where owner_id=$1 and source_type='user_transcript'""", self.f.owner), 0)

    async def test_removed_input_rejects_late_revision_without_stopping_other_capture(self):
        await self.say()
        message, = (await self.messages_by_id()).values()
        source = await self.source_for(message.source_payload_id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id,
            expected_revision=source.revision)
        await self.say(text="Late revised text must not return")
        await self.say("other", "Unaffected next input")
        after = await self.messages_by_id()
        self.assertEqual(len(after), 2)
        self.assertEqual(after[message.id].content, "")
        self.assertEqual({m.content for m in after.values()}, {"", "Unaffected next input"})
        self.assertIsNone(self.capture.issue)

    async def test_same_provider_id_in_different_audio_sessions_is_not_deduplicated(self):
        await self.say(text="First connection input")
        voice, capture = self.make_voice(audio_session_id="second-connection")
        await self.say(text="Second connection input", voice=voice, capture=capture)
        messages = list((await self.messages_by_id()).values())
        self.assertEqual(len(messages), 2)
        self.assertEqual({m.provider_item_id for m in messages}, {"first"})
        self.assertEqual(len({m.source_payload_id for m in messages}), 2)
        self.assertEqual({m.metadata["audio_session_id"] for m in messages},
                         {"fixture-audio", "second-connection"})

    async def test_failed_or_empty_transcription_does_not_create_a_message(self):
        await self.voice._handle_openai_event({"type":"input_audio_buffer.committed", "item_id":"missing"})
        await self.voice._handle_openai_event({"type":"conversation.item.input_audio_transcription.failed",
            "item_id":"missing"})
        await self.say("empty", "  ")
        self.assertEqual(await self.messages_by_id(), {})
        events = await self.history.archive(self.f.owner, self.session.id)
        self.assertTrue(events)
        self.assertTrue(all(e.finality == "unavailable" for e in events))

    async def test_publisher_disabled_still_binds_message_to_canonical_original(self):
        history = ContextRepository(TaskRepository(self.f.pool, TaskLimits()))
        voice, capture = self.make_voice(audio_session_id="no-publisher", history=history)
        await self.say(voice=voice, capture=capture)
        message, = (await self.messages_by_id()).values()
        self.assertIsNotNone(message.source_payload_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources"), 0)
            await connection.execute("""update jarvis_context_payloads set availability='expired',content=null
                where owner_id=$1 and id=$2""", self.f.owner, message.source_payload_id)
        self.assertEqual((await self.messages_by_id())[message.id].content, "")

    async def test_foreign_session_cannot_create_canonical_or_conversation_copy(self):
        with self.assertRaises(TaskError) as error:
            await self.history.append(self.f.other, self.ingress())
        self.assertEqual(error.exception.code, "context_session_not_found")
        self.assertEqual(await self.messages_by_id(), {})

    async def test_replay_of_deleted_receipt_never_recreates_message(self):
        request = self.ingress()
        record = await self.history.append(self.f.owner, request)
        source = await self.source_for(record.payload_id)
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id,
            expected_revision=source.revision)
        replay = await self.history.append(self.f.owner, request)
        self.assertEqual((replay.id, replay.availability), (record.id, "deleted"))
        messages = list((await self.messages_by_id()).values())
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "")
