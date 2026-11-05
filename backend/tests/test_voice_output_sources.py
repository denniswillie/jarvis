"""Policy-admitted assistant copies through actual terminal voice events."""
import asyncio
import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.knowledge.publication import CanonicalSourcePublication
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture
from test_voice_transcript_sources import voice_session


AUDIO = base64.b64encode(b"\x01\x00" * 2400).decode()
TEXT = "Synthetic ordinary assistant reply."


async def start_reply(voice, *, response="reply", item="item", phase="final_answer"):
    reservation = voice._reserve_response_locked("normal_user_turn", {})
    assert await voice._send_reserved_response_locked(reservation)
    sent = json.loads(voice.openai._websocket.send.await_args.args[0])
    await voice._handle_openai_event({"type":"response.created",
        "response":{"id":response, "metadata":sent["response"]["metadata"]}})
    await voice._handle_openai_event({"type":"response.output_item.added", "response_id":response,
        "item":{"id":item, "type":"message", "phase":phase}})
    return reservation


async def transcript(voice, *, response="reply", item="item", text=TEXT, audio=False):
    if audio:
        await voice._handle_openai_event({"type":"response.output_audio.delta",
            "response_id":response, "item_id":item, "delta":AUDIO})
    await voice._handle_openai_event({"type":"response.output_audio_transcript.done",
        "response_id":response, "item_id":item, "transcript":text})


async def finish(voice, *, response="reply", status="completed"):
    await voice._handle_openai_event({"type":"response.done",
        "response":{"id":response, "status":status, "output":[]}})


class VoiceOutputSourceUnitTests(unittest.IsolatedAsyncioTestCase):
    def capture(self, *, provider="openai"):
        history = SimpleNamespace(append=AsyncMock())
        capture = SessionContextCapture(history, owner_id="fixture", session_id=uuid4(),
            provider=provider, audio_session_id="fixture-audio")
        self.addAsyncCleanup(capture.close)
        return capture, history

    def admission(self, **changes):
        return dict(response_id="reply", item_id="item", text=TEXT,
            metadata={"audio_session_id":"fixture-audio", "response_kind":"normal_user_turn",
                      "realtime_output":{"item_id":"item", "delivery_disposition":"forwarded"}}) | changes

    def output(self, capture, *, terminal=True, interrupted=False):
        capture.observe_output({"type":"transcript.done", "response_id":"reply", "item_id":"item",
            "text":TEXT}, forwarded=True)
        if terminal:
            capture.finish_response("reply", interrupted=interrupted, completed=True)

    async def test_exact_terminal_output_is_admitted_without_inventing_hearing(self):
        capture, history = self.capture()
        self.output(capture)
        self.assertTrue(capture.assistant_message(**self.admission()))
        self.assertTrue(await capture.flush())
        event = history.append.await_args.args[1]
        self.assertEqual(event.payload["text"], TEXT)
        self.assertEqual(event.payload["conversation_message"], self.admission()["metadata"])
        self.assertIsNone(event.payload["heard_transcript"])
        self.assertEqual(event.payload["playback_status"], "unknown")

    async def test_capture_alone_does_not_request_a_conversation_message(self):
        capture, history = self.capture()
        self.output(capture)
        self.assertTrue(await capture.flush())
        self.assertNotIn("conversation_message", history.append.await_args.args[1].payload)

    async def test_missing_mismatched_interrupted_partial_or_suppressed_output_is_denied(self):
        for case in ("missing", "wrong-response", "wrong-text", "partial", "interrupted", "suppressed", "gemini", "tts"):
            with self.subTest(case=case):
                capture, history = self.capture(provider="gemini" if case == "gemini" else "openai")
                if case != "missing":
                    self.output(capture, terminal=case != "partial", interrupted=case == "interrupted")
                admission = self.admission()
                if case == "wrong-response":
                    admission["response_id"] = "different"
                if case == "wrong-text":
                    admission["text"] = "Not the captured reply"
                if case == "suppressed":
                    admission["metadata"]["realtime_output"]["delivery_disposition"] = "suppressed"
                if case == "tts":
                    capture._outputs["item"].provenance = {"audio_provider":"openai_tts"}
                self.assertFalse(capture.assistant_message(**admission))
                self.assertFalse(await capture.flush())
                self.assertTrue(all("conversation_message" not in call.args[1].payload
                                    for call in history.append.await_args_list))

    async def test_terminal_event_does_not_wait_for_canonical_write_or_use_raw_fallback(self):
        capture, history = self.capture()
        entered, release = asyncio.Event(), asyncio.Event()
        async def append(owner, event):
            entered.set()
            await release.wait()
        history.append = append
        messages = SimpleNamespace(create_message=AsyncMock())
        voice, phone = voice_session(capture.owner_id, capture.session_id, messages, capture)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        try:
            await start_reply(voice)
            await transcript(voice)
            await asyncio.wait_for(finish(voice), 1)
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(any(json.loads(call.args[0])["type"] == "response.done"
                                for call in phone.send_text.await_args_list))
            messages.create_message.assert_not_awaited()
        finally:
            release.set()
        self.assertTrue(await capture.flush())

    async def test_persistence_failure_does_not_create_independent_message(self):
        capture, history = self.capture()
        async def append(owner, event):
            if event.kind == "assistant_output":
                raise RuntimeError("Synthetic canonical write failure")
        history.append = append
        messages = SimpleNamespace(create_message=AsyncMock())
        voice, _ = voice_session(capture.owner_id, capture.session_id, messages, capture)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        await start_reply(voice)
        await transcript(voice)
        await finish(voice)
        self.assertFalse(await capture.flush())
        messages.create_message.assert_not_awaited()


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class VoiceOutputSourceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        manager = source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.f = await manager.__aenter__()
        self.addAsyncCleanup(manager.__aexit__, None, None, None)
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic assistant copy")
        self.history = ContextRepository(TaskRepository(self.f.pool, TaskLimits()),
            publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.voice, self.capture = self.make_voice()

    def make_voice(self, *, audio_session="fixture-audio"):
        capture = SessionContextCapture(self.history, owner_id=self.f.owner,
            session_id=self.session.id, provider="openai", audio_session_id=audio_session, write_timeout=10)
        voice, _ = voice_session(self.f.owner, self.session.id, self.messages, capture)
        self.addAsyncCleanup(voice.close, "fixture_complete")
        return voice, capture

    async def rows(self):
        return await self.messages.get_recent_messages(self.f.owner, self.session.id)

    async def reply(self, *, voice=None, capture=None, **changes):
        voice, capture = voice or self.voice, capture or self.capture
        reservation = await start_reply(voice)
        await transcript(voice, **changes)
        await finish(voice)
        self.assertTrue(await capture.flush(), capture.diagnostics())
        return reservation

    async def current_output(self, *, item="item", capture=None):
        capture = capture or self.capture
        records = await self.history.archive(self.f.owner, self.session.id)
        return next(e for e in records if e.kind == "assistant_output" and e.is_current
                    and e.source_namespace == capture.namespace
                    and e.source_event_id == capture._source("output", item))

    async def source(self, event):
        async with self.f.pool.acquire() as connection:
            identity = await connection.fetchval("""select source_id from jarvis_knowledge_context_sources
                where owner_id=$1 and event_id=$2""", self.f.owner, event.id)
        self.assertIsNotNone(identity)
        return await self.f.sources.read(self.f.owner, self.f.area, identity)

    async def remove_output(self, *, item="item", capture=None):
        source = await self.source(await self.current_output(item=item, capture=capture))
        await self.f.sources.delete_source(self.f.owner, self.f.area, source.source_id,
            expected_revision=source.revision)

    async def test_own_output_deletion_scrubs_message_and_preserves_unrelated_original(self):
        control = await self.f.publish("Synthetic independent original", source_key="control")
        await self.reply()
        message, = await self.rows()
        output = await self.current_output()
        self.assertEqual(message.content, TEXT)
        self.assertEqual(message.source_payload_id, output.payload_id)
        self.assertEqual(message.provider_response_id, "reply")
        self.assertEqual(message.metadata["realtime_output"]["delivery_disposition"], "forwarded")
        self.assertIsNone(output.content["heard_transcript"])
        await self.remove_output()
        self.assertEqual((await self.rows())[0].content, "")
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, control.source_id)).content["text"],
                         "Synthetic independent original")

    async def test_reply_with_input_snapshot_depends_on_its_own_canonical_output(self):
        upstream = await self.f.publish("Synthetic input evidence", source_key="upstream")
        self.voice._source_context.observe([upstream.payload_id])
        reservation = await self.reply()
        message, = await self.rows()
        snapshot = self.voice._source_context.source_for(reservation.request_id)
        self.assertNotEqual(message.source_payload_id, snapshot)
        async with self.f.pool.acquire() as connection:
            self.assertTrue(await connection.fetchval("""select true from jarvis_context_payload_dependencies
                where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3""",
                self.f.owner, snapshot, message.source_payload_id))
        await self.remove_output()
        self.assertEqual((await self.rows())[0].content, "")
        self.assertEqual((await self.f.sources.read(self.f.owner, self.f.area, upstream.source_id)).content["text"],
                         "Synthetic input evidence")

    async def test_input_ancestor_removal_still_scrubs_the_reply(self):
        upstream = await self.f.publish("Synthetic input evidence", source_key="upstream")
        self.voice._source_context.observe([upstream.payload_id])
        await self.reply()
        await self.f.sources.delete_source(self.f.owner, self.f.area, upstream.source_id,
            expected_revision=upstream.revision)
        self.assertEqual((await self.rows())[0].content, "")
        self.assertEqual((await self.current_output()).availability, "deleted")

    async def test_playback_revisions_and_duplicate_terminal_events_do_not_duplicate_or_resurrect(self):
        await self.reply(audio=True)
        first, = await self.rows()
        await finish(self.voice)
        for samples, state in ((0, "started"), (800, "progress"), (1600, "progress")):
            self.assertTrue(self.capture.playback_receipt("item", response_id="reply", samples=samples, state=state))
            self.assertTrue(await self.capture.flush())
        rows = await self.rows()
        self.assertEqual([(m.id, m.source_payload_id, m.content) for m in rows],
                         [(first.id, first.source_payload_id, TEXT)])
        current = await self.current_output()
        self.assertNotEqual(current.payload_id, first.source_payload_id)
        self.assertIsNone(current.content["heard_transcript"])
        await self.remove_output()
        self.assertTrue(self.capture.playback_receipt("item", response_id="reply", samples=2400, state="completed"))
        self.assertTrue(await self.capture.flush())
        rows = await self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].id, rows[0].content), (first.id, ""))
        self.assertIsNone(self.capture.issue)

    async def test_same_words_in_distinct_output_items_are_not_text_deduplicated(self):
        await start_reply(self.voice)
        await transcript(self.voice)
        await self.voice._handle_openai_event({"type":"response.output_item.added", "response_id":"reply",
            "item":{"id":"other-item", "type":"message", "phase":"final_answer"}})
        await transcript(self.voice, item="other-item")
        await finish(self.voice)
        self.assertTrue(await self.capture.flush())
        rows = await self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual({m.provider_item_id for m in rows}, {"item", "other-item"})
        self.assertEqual({m.content for m in rows}, {TEXT})
        await self.remove_output(item="item")
        self.assertEqual({m.provider_item_id:m.content for m in await self.rows()},
                         {"item":"", "other-item":TEXT})

    async def test_provider_item_reuse_in_a_new_audio_session_has_distinct_identity(self):
        await self.reply()
        voice, capture = self.make_voice(audio_session="second-connection")
        await self.reply(voice=voice, capture=capture)
        rows = await self.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({m.source_payload_id for m in rows}), 2)
        await self.remove_output(capture=self.capture)
        self.assertEqual({m.metadata.get("audio_session_id"):m.content for m in await self.rows() if m.content},
                         {"second-connection":TEXT})

    async def test_suppressed_commentary_and_interrupted_reply_do_not_become_messages(self):
        await start_reply(self.voice, phase="commentary")
        await transcript(self.voice)
        await finish(self.voice)
        self.assertTrue(await self.capture.flush())
        self.assertEqual(await self.rows(), [])
        await start_reply(self.voice, response="interrupted", item="interrupted-item")
        await transcript(self.voice, response="interrupted", item="interrupted-item")
        self.voice._active_responses["interrupted"].interrupted = True
        await finish(self.voice, response="interrupted", status="cancelled")
        self.assertTrue(await self.capture.flush())
        self.assertEqual(await self.rows(), [])

    async def test_unsolicited_capture_and_unfinished_output_are_not_projected(self):
        self.capture.observe_output({"type":"transcript.done", "response_id":"unadmitted",
            "item_id":"unadmitted-item", "text":"Generated but not policy-admitted"}, forwarded=False)
        self.capture.finish_response("unadmitted", interrupted=False, completed=True)
        self.assertTrue(await self.capture.flush())
        self.assertEqual(await self.rows(), [])
        await start_reply(self.voice)
        await transcript(self.voice)
        await self.voice.close("fixture_disconnect")
        self.assertEqual(await self.rows(), [])

    async def test_failed_but_forwarded_partial_reply_keeps_incomplete_provenance(self):
        await start_reply(self.voice)
        await transcript(self.voice)
        await finish(self.voice, status="failed")
        self.assertTrue(await self.capture.flush())
        message, = await self.rows()
        output = await self.current_output()
        self.assertEqual(message.source_payload_id, output.payload_id)
        self.assertEqual(output.content["completion_state"], "incomplete")
        self.assertIsNone(output.content["heard_transcript"])

    async def test_projection_failure_rolls_back_final_output_and_message(self):
        await start_reply(self.voice)
        await transcript(self.voice)
        with patch.object(SessionRepository, "create_message", new=AsyncMock(side_effect=RuntimeError("fixture failure"))):
            await finish(self.voice)
            self.assertFalse(await self.capture.flush())
        self.assertEqual(await self.rows(), [])
        self.assertEqual(self.capture.issue, "context_capture_persistence_failed")
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("""select count(*) from jarvis_context_payloads
                where owner_id=$1 and content ? 'conversation_message'""", self.f.owner), 0)

    async def test_removal_before_queued_projection_prevents_late_copy(self):
        upstream = await self.f.publish("Synthetic input evidence", source_key="upstream")
        self.voice._source_context.observe([upstream.payload_id])
        entered, release = asyncio.Event(), asyncio.Event()
        append = self.history.append
        async def delayed(owner, event):
            if event.payload.get("conversation_message") is not None:
                entered.set()
                await release.wait()
            return await append(owner, event)
        self.history.append = delayed
        try:
            await start_reply(self.voice)
            await transcript(self.voice)
            await finish(self.voice)
            await asyncio.wait_for(entered.wait(), 3)
            await self.f.sources.delete_source(self.f.owner, self.f.area, upstream.source_id,
                expected_revision=upstream.revision)
        finally:
            release.set()
        self.assertTrue(await self.capture.flush())
        self.assertEqual(await self.rows(), [])
        self.assertIsNone(self.capture.issue)

    async def test_archived_area_keeps_bound_reply_without_memory_publication(self):
        async with self.f.pool.acquire() as connection:
            await connection.execute("""update jarvis_knowledge_areas set status='archived',revision=revision+1
                where owner_id=$1 and id=$2""", self.f.owner, self.f.area)
        await self.reply()
        message, = await self.rows()
        self.assertIsNotNone(message.source_payload_id)
        async with self.f.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("select count(*) from jarvis_knowledge_sources"), 0)
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, message.source_payload_id)
        self.assertEqual((await self.rows())[0].content, "")
