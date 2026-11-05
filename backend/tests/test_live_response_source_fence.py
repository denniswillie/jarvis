"""Response submission and native-output lineage; no provider inference."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.assistant.context import AssistantContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveContextError, LiveResponseSources
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.sessions.models import JarvisMessage
from datetime import UTC, datetime
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository
from app.tasks.repository import TaskRepository
from evals.second_brain.source_fixture import source_fixture


class LiveResponseSourceUnitTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_path_has_no_database_dependency(self):
        sources = LiveResponseSources(None, owner_id="fixture", session_id=uuid4(), audio_session_id="a")
        self.assertIsNone(await sources.prepare("request"))
        self.assertIsNone(sources.source_for("request"))

    async def test_writer_enters_guard_before_factory_and_holds_it_through_send(self):
        calls = []
        @asynccontextmanager
        async def guard():
            calls.append("enter")
            yield
            calls.append("exit")
        class Socket:
            async def send(self, payload):
                calls.append(("write", json.loads(payload)))
        connection = OpenAIRealtimeConnection(settings=JarvisSettings(), owner_id="fixture")
        connection._websocket = Socket()
        def factory():
            calls.append("factory")
            return {"type":"response.create"}
        self.assertTrue(await connection.send_event(factory, write_guard=guard))
        self.assertEqual(calls, ["enter", "factory", ("write", {"type":"response.create"}), "exit"])

    async def test_rejected_guard_never_evaluates_or_writes_payload(self):
        @asynccontextmanager
        async def guard():
            raise LiveContextError()
            yield
        class Socket:
            async def send(self, payload):
                raise AssertionError("Must not write revoked content")
        connection = OpenAIRealtimeConnection(settings=JarvisSettings(), owner_id="fixture")
        connection._websocket = Socket()
        def factory():
            raise AssertionError("Must not mark this response submitted")
        with self.assertRaises(LiveContextError):
            await connection.send_event(factory, write_guard=guard)

    async def test_capture_retains_bound_source_and_rejects_rebinding(self):
        events = []
        class Repository:
            async def append(self, owner, event):
                events.append(event)
        capture = SessionContextCapture(Repository(), owner_id="fixture", session_id=uuid4(),
            provider="openai", audio_session_id="a")
        source = uuid4()
        capture.bind_openai_response("r", None, source_payload_id=source)
        capture.observe_output({"type":"transcript.done","item_id":"i","response_id":"r","text":"Synthetic"},
            forwarded=True)
        self.assertTrue(await capture.flush())
        self.assertEqual(events[0].source_payload_ids, (source,))
        capture.bind_openai_response("r", None, source_payload_id=uuid4())
        self.assertEqual(capture.issue, "context_response_source_conflict")
        await capture.close()

    async def test_source_identity_and_bounds_are_not_model_selected(self):
        sources = LiveResponseSources(None, owner_id="fixture", session_id=uuid4(), audio_session_id="a",
            max_sources=1)
        source = uuid4()
        sources.observe([source, source])
        self.assertTrue(sources.has_sources)
        with self.assertRaises(LiveContextError):
            sources.observe([str(source)])
        with self.assertRaises(LiveContextError):
            sources.observe([uuid4()])
        with self.assertRaises(LiveContextError):
            await sources.prepare("request")

    def test_redacted_or_suppressed_messages_do_not_poison_rebuilt_context(self):
        source = uuid4()
        message = JarvisMessage(id=uuid4(), session_id=uuid4(), role="assistant", modality="audio",
            content="Synthetic", provider_item_id=None, provider_response_id=None,
            created_at=datetime.now(UTC), source_payload_id=source)
        context = SimpleNamespace(voice_continuity_source_ids=(), provider_source_payload_ids=(), recent_messages=[message,
            message.model_copy(update={"content":"", "source_payload_id":uuid4()}),
            message.model_copy(update={"source_payload_id":uuid4(),
                "metadata":{"realtime_output":{"delivery_disposition":"suppressed"}}})])
        self.assertEqual(AssistantContext.source_payload_ids.fget(context), (source,))


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Requires explicitly manifested isolated database fixtures")
class LiveResponseSourceDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        manager = source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id())
        self.f = await manager.__aenter__()
        self.addAsyncCleanup(manager.__aexit__, None, None, None)
        self.messages = SessionRepository(self.f.pool)
        self.session = await self.messages.create_session(self.f.owner, "Synthetic live source fence")
        self.sources = LiveResponseSources(self.f.pool, owner_id=self.f.owner, session_id=self.session.id,
            audio_session_id="fixture-audio")
        self.original = await self.f.publish()
        self.sources.observe([self.original.payload_id])

    async def remove(self, connection=None):
        if connection is not None:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.f.owner, self.original.payload_id)
        else:
            async with self.f.pool.acquire() as c:
                await self.remove(c)

    def voice(self):
        provider_writes, phone_writes = [], []
        class Provider:
            async def send(self, payload):
                provider_writes.append(json.loads(payload))
            async def close(self):
                pass
        class Phone:
            async def send_text(self, payload):
                phone_writes.append(json.loads(payload))
            async def close(self, **kwargs):
                pass
        capture = SessionContextCapture(ContextRepository(TaskRepository(self.f.pool, TaskLimits())),
            owner_id=self.f.owner, session_id=self.session.id, provider="openai", audio_session_id="fixture-audio")
        voice = RealtimeAudioSession(websocket=Phone(),
            context=RealtimeSessionContext(audio_session_id="fixture-audio", owner_id=self.f.owner,
                jarvis_session_id=self.session.id, client_kind="phone_audio", session=self.session),
            settings=JarvisSettings(), repository=self.messages, tool_modules=(), context_providers=(),
            context_capture=capture)
        voice.openai._websocket = Provider()
        voice._source_context.observe([self.original.payload_id])
        self.addAsyncCleanup(voice.close, "fixture_complete")
        return voice, provider_writes, phone_writes

    async def test_actual_session_refuses_stale_instruction_refresh(self):
        voice, writes, _ = self.voice()
        context = SimpleNamespace(source_payload_ids=(self.original.payload_id,),
            as_prompt_payload=lambda: {"fixture":"Synthetic source context"})
        await self.remove()
        with self.assertRaises(LiveContextError):
            await voice._send_complete_session_update(context)
        self.assertTrue(voice._context_revoked)
        self.assertEqual(writes, [])

    async def test_actual_session_binds_and_persists_native_followup(self):
        voice, provider_writes, phone_writes = self.voice()
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        self.assertTrue(await voice._send_reserved_response_locked(reservation))
        metadata = provider_writes[-1]["response"]["metadata"]
        await voice._register_response({"id":"fixture-response", "metadata":metadata})
        await voice._forward_output_media({"response_id":"fixture-response", "item_id":"fixture-item",
            "transcript":"Synthetic native follow-up."}, event_type="response.output_audio_transcript.done")
        await voice._handle_openai_event({"type":"response.done",
            "response":{"id":"fixture-response", "status":"completed", "output":[]}})
        self.assertTrue(await voice._context_capture.flush())
        message = (await self.messages.get_recent_messages(self.f.owner, self.session.id))[0]
        snapshot = voice._source_context.source_for(reservation.request_id)
        self.assertIsNotNone(message.source_payload_id)
        self.assertNotEqual(message.source_payload_id, snapshot)
        async with self.f.pool.acquire() as connection:
            self.assertTrue(await connection.fetchval("""select true from jarvis_context_payload_dependencies
                where owner_id=$1 and source_payload_id=$2 and dependent_payload_id=$3""",
                self.f.owner, snapshot, message.source_payload_id))
        self.assertTrue(phone_writes)
        await self.remove()
        self.assertEqual((await self.messages.get_recent_messages(self.f.owner, self.session.id))[0].content, "")
        next_turn = voice._reserve_response_locked("normal_user_turn", {})
        with self.assertRaises(LiveContextError):
            await voice._send_reserved_response_locked(next_turn)
        self.assertFalse(next_turn.submitted)
        self.assertTrue(voice._context_revoked)
        self.assertEqual(len([e for e in provider_writes if e["type"]=="response.create"]), 1)

    async def test_unbound_early_provider_output_is_not_forwarded_or_captured(self):
        voice, _, phone_writes = self.voice()
        await voice._forward_output_media({"response_id":"unbound", "item_id":"unbound",
            "transcript":"Unattributed output."}, event_type="response.output_audio_transcript.done")
        self.assertEqual(phone_writes, [])
        self.assertEqual(voice._context_capture._outputs, {})

    async def test_source_preparation_timeout_retires_instead_of_stalling_the_turn(self):
        voice, writes, _ = self.voice()
        voice.settings = voice.settings.model_copy(update={"OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS":0.03})
        async def stalled(request_id):
            await asyncio.Event().wait()
        voice._source_context.prepare = stalled
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        with self.assertRaises(TimeoutError):
            await asyncio.wait_for(voice._send_reserved_response_locked(reservation), 1)
        self.assertTrue(voice._context_revoked)
        self.assertFalse(reservation.submitted)
        self.assertEqual(writes, [])

    async def test_ancestor_removal_while_admission_waits_has_no_reverse_lock_cycle(self):
        # UUID order deliberately disagrees with database creation order.
        child = UUID(int=1)
        async with self.f.pool.acquire() as c:
            await c.execute("""insert into jarvis_context_payloads(id,owner_id,availability,content)
                values($1,$2,'available','{"fixture":"dependent"}'::jsonb)""", child, self.f.owner)
            await c.execute("""insert into jarvis_context_payload_dependencies
                (owner_id,source_payload_id,dependent_payload_id) values($1,$2,$3)""",
                self.f.owner, self.original.payload_id, child)
        self.sources.observe([child])
        snapshot = await self.sources.prepare("contention")
        entered = False
        async def admission():
            nonlocal entered
            async with self.sources.guard(snapshot):
                entered = True
        # Identify the actual reader after its real guard transaction starts.
        # Looking for any blocked backend can miss a newly established pooled
        # connection in a cached pg_stat_activity observation.
        reader_started = asyncio.Event()
        reader_pid = None
        original_lock = self.sources._lock
        async def observed_lock(connection, sources):
            nonlocal reader_pid
            reader_pid = await connection.fetchval("select pg_backend_pid()")
            reader_started.set()
            await original_lock(connection, sources)
        self.sources._lock = observed_lock
        pending = None
        try:
            async with self.f.pool.acquire() as parent, parent.transaction():
                blocker = await parent.fetchval("select pg_backend_pid()")
                await parent.fetchval("""select id from jarvis_context_payloads
                    where owner_id=$1 and id=$2 for update""", self.f.owner, self.original.payload_id)
                pending = asyncio.create_task(admission())
                await asyncio.wait_for(reader_started.wait(), 5)
                async def wait_for_database_block():
                    async with self.f.pool.acquire() as observer:
                        while not await observer.fetchval(
                            "select $1::int=any(pg_blocking_pids($2::int))", blocker, reader_pid):
                            if pending.done():
                                await pending
                                self.fail("Admission did not wait on its ancestor")
                            await asyncio.sleep(0.01)
                await asyncio.wait_for(wait_for_database_block(), 5)
                await asyncio.wait_for(self.remove(parent), 5)
            with self.assertRaises(LiveContextError):
                await asyncio.wait_for(pending, 5)
            self.assertFalse(entered)
        finally:
            self.sources._lock = original_lock
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    async def test_queued_response_cannot_cross_committed_removal(self):
        snapshot = await self.sources.prepare("request")
        connection = OpenAIRealtimeConnection(settings=JarvisSettings(), owner_id=self.f.owner)
        writes = []
        class Socket:
            async def send(self, value):
                writes.append(value)
        connection._websocket = Socket()
        await connection._send_lock.acquire()
        submitted = False
        def factory():
            nonlocal submitted
            submitted = True
            return {"type":"response.create"}
        pending = asyncio.create_task(connection.send_event(factory,
            write_guard=lambda: self.sources.guard(snapshot)))
        try:
            await self.remove()
        finally:
            connection._send_lock.release()
        with self.assertRaises(LiveContextError):
            await pending
        self.assertFalse(submitted)
        self.assertEqual(writes, [])

    async def test_later_native_copies_follow_the_same_original_removal(self):
        snapshot = await self.sources.prepare("request")
        message = await self.messages.create_message(owner_id=self.f.owner, session_id=self.session.id,
            role="assistant", modality="audio", content="Synthetic native follow-up.",
            source_payload_id=snapshot.payload_id)
        self.assertEqual(message.source_payload_id, snapshot.payload_id)
        tasks = TaskRepository(self.f.pool, TaskLimits())
        capture = SessionContextCapture(ContextRepository(tasks), owner_id=self.f.owner,
            session_id=self.session.id, provider="openai", audio_session_id="fixture-audio")
        self.addAsyncCleanup(capture.close)
        capture.bind_openai_response("r", None, source_payload_id=snapshot.payload_id)
        capture.observe_output({"type":"transcript.done","item_id":"i","response_id":"r",
            "text":"Synthetic native follow-up."}, forwarded=True)
        self.assertTrue(await capture.flush())
        await self.remove()
        history = await self.messages.get_recent_messages(self.f.owner, self.session.id)
        self.assertEqual(history[0].content, "")
        self.assertEqual(history[0].metadata["response_kind"], "source_bound_unavailable")
        async with self.f.pool.acquire() as c:
            copied = await c.fetch("""select p.availability,p.content from jarvis_context_events e
                join jarvis_context_payloads p on p.owner_id=e.owner_id and p.id=e.payload_id
                where e.owner_id=$1 and e.source_namespace=$2 and e.kind='assistant_output'""",
                self.f.owner, capture.namespace)
        self.assertTrue(copied)
        self.assertTrue(all(p["availability"]=="deleted" and p["content"] is None for p in copied))

    async def test_new_sources_cannot_slip_into_an_older_prepared_response(self):
        snapshot = await self.sources.prepare("request")
        other = await self.f.publish(source_key="second")
        self.sources.observe([other.payload_id])
        with self.assertRaises(LiveContextError) as error:
            async with self.sources.guard(snapshot):
                self.fail("Stale source set must not enter the writer")
        self.assertEqual(error.exception.code, "live_context_changed")

    async def test_foreign_source_and_unavailable_parent_fail_admission(self):
        foreign = LiveResponseSources(self.f.pool, owner_id=self.f.other, session_id=self.session.id,
            audio_session_id="foreign-audio")
        foreign.observe([self.original.payload_id])
        with self.assertRaises(LiveContextError):
            await foreign.prepare("foreign")
        await self.remove()
        with self.assertRaises(LiveContextError):
            await self.sources.prepare("removed")

    async def test_original_deletion_waits_for_an_already_started_guarded_write(self):
        snapshot = await self.sources.prepare("request")
        entered, release = asyncio.Event(), asyncio.Event()
        connection = OpenAIRealtimeConnection(settings=JarvisSettings(), owner_id=self.f.owner)
        class Socket:
            async def send(self, value):
                entered.set()
                await release.wait()
        connection._websocket = Socket()
        sending = asyncio.create_task(connection.send_event({"type":"response.create"},
            write_guard=lambda: self.sources.guard(snapshot)))
        await asyncio.wait_for(entered.wait(), 3)
        deletion = asyncio.create_task(self.remove())
        try:
            await asyncio.sleep(0.05)
            self.assertFalse(deletion.done())
        finally:
            release.set()
        self.assertTrue(await sending)
        await asyncio.wait_for(deletion, 5)
        with self.assertRaises(LiveContextError):
            async with self.sources.guard(snapshot):
                self.fail("Removal must win after the earlier write settles")

    async def test_ancestor_and_copy_lock_order_is_creation_order_not_uuid_order(self):
        snapshot = await self.sources.prepare("first")
        # A resumed context may contain both the original and a native copy.
        self.sources.observe([snapshot.payload_id])
        second = await self.sources.prepare("second")
        async with self.sources.guard(second):
            self.assertEqual(self.sources.source_for("second"), second.payload_id)
        await self.remove()
        with self.assertRaises(LiveContextError):
            async with self.sources.guard(second):
                self.fail("A removed ancestor invalidates the materialization")
