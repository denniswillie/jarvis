"""Opt-in real-SQL attribution checks; no provider, model or worker execution.

The existing canonical fixture owns migrations, exact random owners/schema and
independent cleanup. Task effects and capture use real repositories. Only ambient
readiness and the encoded provider socket are controlled. These checks do not
qualify task completion, announcement suppression, native playback or hearing.
"""
import asyncio
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.experience.repository import ToolExecutionRepository
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskStatus, TaskVoiceSource
from app.tasks.voice_tools import BackgroundTaskToolModule
import test_second_brain_context_integration as canonical_fixture
from test_realtime_unwritten_followup_wire import ContextReadyBoundary


SELECTOR = "jarvis_request_source"
OLD_TEXT = "Cancel the background task named Workbench Plan. Do not restart it."
CURRENT_TEXT = "What is two plus three? This does not retract my earlier instruction."


class EncodedSocket:
    def __init__(self):
        self.events = []

    async def send(self, encoded):
        self.events.append(json.loads(encoded))

    async def close(self):
        pass


@unittest.skipUnless(
    os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
    and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
    "Explicit manifested second-brain SQL fixture required; no default DSN",
)
class UnwrittenRequestSourceSQLTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Reuse the existing lifecycle verbatim, including migration-before-pool
        # ownership and DatabaseFixture's independent public-owner/schema audit.
        await canonical_fixture.CanonicalContextDatabaseTests.asyncSetUp(self)
        self.mutations = []
        self.executions = ToolExecutionRepository(self.pool)
        seed = await self.context.append(self.owner, ContextIngress(
            session_id=self.session, source_namespace="fixture:unwritten-sql",
            source_event_id="original-task", receipt_id="original-task-final",
            kind="user_request", trust="user_input", finality="final",
            payload={"text": "Prepare a hypothetical Workbench Plan; no external actions."},
        ))
        self.task = await self.tasks.create(self.owner, TaskCreate(
            request_id=uuid4(), title="Workbench Plan", requested_outcome="Compare hypothetical layouts only.",
            session_id=self.session,
        ), model="fixture-no-model", source=TaskVoiceSource(seed.id, seed.revision, self.session), mode="background")
        self.case = await self.open_case(self.owner, self.session)

    async def open_case(self, owner, session_id):
        async def mutate(actual_owner, task_id, operation, request, *, source=None,
                         suppress_result_announcement=False):
            # The disjoint suppression feature is deliberately outside this test.
            self.assertFalse(suppress_result_announcement)
            self.mutations.append((actual_owner, task_id, operation, request, source))
            return await self.tasks.mutate(actual_owner, task_id, operation, request, source=source)

        service = SimpleNamespace(repository=self.tasks, mutate=mutate,
                                  provider_configured=lambda kind: False)
        audio_id = str(uuid4())
        capture = SessionContextCapture(self.context, owner_id=owner, session_id=session_id,
            provider="openai", audio_session_id=audio_id, write_timeout=15)
        self.addAsyncCleanup(capture.close)
        settings = JarvisSettings(_env_file=None, OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=15)
        provider = OpenAIRealtimeConnection(settings=settings, owner_id=owner)
        socket = EncodedSocket()
        provider._websocket = socket
        provider.capture_mode, provider.transport = "auto", "webrtc"
        voice = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id=audio_id, owner_id=owner,
                jarvis_session_id=session_id, client_kind="rayban_audio", session=SimpleNamespace()),
            settings=settings, repository=SessionRepository(self.pool),
            tool_modules=(BackgroundTaskToolModule(service),), context_providers=(),
            context_capture=capture, tool_execution_repository=self.executions,
        )
        voice.openai = provider
        voice._realtime_context_ready.set()
        voice._send_app_event = AsyncMock(return_value=True)
        route = ContextReadyBoundary()
        voice._voice_route_runtime = route

        async def close():
            voice._voice_route_runtime = None  # Readiness fixture owns no runtime resources.
            await voice.close("unwritten_source_sql_complete")
        self.addAsyncCleanup(close)
        return SimpleNamespace(voice=voice, capture=capture, provider=provider, socket=socket,
                               route=route, owner=owner, session_id=session_id)

    async def until(self, predicate):
        async with asyncio.timeout(20):
            while not predicate():
                await asyncio.sleep(0.005)

    async def event(self, case, kind, **fields):
        await case.voice._handle_openai_event({"type": kind, **fields})

    def creates(self, case):
        return [event for event in case.socket.events if event["type"] == "response.create"]

    async def commit(self, case, item):
        await self.event(case, "input_audio_buffer.speech_stopped", item_id=item)
        await self.event(case, "input_audio_buffer.committed", item_id=item)

    async def final(self, case, item, text):
        await self.event(case, "conversation.item.input_audio_transcription.completed",
                         item_id=item, transcript=text)
        self.assertTrue(await case.capture.flush())
        self.assertIsNone(case.capture.issue)

    async def offered_pair(self, case):
        await self.event(case, "input_audio_buffer.speech_started", item_id="old")
        await self.commit(case, "old")
        await self.until(case.route.entered.is_set)
        self.assertTrue(await case.capture.flush())
        identity = case.capture._generations[1]
        # Real persisted pending input is not a finalized user authorization.
        with self.assertRaises(TaskError) as pending:
            await self.context.request_evidence(case.owner, case.session_id,
                source_namespace=case.capture.namespace,
                source_event_id=case.capture._source("input", identity))
        self.assertEqual(pending.exception.code, "context_request_not_finalized")
        await self.final(case, "old", OLD_TEXT)
        preparation = case.voice._response_preparation_task
        reservation = case.voice._pending_response
        await self.event(case, "input_audio_buffer.speech_started", item_id="new")
        await self.until(preparation.done)
        self.assertTrue(preparation.cancelled())
        self.assertFalse(reservation.submitted)
        self.assertEqual(self.creates(case), [])
        await self.commit(case, "new")
        await self.final(case, "new", CURRENT_TEXT)
        case.route.release.set()
        await self.until(lambda: len(self.creates(case)) == 1)
        written = self.creates(case)[0]
        await self.event(case, "response.created", response={
            "id": "resp_selected", "metadata": written["response"]["metadata"]})
        state = case.voice._active_responses["resp_selected"]
        old = next(item for item in state.request_sources.requests if item.kind == "retained_unwritten")
        current = next(item for item in state.request_sources.requests if item.kind == "current")
        for offered in (old, current):
            self.assertEqual(offered.record.owner_id, case.owner)
            self.assertEqual(offered.record.session_id, case.session_id)
            self.assertEqual(offered.record.finality, "final")
            self.assertTrue(offered.record.authorizes_request)
            exact = await self.context.request_evidence(case.owner, case.session_id,
                source_namespace=offered.record.source_namespace,
                source_event_id=offered.record.source_event_id)
            self.assertEqual((exact.id, exact.revision, exact.payload_id),
                             (offered.record.id, offered.record.revision, offered.record.payload_id))
        self.assertNotEqual(old.record.id, current.record.id)
        self.assertEqual(state.metadata["jarvis_turn_generation"], "2")
        self.assertEqual(state.request_sources.generation, 2)
        return state, old, current

    async def cancel_call(self, case, call_id, reference):
        item = {"id": call_id + "-item", "type": "function_call", "call_id": call_id,
            "name": "cancel_background_task", "arguments": json.dumps({
                "task_ref": self.tasks.reference(self.task.id),
                "expected_revision": self.task.revision, SELECTOR: reference})}
        await self.event(case, "response.output_item.done", response_id="resp_selected", item=item)

        def output():
            return next((event["item"] for event in case.socket.events
                if event["type"] == "conversation.item.create"
                and event.get("item", {}).get("type") == "function_call_output"
                and event["item"].get("call_id") == call_id), None)
        await self.until(lambda: output() is not None)
        await self.until(lambda: call_id not in case.voice._active_responses["resp_selected"].pending_tool_call_ids)
        self.assertTrue(await case.capture.flush())
        return json.loads(output()["output"])

    async def assert_current_preserved(self, case, current):
        record = await self.context.get(case.owner, current.record.id, session_id=case.session_id)
        self.assertEqual((record.revision, record.payload_id, record.content["text"]),
                         (current.record.revision, current.record.payload_id, CURRENT_TEXT))
        self.assertTrue(record.authorizes_request)
        self.assertEqual(case.voice._turn_generation, 2)
        self.assertEqual(case.voice._active_responses["resp_selected"].metadata["jarvis_turn_generation"], "2")

    async def assert_no_mutation(self):
        task = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual((task.status, task.revision, task.fence, task.request_event_id, task.request_revision),
            (self.task.status, self.task.revision, self.task.fence, self.task.request_event_id, self.task.request_revision))
        async with self.pool.acquire() as connection:
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_requests where owner_id=$1 and task_id=$2 and operation='cancel'",
                self.owner, self.task.id), 0)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_events where owner_id=$1 and task_id=$2 and event_type='cancel_requested'",
                self.owner, self.task.id), 0)
        self.assertEqual(self.mutations, [])

    async def test_selected_old_source_mutates_once_for_two_provider_call_ids(self):
        state, old, current = await self.offered_pair(self.case)
        first = await self.cancel_call(self.case, "cancel-first", old.reference)
        second = await self.cancel_call(self.case, "cancel-second-new-id", old.reference)
        self.assertEqual((first["status"], second["status"]), ("success", "success"))
        self.assertEqual(len(self.mutations), 2)
        self.assertEqual(self.mutations[0][3].request_id, self.mutations[1][3].request_id)
        request_id = self.mutations[0][3].request_id
        for mutation in self.mutations:
            self.assertEqual(mutation[4], TaskVoiceSource(old.record.id, old.record.revision, self.session))
        task = await self.tasks.get(self.owner, self.task.id)
        self.assertEqual((task.status, task.revision, task.fence),
                         (TaskStatus.CANCELLED, self.task.revision + 1, self.task.fence + 1))
        async with self.pool.acquire() as connection:
            joined = await connection.fetch("""select r.request_id,r.accepted_revision,
                e.id as user_event_id,e.revision as user_revision,e.session_id,e.finality,e.trust,
                e.source_namespace,e.source_event_id,e.payload_id as user_payload,
                te.task_revision,te.payload_id as control_payload,d.source_payload_id
                from jarvis_task_requests r
                join jarvis_background_tasks t on t.owner_id=r.owner_id and t.id=r.task_id
                join jarvis_context_events e on e.owner_id=t.owner_id
                    and e.id=t.request_event_id and e.revision=t.request_revision
                join jarvis_task_events te on te.owner_id=t.owner_id and te.task_id=t.id
                    and te.event_type='cancel_requested' and te.task_revision=r.accepted_revision
                join jarvis_context_payload_dependencies d on d.owner_id=e.owner_id
                    and d.source_payload_id=e.payload_id and d.dependent_payload_id=te.payload_id
                where r.owner_id=$1 and r.task_id=$2 and r.operation='cancel'""", self.owner, self.task.id)
            self.assertEqual(len(joined), 1)
            row = joined[0]
            self.assertEqual((row["request_id"], row["accepted_revision"], row["user_event_id"],
                row["user_revision"], row["session_id"], row["user_payload"], row["source_payload_id"]),
                (request_id, task.revision, old.record.id, old.record.revision, self.session,
                 old.record.payload_id, old.record.payload_id))
            self.assertEqual((row["finality"], row["trust"], row["source_namespace"], row["source_event_id"]),
                ("final", "user_input", old.record.source_namespace, old.record.source_event_id))
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_actions where owner_id=$1 and task_id=$2",
                self.owner, self.task.id), 0)
            self.assertEqual(await connection.fetchval(
                "select count(*) from jarvis_task_attempts where owner_id=$1 and task_id=$2",
                self.owner, self.task.id), 0)
        receipts = await self.executions.get_executions_by_call_ids(owner_id=self.owner,
            session_id=self.session, provider_call_ids=["cancel-first", "cancel-second-new-id"])
        self.assertEqual({item.provider_call_id for item in receipts}, {"cancel-first", "cancel-second-new-id"})
        self.assertTrue(all(item.status == "completed" for item in receipts))
        self.assertEqual(state.tool_followup.request.event_id, old.record.id)
        self.assertEqual(state.tool_followup.conversation_request.event_id, current.record.id)
        await self.assert_current_preserved(self.case, current)

    async def test_foreign_response_reference_and_foreign_owner_source_do_not_mutate(self):
        state, old, current = await self.offered_pair(self.case)
        async with self.pool.acquire() as connection:
            other_session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.other)
        foreign_case = await self.open_case(self.other, other_session)
        foreign_state, foreign_old, _ = await self.offered_pair(foreign_case)
        self.assertNotEqual(state.request_sources.scope, foreign_state.request_sources.scope)
        result = await self.cancel_call(self.case, "foreign-offer", foreign_old.reference)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["code"], "tool_request_source_invalid")
        with self.assertRaises(TaskError) as forbidden:
            await self.tasks.mutate(self.owner, self.task.id, "cancel",
                TaskMutation(request_id=uuid4(), expected_revision=self.task.revision),
                source=TaskVoiceSource(foreign_old.record.id, foreign_old.record.revision, other_session))
        self.assertEqual(forbidden.exception.code, "task_request_evidence_missing")
        await self.assert_no_mutation()
        await self.assert_current_preserved(self.case, current)

    async def test_revised_original_is_rejected_despite_old_capture_cache(self):
        state, old, current = await self.offered_pair(self.case)
        revised = await self.context.append(self.owner, ContextIngress(
            session_id=self.session, source_namespace=old.record.source_namespace,
            source_event_id=old.record.source_event_id, receipt_id="independent-source-correction",
            kind="user_transcript", trust="user_input", finality="revised",
            expected_revision=old.record.revision,
            payload={**old.record.content, "text": "Correction: do not cancel Workbench Plan."},
        ))
        self.assertEqual(revised.revision, old.record.revision + 1)
        self.assertEqual(self.case.capture.cached_request(old.input_id).id, old.record.id)
        result = await self.cancel_call(self.case, "stale-original", old.reference)
        self.assertEqual(result["status"], "failed")
        with self.assertRaises(TaskError):
            await self.case.capture.request_for_call("stale-original")
        await self.assert_no_mutation()
        await self.assert_current_preserved(self.case, current)

    async def test_withdrawn_original_is_rejected_without_losing_current_request(self):
        state, old, current = await self.offered_pair(self.case)
        # Exact fixture-only canonical tombstone, using the normal payload
        # withdrawal trigger; not a fabricated running task or task-state update.
        async with self.tasks.transaction(self.owner) as connection:
            await connection.execute("""update jarvis_context_payloads set availability='deleted',content=null
                where owner_id=$1 and id=$2""", self.owner, old.record.payload_id)
        removed = await self.context.get(self.owner, old.record.id, session_id=self.session)
        self.assertEqual(removed.availability, "deleted")
        self.assertFalse(removed.authorizes_request)
        self.assertEqual(self.case.capture.cached_request(old.input_id).id, old.record.id)
        result = await self.cancel_call(self.case, "withdrawn-original", old.reference)
        self.assertEqual(result["status"], "failed")
        with self.assertRaises(TaskError):
            await self.case.capture.request_for_call("withdrawn-original")
        await self.assert_no_mutation()
        await self.assert_current_preserved(self.case, current)
