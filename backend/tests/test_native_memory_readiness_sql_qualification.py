"""Real manifested SQL joins, with explicit synthetic external transport.

No native/provider/Hindsight acceptance: client/provider events and retention
responses are test inputs. Capture, tool execution, immutable response admission,
publication, ingestion fencing, observer queries and revocation use real services.
"""
import asyncio
import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.experience.repository import ToolExecutionRepository
from app.knowledge.conversation_writes import ConversationMemoryWriteBackend
from app.knowledge.ingestion import IngestionWorker
from app.knowledge.lifecycle import SourceLifecycleService
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.outbox import OutboxRepository
from app.knowledge.publication import CanonicalSourcePublication
from app.memory.tools import MemoryToolModule
from app.migrations.runner import quote_schema
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime import telemetry
from app.runtime.feature_config import TaskLimits
from app.sessions.repository import SessionRepository
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_events import ContextIngress
from app.tasks.context_repository import ContextRepository
from app.tasks.models import TaskVoiceSource
from app.tasks.repository import TaskRepository
from evals.responsive_voice.frontend_host import ReplayHostScope
from evals.responsive_voice.native_memory_readiness import ReadinessRequest
from evals.responsive_voice.native_memory_readiness_observer import (
    MemoryObserverConfig, MemoryReadinessObserver,
)
from evals.responsive_voice.native_memory_replay_control import Scope
from evals.second_brain.source_fixture import source_fixture


class LocalTransport:
    def __init__(self):
        self.writes = []
        self.closed = False

    async def send(self, raw):
        self.writes.append(json.loads(raw))

    async def send_text(self, raw):
        await self.send(raw)

    async def close(self, *args, **kwargs):
        self.closed = True


class FixtureRetentionTransport:
    """Controlled worker responses only; never an actual remote-index receipt."""
    def __init__(self):
        self.operations = {}
        self.documents = {}

    async def ensure_bank(self, bank):
        pass

    async def operation(self, bank, operation):
        return {"status": "completed"} if (bank, operation) in self.operations else None

    async def retain(self, source, operation):
        self.operations[(source.bank_name, operation)] = source.document_id
        self.documents[(source.bank_name, source.document_id)] = source.original_evidence()

    async def document(self, bank, document):
        return self.documents[(bank, document)]

    def verify_document(self, source, document):
        if document != source.original_evidence():
            # Ingestion status changes after the first write; original identity
            # and content are the fixture transport's explicit document proof.
            expected = source.original_evidence()
            expected.pop("ingestion_status")
            actual = dict(document)
            actual.pop("ingestion_status")
            if expected != actual:
                raise AssertionError("fixture_retention_original_changed")


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested SQL observer fixture required")
class MemoryReadinessSQLQualificationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"])
        self.evidence = {"scope": "real SQL; synthetic transport; no native/model/index acceptance",
                         "cases": [], "flows": [], "observer_shutdown": None}
        self.query_window = None
        self.voices = []
        self.observer = None
        self.manager = source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            self.directory, test_name=self.id(), pool_init=self.install_query_logger)
        self.f = await asyncio.wait_for(self.manager.__aenter__(), 360)
        self.addAsyncCleanup(self.finish_fixture)
        bootstrap = await self.f.publish("SQL observer scope bootstrap; not response context")
        self.bank = bootstrap.bank_name
        self.scope = Scope(run_id=self.f.fixture.run_id, owner_id=self.f.owner,
            database_schema=self.f.fixture.manifest["schemas"][0],
            area_id=str(self.f.area), memory_bank_id=self.bank)
        self.host_scope = ReplayHostScope(run_id=self.scope.run_id, owner_id=self.f.owner,
            database_schema=self.scope.database_schema, area_id=self.f.area,
            memory_bank_id=self.bank, expires_at=datetime.now(UTC)+timedelta(minutes=10))
        self.config = MemoryObserverConfig(invocation_id=uuid4(),
            manifest_sha256=hashlib.sha256(b"SQL observer fixture configuration").hexdigest())
        self.requests = TaskRepository(self.f.pool, TaskLimits())
        self.history = ContextRepository(self.requests,
            publication=CanonicalSourcePublication(self.f.sources, self.f.area))
        self.sessions = SessionRepository(self.f.pool)
        self.writer = ConversationMemoryWriteBackend(self.f.sources, self.requests,
            owner_id=self.f.owner, area_id=self.f.area)
        self.lifecycle = SourceLifecycleService(self.f.sources, self.requests)
        self.retention = FixtureRetentionTransport()
        self.worker = IngestionWorker(OutboxRepository(self.f.sources), self.retention)
        self.old_level = telemetry.LOGGER.level
        telemetry.LOGGER.setLevel(logging.INFO)
        self.observer = MemoryReadinessObserver(self.host_scope, self.config)
        self.observer.start(SimpleNamespace(db_pool=self.f.pool, _closed=False,
            knowledge_runtime=SimpleNamespace(sources=self.f.sources, owner_id=self.f.owner,
                context_repository=self.history,
                configuration=SimpleNamespace(area_id=self.f.area), _closed=False)),
            started_at=datetime.now(UTC))

    async def install_query_logger(self, connection):
        def log(record):
            if self.query_window is not None:
                sql = record.query.strip()
                self.query_window.append({"verb": sql.split(None, 1)[0].upper().rstrip(";"),
                    "read_only": "READ ONLY" in sql.upper(),
                    "fixture_pool_setup": sql == (
                        f"set search_path to {quote_schema(self.scope.database_schema)},pg_catalog,public,extensions"),
                    "pool_reset": "pg_advisory_unlock_all()" in sql and "RESET ALL" in sql,
                    "query_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                    "elapsed_seconds": record.elapsed,
                    "exception_type": type(record.exception).__name__ if record.exception else None})
        connection.add_query_logger(log)

    async def finish_fixture(self):
        errors = []
        try:
            for voice in self.voices:
                try:
                    await asyncio.wait_for(voice.close("sql_fixture_complete"), 10)
                except BaseException as error:
                    errors.append(type(error).__name__)
            if self.observer is not None:
                try:
                    await self.observer.shutdown()
                    self.evidence["observer_shutdown"] = self.observer.status()
                except BaseException as error:
                    errors.append(type(error).__name__)
            if hasattr(self, "old_level"):
                telemetry.LOGGER.setLevel(self.old_level)
        finally:
            try:
                await asyncio.wait_for(self.manager.__aexit__(None, None, None), 180)
            except BaseException as error:
                errors.append(type(error).__name__)
            self.evidence["fixture_manifest"] = str(self.f.fixture.path)
            self.evidence["fixture"] = self.f.fixture.manifest
            self.evidence["cleanup_errors"] = errors
            self.evidence["external_resources"] = {
                "remote_hindsight_banks_created": 0, "storage_uploads": 0,
                "provider_sessions_created": 0, "native_launches": 0}
            self.evidence["transport_handles_closed"] = all(
                v.openai._websocket is None or v.openai._websocket.closed for v in self.voices)
            (self.directory/"SQL_EVIDENCE.json").write_text(
                json.dumps(self.evidence, indent=2, default=str)+"\n")
        self.assertEqual(errors, [])
        self.assertEqual(self.f.fixture.manifest["status"], "cleaned")

    async def response(self, voice, transport, *, phase, label):
        reservation = voice._reserve_response_locked("normal_user_turn", {})
        self.assertTrue(await voice._send_reserved_response(reservation))
        metadata = transport.writes[-1]["response"]["metadata"]
        response_id, item_id = "response-"+uuid4().hex, "item-"+uuid4().hex
        await voice._handle_openai_event({"type":"response.created", "response":{
            "id":response_id, "metadata":metadata}})
        await voice._handle_openai_event({"type":"response.output_item.added",
            "response_id":response_id, "output_index":0,
            "item":{"id":item_id, "type":"message", "role":"assistant", "phase":phase}})
        await voice._handle_openai_event({"type":"response.output_audio_transcript.done",
            "response_id":response_id, "item_id":item_id, "transcript":label})
        return response_id, item_id, reservation.request_id

    async def finish_response(self, voice, identity):
        await voice._handle_openai_event({"type":"response.done", "response":{
            "id":identity[0], "status":"completed", "output":[]}})
        self.assertTrue(await voice._context_capture.flush())

    async def flow(self):
        session = await self.sessions.create_session(self.f.owner, "Synthetic SQL observer")
        audio = str(uuid4())
        capture = SessionContextCapture(self.history, owner_id=self.f.owner,
            session_id=session.id, provider="openai", audio_session_id=audio)
        reader = SourceMemoryReadBackend(self.f.sources, SimpleNamespace(sources=self.f.sources),
            owner_id=self.f.owner, area_id=self.f.area)
        module = MemoryToolModule(read_backend=reader, write_backend=self.writer)
        provider, app = LocalTransport(), LocalTransport()
        voice = RealtimeAudioSession(websocket=app,
            context=RealtimeSessionContext(audio_session_id=audio, owner_id=self.f.owner,
                jarvis_session_id=session.id, client_kind="phone_audio", session=session),
            settings=JarvisSettings(), repository=self.sessions, tool_modules=(module,),
            context_providers=(), context_capture=capture,
            tool_execution_repository=ToolExecutionRepository(self.f.pool))
        voice.openai._websocket = provider
        self.voices.append(voice)
        segment, client_event = str(uuid4()), str(uuid4())
        await voice._handle_client_event({"type":"audio.input.append",
            "audio_base64":base64.b64encode(b"\x01\x00"*2400).decode()})
        await voice._handle_client_event({"type":"audio.input.commit",
            "segment_id":segment, "client_event_id":client_event})
        commit = next(p["event_id"] for p in provider.writes if p["type"]=="input_audio_buffer.commit")
        provider_input = "input-"+uuid4().hex
        await voice._handle_openai_event({"type":"input_audio_buffer.committed", "item_id":provider_input})
        await voice._handle_openai_event({"type":"conversation.item.input_audio_transcription.completed",
            "item_id":provider_input, "transcript":"Remember this synthetic SQL qualification fact."})
        self.assertTrue(await capture.flush())
        original = await self.history.request_evidence(self.f.owner, session.id,
            source_namespace=capture.namespace, source_event_id=capture._source("input", "commit:"+commit))
        voice._source_context.observe((original.payload_id,))
        commentary = await self.response(voice, provider, phase="commentary", label="Synthetic commentary.")
        call = "call-"+uuid4().hex
        capture.bind_openai_call(call, commentary[0])
        saved = await voice.tool_executor.execute("remember", {"memory_type":"fact",
            "content":"Synthetic SQL qualification fact, not a calendar oracle."}, provider_call_id=call)
        self.assertEqual(saved.get("status"), "success", saved.get("error"))
        self.assertTrue(await voice._write_tool_output(call, saved))
        await self.finish_response(voice, commentary)
        final = await self.response(voice, provider, phase="final_answer", label="Synthetic final acknowledgment.")
        await self.finish_response(voice, final)
        source = await self.f.sources.read(self.f.owner, self.f.area, UUID(saved["memory_id"]))
        flow = SimpleNamespace(voice=voice, capture=capture, session=session.id, audio=audio,
            segment=segment, client_event=client_event, commit=commit, provider_input=provider_input,
            commentary=commentary, final=final, source=source, call=call, original=original,
            connection=str(uuid4()))
        self.evidence["flows"].append({"session_id":str(session.id), "audio_session_id":audio,
            "segment_id":segment,"client_event_id":client_event,"actual_commit_request_id":commit,
            "request_event_id":str(original.id),"request_revision":original.revision,
            "tool_event_id":str(saved.event_id),"tool_payload_id":str(saved.payload_id),
            "provider_call_id":call,"source_id":str(source.source_id),
            "document_id":source.document_id,"final_response_id":final[0],"final_item_id":final[1],
            "response_source_snapshot_id":str(voice._source_context.source_for(final[2]))})
        return flow

    async def observe(self, flow, label, **changes):
        now = datetime.now(UTC)
        request = ReadinessRequest(scope=self.scope, invocation_id=str(self.config.invocation_id),
            manifest_sha256=self.config.manifest_sha256, session_id=flow.session,
            audio_session_id=flow.audio, connection_id=flow.connection,
            prior_input_ordinal=0, target_input_ordinal=1, segment_id=flow.segment,
            client_event_id=flow.client_event, response_id=flow.final[0], item_id=flow.final[1],
            nonce=str(uuid4()), issued_at=now, expires_at=now+timedelta(seconds=30))
        request = replace(request, **changes)
        await asyncio.sleep(0)
        self.query_window = []
        started = time.monotonic()
        try:
            result = await self.observer.observe(request)
            await asyncio.sleep(0)
            queries = self.query_window
        finally:
            self.query_window = None
        self.evidence["cases"].append({"case":label,"receipt":result,
            "elapsed_seconds":time.monotonic()-started,"queries":queries})
        self.assertTrue(all(q["verb"] in {"SELECT","WITH","BEGIN","START","COMMIT","ROLLBACK"}
                            or q["verb"] == "SET" and q["fixture_pool_setup"]
                            for q in queries), queries)
        if any(q["verb"] in {"SELECT","WITH"} for q in queries):
            self.assertTrue(any(q["read_only"] for q in queries), queries)
        self.assertLessEqual(sum(q["verb"] in {"SELECT","WITH"} and not q["pool_reset"]
                                 for q in queries), 9)
        self.assertNotIn("Synthetic SQL qualification fact", json.dumps(result))
        return result

    async def test_real_sql_exact_observer_boundaries(self):
        async with asyncio.timeout(180):
            flow = await self.flow()
            with self.subTest(boundary="retention_pending"):
                self.assertEqual((await self.observe(flow,"retention_pending"))["status"], "waiting")
            for label, change in (
                ("wrong_commit_client_event", {"client_event_id":str(uuid4())}),
                ("wrong_segment", {"segment_id":str(uuid4())}),
                ("wrong_item", {"item_id":"item-unrelated"}),
                ("wrong_response", {"response_id":"response-unrelated"}),
                ("wrong_audio_scope", {"audio_session_id":str(uuid4())}),
                ("wrong_owner_scope", {"scope":replace(self.scope,owner_id=self.f.other)}),
                ("wrong_bank_scope", {"scope":replace(self.scope,memory_bank_id="jarvis_"+uuid4().hex)}),
                ("commentary_not_final", {"response_id":flow.commentary[0],"item_id":flow.commentary[1],
                                          "canonical_requirement":"finalized_input"}),
            ):
                with self.subTest(boundary=label):
                    self.assertNotEqual((await self.observe(flow,label,**change))["status"], "ready")
            with self.subTest(boundary="finalized_input_final_policy"):
                result = await self.observe(flow,"finalized_input_final_policy",canonical_requirement="finalized_input")
                self.assertEqual(result["status"],"ready",result)
                self.assertEqual(result["final_output_policy"]["realtime_output"]["phase"],"final_answer")
            accepted_seen = False
            for iteration in range(24):
                await self.f.ready_jobs()
                await self.worker.step(self.f.owner,"sql-observer-fixture-worker")
                source = await self.f.sources.read(self.f.owner,self.f.area,flow.source.source_id)
                if source.ingestion_status == "accepted" and not accepted_seen:
                    accepted_seen = True
                    with self.subTest(boundary="retention_accepted"):
                        self.assertEqual((await self.observe(flow,"retention_accepted"))["status"],"waiting")
                if source.ingestion_status == "searchable":
                    break
            self.evidence["retention_worker_steps"] = iteration+1
            self.assertTrue(accepted_seen)
            self.assertEqual(source.ingestion_status,"searchable")
            with self.subTest(boundary="exact_searchable_final_continuation"):
                ready = await self.observe(flow,"exact_searchable_final_continuation")
                self.assertEqual(ready["status"],"ready",ready)
                self.assertEqual(ready["retained_source"]["source_id"],str(flow.source.source_id))
                self.assertFalse(ready["full_chain_proven"])
            flow.capture.openai_transcript(flow.provider_input,"Corrected synthetic qualification request.")
            self.assertTrue(await flow.capture.flush())
            with self.subTest(boundary="request_correction_revokes_old_response"):
                self.assertNotEqual((await self.observe(flow,"request_correction_revokes_old_response"))["status"],"ready")
            deletion = await self.flow()
            with self.subTest(boundary="pre_deletion_finalized_input_ready"):
                self.assertEqual((await self.observe(deletion,"pre_deletion_finalized_input_ready",
                    canonical_requirement="finalized_input"))["status"],"ready")
            request = await self.history.append(self.f.owner,ContextIngress(session_id=deletion.session,
                source_namespace="sql-fixture-forget",source_event_id=str(uuid4()),receipt_id=str(uuid4()),
                kind="user_transcript",trust="user_input",payload={"text":"Forget this synthetic saved fact."}))
            await self.lifecycle.forget(self.f.owner,self.f.area,deletion.source.source_id,
                expected_revision=deletion.source.revision,
                request=TaskVoiceSource(request.id,request.revision,deletion.session))
            with self.subTest(boundary="memory_deletion_revokes_output_dependency"):
                self.assertNotEqual((await self.observe(deletion,"memory_deletion_revokes_output_dependency",
                    canonical_requirement="finalized_input"))["status"],"ready")
