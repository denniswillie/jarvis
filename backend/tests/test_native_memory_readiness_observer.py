"""Offline real-capture/tool/source-snapshot composition with controlled storage.

No database commit, worker/provider operation, native playback or frontend trigger
is exercised. The transport fixture implements exact repository query projections;
the normal remember/capture/response-registration code is not a new state machine.
"""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.conversation_writes import ConversationMemoryWriteBackend
from app.knowledge.memory_backend import SourceMemoryReadBackend
from app.knowledge.models import SourceVersion
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.realtime.telemetry import LOGGER, RealtimeTrace
from app.settings import JarvisSettings
from app.tasks.context_capture import SessionContextCapture
from app.tasks.context_repository import ContextRepository, RECORD_SQL
from app.tools.invocation import bind_tool_invocation
from app.tools.provenance import ToolResultCapture
from evals.responsive_voice.frontend_host import FrontendReplayHost, ReplayHostScope
from evals.responsive_voice.native_memory_readiness import ReadinessRequest
from evals.responsive_voice.native_memory_readiness_observer import MemoryObserverConfig, MemoryReadinessObserver
from test_context_response_association import CanonicalMemoryStore


class Store(CanonicalMemoryStore):
    def __init__(self):
        super().__init__()
        self.parents = {}

    async def append(self, owner, event):
        row = await super().append(owner, event)
        row = row.model_copy(update={"content_digest": "b" * 64})
        self.rows[(owner, event.source_namespace, event.source_event_id)] = row
        self.parents[row.payload_id] = event.source_payload_ids
        return row

    async def append_in_transaction(self, connection, owner, event, **kwargs):
        return await self.append(owner, event)


class Transport:
    """Controlled SQL transport, NOT SQL integration or PostgreSQL semantics."""
    def __init__(self, case):
        self.case = case
        self.queries = []
        self.transactions = []
        self.source_payloads = {}
        self.dependencies = []
        self.pause = None
        self.entered = asyncio.Event()

    @asynccontextmanager
    async def acquire(self):
        yield self

    @asynccontextmanager
    async def transaction(self, **options):
        self.transactions.append(options)
        yield self

    async def fetchval(self, sql, *args):
        if "lock_jarvis_copy_sources" in sql:
            return True
        if "insert into jarvis_context_payloads" in sql:
            identity = uuid4()
            self.source_payloads[identity] = json.loads(args[1])
            return identity
        if "jarvis_context_ingress_receipts" in sql:
            return None
        raise AssertionError(sql)

    async def executemany(self, sql, values):
        assert "insert into jarvis_context_payload_dependencies" in sql
        self.dependencies.extend(values)

    async def fetchrow(self, sql, *args):
        assert "jarvis_knowledge_context_sources" in sql
        return {"id": self.case.source_id, "area_id": self.case.host_scope.area_id}

    async def fetch(self, sql, *args):
        c = self.case
        self.queries.append((sql, args))
        if self.pause is not None:
            self.entered.set()
            await self.pause.wait()
        if "select s.id,s.area_id" in sql:
            return [{"id": c.parent_source_id, "area_id": c.host_scope.area_id}]
        assert args[0] == c.host_scope.owner_id
        if "select current_schema()" in sql:
            assert args[1:3] == (c.request.session_id, c.host_scope.area_id)
            return [{"schema": c.host_scope.database_schema, "bank_name": c.host_scope.memory_bank_id}]
        if sql.startswith(RECORD_SQL):
            records = list(c.store.rows.values())
            if "d.dependent_payload_id=$5" in sql:
                records = [r for r in records if r.kind == "tool_outcome"
                    and (r.content or {}).get("tool_name") == "remember"
                    and (r.content or {}).get("input_id") == args[3]
                    and (args[0], r.payload_id, args[4]) in self.dependencies]
                records = records * c.duplicate_tools
            elif "e.id=$2" in sql:
                records = [r for r in records if r.id == args[1]]
            else:
                records = [r for r in records if r.session_id == args[1]
                    and r.source_namespace == args[2] and r.source_event_id == args[3]]
            return [r.model_dump() for r in records][:2]
        if "select p.id,p.availability" in sql:
            if args[1] not in c.store.parents.get(args[2], ()):
                return []
            return [{"id": args[1], "availability": "available", "content_digest": "e" * 64,
                     "content": self.source_payloads[args[1]]}]
        if "from jarvis_tool_executions" in sql:
            return [{"status": c.execution_status, "result": c.bound.audit_receipt()}]
        if "from jarvis_knowledge_source_versions" in sql:
            assert args[2:] == (c.source.source_id, c.source.revision)
            return [c.source_row()]
        if "from jarvis_memory_outbox" in sql:
            assert args[1:] == (c.source.document_id, c.source.generation)
            return [dict(c.retain)] * c.duplicate_retains
        raise AssertionError(sql)


class MemoryObserverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.old_level = LOGGER.level
        LOGGER.setLevel(logging.INFO)
        self.addCleanup(LOGGER.setLevel, self.old_level)
        self.now = datetime.now(UTC)
        run, invocation = uuid4().hex, uuid4()
        self.host_scope = ReplayHostScope(run_id=run,
            owner_id=f"direct-image-eval-second-brain-{run}-a", database_schema=f"jarvis_sb_eval_{run}_0",
            area_id=uuid4(), memory_bank_id="jarvis_" + uuid4().hex,
            expires_at=self.now + timedelta(seconds=60))
        self.config = MemoryObserverConfig(invocation, "a" * 64)
        self.observer = MemoryReadinessObserver(self.host_scope, self.config, clock=lambda: self.now)
        self.addAsyncCleanup(self.observer.shutdown)
        self.request = ReadinessRequest(self.observer.scope, str(invocation), "a" * 64, uuid4(),
            str(uuid4()), str(uuid4()), 0, 1, str(uuid4()), str(uuid4()),
            "response-after-tool", "final-item", uuid4().hex,
            self.now - timedelta(seconds=1), self.now + timedelta(seconds=10))
        self.store = Store()
        self.pool = Transport(self)
        repository = ContextRepository(SimpleNamespace(pool=self.pool))
        runtime = SimpleNamespace(db_pool=self.pool, knowledge_runtime=SimpleNamespace(
            owner_id=self.host_scope.owner_id, configuration=SimpleNamespace(area_id=self.host_scope.area_id),
            sources=SimpleNamespace(pool=self.pool), context_repository=repository))
        self.observer.start(runtime, started_at=self.now-timedelta(seconds=5))
        self.trace = RealtimeTrace(provider="openai", audio_session_id=self.request.audio_session_id,
                                   jarvis_session_id=self.request.session_id)
        self.capture = SessionContextCapture(self.store, owner_id=self.host_scope.owner_id,
            session_id=self.request.session_id, provider="openai", audio_session_id=self.request.audio_session_id)
        self.addAsyncCleanup(self.capture.close)
        self.commit = uuid4().hex  # Opaque backend ID: never normalize this to hyphenated UUID.
        self.capture.admit_openai_commit(self.commit, 7)
        self.trace.event("input_commit_admitted", request_id=self.commit, turn_generation=7,
            segment_id=self.request.segment_id.upper(), client_event_id=self.request.client_event_id.upper())
        self.capture.openai_committed("provider-input")
        self.capture.openai_transcript("provider-input", "Remember my dated calendar correction.")
        await self.capture.flush()
        self.parent = next(r for r in self.store.rows.values() if r.kind == "user_transcript")
        self.source_id, self.parent_source_id = uuid4(), uuid4()
        self.duplicate_tools = self.duplicate_retains = 1
        self.execution_status = "completed"

        async def read_parent(*args):
            return SimpleNamespace(context_event_id=self.parent.id, context_event_revision=self.parent.revision,
                context_session_id=self.request.session_id,
                original_evidence=lambda: {"payload_id": str(self.parent.payload_id)})

        async def read_saved(*args):
            memory = next(r for r in self.store.rows.values() if r.source_namespace == "conversation-explicit-memory")
            self.source = SourceVersion(owner_id=self.host_scope.owner_id, area_id=self.host_scope.area_id,
                source_id=self.source_id, revision=1, generation=1, payload_id=memory.payload_id,
                document_id="calendar-original-1", bank_name=self.host_scope.memory_bank_id,
                content_digest=memory.content_digest, source_type="tool_outcome", trust="tool_receipt",
                source_time=self.now, time_basis="canonical source_time", recorded_at=self.now,
                content=memory.content, is_current=True, ingestion_status="pending",
                session_id=self.request.session_id, context_session_id=self.request.session_id,
                context_event_id=memory.id, context_event_revision=memory.revision, context_recorded_at=self.now)
            return self.source

        sources = SimpleNamespace(pool=self.pool, read=read_parent, _read=read_saved, _area=AsyncMock())
        requests = SimpleNamespace(pool=self.pool, lock_owner=AsyncMock(), _voice_source=AsyncMock())
        writer = ConversationMemoryWriteBackend(sources, requests, owner_id=self.host_scope.owner_id,
                                                area_id=self.host_scope.area_id)
        # Only the storage/provenance transport is controlled; the normal memory
        # formulation, source identities, result capture and response binding run.
        writer.publisher.append_in_transaction = self.store.append_in_transaction
        async def bind(provenance, owner, area, metadata, materialize):
            originals = tuple(UUID(item["payload_id"]) for item in metadata)
            value, _ = await materialize(self.pool, originals)
            return value
        with patch("app.knowledge.read_provenance.SourceReadProvenance.bind", bind):
            with bind_tool_invocation(provider_call_id="calendar-call", request_event_id=self.parent.id,
                                      request_revision=self.parent.revision):
                saved = await writer.remember(owner_id=self.host_scope.owner_id, session_id=self.request.session_id,
                    parameters={"memory_type": "fact", "content": "PRIVATE correction, not a fixture answer",
                                "occurred_at": "2026-08-01T10:00:00+00:00"})
            self.capture.bind_call("calendar-call", "commit:" + self.commit)
            reader = SourceMemoryReadBackend(sources, SimpleNamespace(sources=sources),
                owner_id=self.host_scope.owner_id, area_id=self.host_scope.area_id)
            self.store.pool = self.pool
            self.bound = await reader.capture_tool_result({"status": "success", **saved},
                ToolResultCapture(self.capture, "calendar-call", "remember"))
        self.source = self.source.model_copy(update={"ingestion_status": "searchable"})
        self.retain = {"id": uuid4(), "owner_id": self.host_scope.owner_id, "document_id": self.source.document_id,
            "generation": 1, "operation": "retain", "status": "searchable", "dispatch_count": 1}
        self.voice = RealtimeAudioSession(websocket=SimpleNamespace(),
            context=RealtimeSessionContext(audio_session_id=self.request.audio_session_id,
                owner_id=self.host_scope.owner_id, jarvis_session_id=self.request.session_id,
                client_kind="phone_audio", session=SimpleNamespace(id=self.request.session_id)),
            settings=JarvisSettings(), repository=SimpleNamespace(_pool=self.pool),
            tool_modules=(), context_providers=(), context_capture=self.capture)
        self.voice._turn_generation = 7
        self.voice._source_context.observe((self.bound.payload_id,))
        self.response_request = uuid4().hex
        self.snapshot = await self.voice._source_context.prepare(self.response_request)
        self.trace.event("response_write_gate", request_id=self.response_request, turn_generation=7,
                         boundary="payload_at_write", status="accepted")
        self.voice._latest_response_request_id = self.response_request
        from app.realtime.models import PendingRealtimeResponse
        self.voice._unacknowledged_response_requests[self.response_request] = PendingRealtimeResponse(
            request_id=self.response_request, turn_generation=7, metadata={},
            kind="normal_user_turn", submitted=True)
        self.trace.event("provider_response_started", request_id=self.response_request, response_id=self.request.response_id)
        await self.voice._register_response({"id": self.request.response_id, "metadata": {
            "jarvis_request_id": self.response_request, "jarvis_turn_generation": "7"}})
        self.capture.observe_output({"type": "transcript.done", "response_id": self.request.response_id,
            "item_id": self.request.item_id, "text": "PRIVATE acknowledgement"}, forwarded=True)
        self.capture.finish_response(self.request.response_id, interrupted=False, completed=True)
        await self.capture.flush()
        self.policy = {'response_kind':'normal_user_turn', 'audio_session_id':self.request.audio_session_id,
            'realtime_output':{'item_id':self.request.item_id, 'phase':'final_answer',
                'phase_anomaly':False, 'delivery_disposition':'forwarded'}}
        self.assertTrue(self.capture.assistant_message(response_id=self.request.response_id,
            item_id=self.request.item_id, text='PRIVATE acknowledgement', metadata=self.policy))
        await self.capture.flush()
        self.pool.queries.clear()

    def source_row(self):
        return self.source.model_dump() | {"current_revision": self.source.revision,
            "current_generation": self.source.generation, "context_is_current": True,
            "source_status": "active", "availability": "available", "area_status": "active",
            "source_namespace": "canonical-context", "source_key": "fixture",
            "context_source_namespace": "conversation-explicit-memory", "context_kind": "tool_outcome",
            "context_trust": "tool_receipt", "context_binding_matches": True}

    async def observe(self, **changes):
        return await self.observer.observe(replace(self.request, nonce=uuid4().hex, **changes))

    async def test_real_capture_remember_bound_result_and_registered_response_ready(self):
        result = await self.observe()
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["observer"]["queries"], 9)
        self.assertEqual(self.pool.transactions[-1], {"readonly": True, "isolation": "repeatable_read"})
        self.assertEqual(self.voice._active_responses[self.request.response_id].source_snapshot, self.snapshot)
        self.assertEqual(result["inline_evidence"]["continuation"]["tool_payload_id"], str(self.bound.payload_id))
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertFalse(result["full_chain_proven"])
        self.assertEqual(result["model_use_of_tool_result"], "not_established")
        self.assertEqual(result['final_output_policy'], self.policy)

    async def test_missing_commentary_suppressed_anomalous_or_wrong_policy_never_ready(self):
        key = next(k for k,r in self.store.rows.items() if r.kind=='assistant_output')
        original = self.store.rows[key]
        cases = [None, {**self.policy,'response_kind':'prospective'},
                 {**self.policy,'audio_session_id':'other'}]
        for change in ({'phase':'commentary'}, {'phase_anomaly':True},
                       {'item_id':'other'}, {'delivery_disposition':'suppressed'}):
            cases.append({**self.policy,'realtime_output':{**self.policy['realtime_output'],**change}})
        for policy in cases:
            with self.subTest(policy=policy):
                self.store.rows[key] = original.model_copy(update={'content':{
                    **original.content,'conversation_message':policy}})
                result = await self.observe()
                self.assertNotEqual(result['status'],'ready')
                self.assertNotIn('final_output_policy',result)
        self.store.rows[key] = original

    async def test_later_global_context_does_not_replace_admitted_snapshot(self):
        self.voice._source_context.observe((uuid4(),))
        await self.voice._source_context.prepare("unrelated-later-response")
        result = await self.observe()
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["inline_evidence"]["continuation"]["snapshot_payload_id"], str(self.snapshot.payload_id))

    async def test_missing_output_snapshot_and_tool_edges_never_release(self):
        parents = dict(self.store.parents)
        self.store.parents.clear()
        self.assertEqual((await self.observe())["reason"], "exact_output_snapshot_edge_unavailable")
        self.store.parents.update(parents)
        self.pool.dependencies.clear()
        self.assertEqual((await self.observe())["reason"], "remember_snapshot_edge_missing_or_ambiguous")

    async def test_duplicate_tool_or_retain_and_unfinished_execution_fail_closed(self):
        self.duplicate_tools = 2
        self.assertNotEqual((await self.observe())["status"], "ready")
        self.duplicate_tools, self.duplicate_retains = 1, 2
        self.assertNotEqual((await self.observe())["status"], "ready")
        self.duplicate_retains, self.execution_status = 1, "started"
        self.assertEqual((await self.observe())["reason"], "completed_source_bound_execution_unavailable")

    async def test_nonsearchable_terminal_and_unknown_retains_do_not_release(self):
        for status in ("pending", "accepted", "leased", "outcome_unknown", "cancelled", "failed", "done"):
            with self.subTest(status=status):
                self.retain["status"] = status
                self.assertNotEqual((await self.observe())["status"], "ready")

    async def test_missing_wrong_commit_and_response_identity_do_not_query(self):
        for change in ({"client_event_id": str(uuid4())}, {"segment_id": str(uuid4())},
                       {"response_id": "unrelated"}, {"invocation_id": str(uuid4())},
                       {"manifest_sha256": "f" * 64}, {"session_id": uuid4()}):
            with self.subTest(change=change):
                self.assertNotEqual((await self.observe(**change))["status"], "ready")
        self.assertEqual(self.pool.queries, [])

    async def test_stale_request_revision_rejected(self):
        self.capture.openai_transcript("provider-input", "Corrected request with different meaning")
        await self.capture.flush()
        self.assertNotEqual((await self.observe())["status"], "ready")

    async def test_nonce_budget_expiry_and_closed_audio_session(self):
        result = await self.observer.observe(self.request)
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual((await self.observer.observe(self.request))["reason"], "consume_reused_busy_or_budget_exhausted")
        self.assertEqual((await self.observe(expires_at=self.now))["status"], "expired")
        self.trace.event("session_closed_forwarded")
        self.assertEqual((await self.observe())["reason"], "audio_session_closed")

    async def test_trace_truncation_and_conflicting_commit_fail_closed(self):
        self.trace.event("input_commit_admitted", request_id="other", turn_generation=7,
            segment_id=self.request.segment_id, client_event_id=self.request.client_event_id)
        self.assertEqual((await self.observe())["reason"], "conflicting_observed_binding")
        self.observer.issue = None
        self.observer.records = [{}] * self.config.max_trace_records
        self.trace.event("provider_response_started", request_id="extra", response_id="extra")
        self.assertEqual((await self.observe())["reason"], "trace_record_limit")

    async def test_timeout_and_shutdown_join_without_retries(self):
        self.pool.pause = asyncio.Event()
        self.observer.config = replace(self.config, read_timeout_seconds=.01)
        result = await self.observe()
        self.assertEqual(result["reason"], "canonical_read_timeout")
        self.assertEqual(len(self.pool.queries), 1)
        task = asyncio.create_task(self.observe())
        await self.pool.entered.wait()
        await asyncio.sleep(0)
        receipt = await self.observer.shutdown()
        self.assertTrue(receipt["joined"])
        self.assertTrue(task.done())
        self.assertEqual(self.observer.inflight, set())
        self.assertNotIn(self.observer, LOGGER.handlers)

    async def test_finalized_input_mode_still_requires_exact_admitted_snapshot(self):
        result = await self.observe(canonical_requirement="finalized_input")
        self.assertEqual(result["status"], "ready", result)
        self.assertEqual(result["observer"]["queries"], 4)
        self.assertEqual(result["playback"], "not_evaluated")

    def test_bounds_are_finite_and_explicit(self):
        for kwargs in ({"max_consumes": 0}, {"max_trace_records": 2049},
                       {"read_timeout_seconds": float("nan")}, {"read_timeout_seconds": 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)


class HostOptInTests(unittest.TestCase):
    def test_default_does_not_create_observer_and_explicit_opt_in_only_attaches(self):
        from starlette.applications import Starlette
        run = uuid4().hex
        scope = ReplayHostScope(run_id=run, owner_id=f"direct-image-eval-second-brain-{run}-a",
            database_schema=f"jarvis_sb_eval_{run}_0", area_id=uuid4(),
            memory_bank_id="jarvis_" + uuid4().hex, expires_at=datetime.now(UTC)+timedelta(seconds=30))
        with patch("evals.responsive_voice.frontend_host.validate_configuration"):
            ordinary = FrontendReplayHost(Starlette(), scope, settings=None, flags=None)
            opted = FrontendReplayHost(Starlette(), scope, settings=None, flags=None,
                native_memory_readiness=MemoryObserverConfig(uuid4(), "a" * 64))
        self.assertIsNone(ordinary.memory_readiness)
        self.assertFalse(opted.memory_readiness.active)
        self.assertNotIn(opted.memory_readiness, LOGGER.handlers)


class HostShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_optional_control_joins_even_if_other_close_raises(self):
        from starlette.applications import Starlette
        for failing in ('memory','task'):
            with self.subTest(failing=failing):
                run = uuid4().hex
                scope = ReplayHostScope(run_id=run, owner_id=f'direct-image-eval-second-brain-{run}-a',
                    database_schema=f'jarvis_sb_eval_{run}_0', area_id=uuid4(),
                    memory_bank_id='jarvis_'+uuid4().hex, expires_at=datetime.now(UTC)+timedelta(seconds=30))
                app = Starlette()
                with patch('evals.responsive_voice.frontend_host.validate_configuration'):
                    host = FrontendReplayHost(app,scope,settings=None,flags=None)
                host.activate = AsyncMock()
                app.state.runtime_services = object()
                memory = AsyncMock(side_effect=RuntimeError('memory') if failing=='memory' else None)
                task = AsyncMock(side_effect=RuntimeError('task') if failing=='task' else None,
                    return_value={'observer_joined':True,'release_loop_joined':True})
                host.memory_readiness = SimpleNamespace(shutdown=memory)
                host.task_replay = SimpleNamespace(close=task)
                with patch('evals.responsive_voice.frontend_host.validate_configuration'), self.assertRaises(RuntimeError):
                    async with app.router.lifespan_context(app):
                        pass
                memory.assert_awaited_once()
                task.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
