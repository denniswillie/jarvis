"""Offline real admission/policy composition; fake storage is not SQL evidence."""
import asyncio
from contextlib import asynccontextmanager
import json
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.admission_diagnostics import (
    AdmissionDiagnostics, MAX_EPISODES, observe_admission, record_admission, shutdown_snapshot,
)
from app.attention.native_voice import NativeAttentionSink
from app.attention.quiet import OwnerQuietRepository, QuietCommand
from app.attention.repository import AttentionRepository
from app.attention.service import AttentionService
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.session import RealtimeAudioSession
from app.realtime.session_manager import RealtimeSessionManager
from app.tools.invocation import ToolInvocationContext
from test_attention_quiet import LedgerFixture


OWNER = "offline-admission-diagnostic"


class ClaimLedger(LedgerFixture):
    def __init__(self):
        super().__init__()
        self.summary = None
        self.candidate = None
        self.attempts = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        if "select id,intent_id from jarvis_attention_attempts" in sql:
            return []
        if "select id from jarvis_attention_intents" in sql:
            if "and not_before<=clock_timestamp()" in sql and self.candidate:
                return [{"id": self.candidate["id"]}]
            return []
        raise AssertionError("Unexpected fixture query")

    async def fetchval(self, sql, *args):
        if "select exists(select 1 from jarvis_attention_intents" in sql:
            return True
        if "select id from jarvis_attention_intents" in sql:
            return self.summary
        if "insert into jarvis_attention_intents" in sql:
            self.summary = uuid4()
            self.calls.append((sql, args))
            return self.summary
        if "insert into jarvis_attention_attempts" in sql:
            identity = uuid4()
            self.attempts.append(identity)
            return identity
        return await super().fetchval(sql, *args)

    async def fetchrow(self, sql, *args):
        if "from jarvis_attention_intents" in sql:
            return self.candidate
        if "select id from jarvis_context_payloads" in sql:
            return {"id": self.candidate["source_payload_id"]}
        return await super().fetchrow(sql, *args)

    async def execute(self, sql, *args):
        if "update jarvis_attention_intents" in sql or "update jarvis_attention_attempts" in sql:
            self.calls.append((sql, args))
            return "UPDATE 0"
        return await super().execute(sql, *args)


class TaskStore:
    def __init__(self, db):
        self.pool = db
        self.exit_error = None

    @asynccontextmanager
    async def transaction(self, owner):
        async with self.pool.lock:
            yield self.pool
            if self.exit_error:
                raise self.exit_error

    async def payload(self, connection, owner, content, **kwargs):
        identity = uuid4()
        connection.payloads[identity] = {"content": content, "availability": "available"}
        return identity


class AdmissionDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = ClaimLedger()
        self.tasks = TaskStore(self.db)
        self.repo = AttentionRepository(self.tasks)
        self.quiet = OwnerQuietRepository(self.db)
        self.session_id = uuid4()
        self.diagnostics = AdmissionDiagnostics()

    async def command(self, action):
        source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.db.authorize(OWNER, self.session_id, source)
        revision = (await self.quiet.get(OWNER)).revision
        return await self.quiet.save(OWNER, self.session_id, source,
                                    QuietCommand(action=action, expected_revision=revision))

    async def claim(self):
        with observe_admission(self.diagnostics, OWNER):
            return await self.repo.claim_next(OWNER, self.session_id, "offline-worker")

    def count(self, reason):
        return self.diagnostics.snapshot()["counters"][reason]["count"]

    async def test_two_distinct_completed_quiet_denials_are_not_attempts_or_tool_reads(self):
        await self.command("activate")
        for _ in range(3):
            await self.quiet.get(OWNER)
        self.assertEqual(self.diagnostics.sequence, 0)
        for _ in range(3):
            self.assertIsNone(await self.claim())
        snapshot = self.diagnostics.snapshot()
        self.assertEqual(self.count("quiet_denied"), 3)
        episode = snapshot["quiet_denials"][0]
        self.assertEqual(episode["completed_denials"], 3)
        self.assertEqual([sample["sequence"] for sample in episode["samples"]], [1, 2])
        self.assertTrue(all(sample["transaction_exited_successfully"] for sample in episode["samples"]))
        self.assertEqual({sample["session_id"] for sample in episode["samples"]}, {str(self.session_id)})
        self.assertEqual(snapshot["dropped"]["samples"], 1)
        self.assertEqual(self.db.attempts, [])
        self.assertIsNone(self.db.summary)

    async def test_commit_failure_does_not_count_staged_denial_or_leak_exception_text(self):
        await self.command("activate")
        self.tasks.exit_error = RuntimeError("Bearer private-secret")
        with self.assertRaises(RuntimeError):
            await self.claim()
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.count("exception"), 1)
        self.assertEqual(self.diagnostics.snapshot()["quiet_denials"], [])
        self.assertNotIn("private-secret", json.dumps(self.diagnostics.snapshot()))

    async def test_cancel_at_transaction_exit_is_not_a_completed_denial(self):
        await self.command("activate")
        self.tasks.exit_error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.claim()
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.count("cancelled"), 1)

    async def test_unavailable_policy_is_not_a_quiet_denial(self):
        await self.command("activate")
        self.db.payloads[self.db.rows[-1]["payload"]]["availability"] = "removed"
        self.assertIsNone(await self.claim())
        self.assertEqual(self.count("quiet_unavailable"), 1)
        self.assertEqual(self.count("quiet_denied"), 0)

    async def test_real_coalescing_blocked_then_completed_and_existing_summary(self):
        await self.command("activate")
        await self.command("resume")
        with patch("app.attention.quiet_resume.attention_quiet_schema_ready", AsyncMock(return_value=False)):
            self.assertIsNone(await self.claim())
        self.assertEqual(self.count("coalescing_blocked"), 1)
        with patch("app.attention.quiet_resume.attention_quiet_schema_ready", AsyncMock(return_value=True)):
            self.assertIsNone(await self.claim())
            original = self.db.summary
            self.assertIsNone(await self.claim())
        self.assertIsNotNone(original)
        self.assertEqual(self.db.summary, original)
        self.assertEqual(self.count("coalescing_completed"), 2)
        self.assertEqual(self.count("claimed"), 0)
        self.assertEqual(self.db.attempts, [])

    async def test_failed_resume_commit_does_not_report_completed_coalescing(self):
        await self.command("activate")
        await self.command("resume")
        self.tasks.exit_error = RuntimeError("controlled commit failure")
        with patch("app.attention.quiet_resume.attention_quiet_schema_ready", AsyncMock(return_value=True)):
            with self.assertRaises(RuntimeError):
                await self.claim()
        self.assertEqual(self.count("coalescing_completed"), 0)
        self.assertEqual(self.count("exception"), 1)

    async def test_claim_recorded_only_after_owner_transaction_exit(self):
        self.db.candidate = dict(id=uuid4(), source_payload_id=uuid4(), source_namespace="reminder_event",
            task_id=None, owner_id=OWNER, expires_at=None, attempt_count=0, generation=1)
        self.repo._source = AsyncMock(return_value=(SimpleNamespace(speech_ready=True, content=SimpleNamespace()), None))
        add = self.diagnostics.add
        def outside_transaction(*args, **kwargs):
            self.assertFalse(self.db.lock.locked())
            return add(*args, **kwargs)
        self.diagnostics.add = outside_transaction
        claim = await self.claim()
        self.assertEqual(claim.attempt_id, self.db.attempts[0])
        self.assertEqual(self.count("claimed"), 1)
        self.assertEqual(self.count("coalescing_completed"), 0)

    def session(self):
        session = SimpleNamespace(_response_admission_lock=asyncio.Lock(), _closed=asyncio.Event(),
            _realtime_context_ready=asyncio.Event(), _attention_delivery=None, _user_audio_pending=False,
            _normal_turn_pending=False, _pending_response=None, _active_responses={}, _native_playback={},
            context=SimpleNamespace(owner_id=OWNER, jarvis_session_id=self.session_id))
        session._realtime_context_ready.set()
        session.try_attention = MethodType(RealtimeAudioSession.try_attention, session)
        return session

    async def test_actual_manager_no_session_and_native_not_ready_never_read_quiet(self):
        manager = RealtimeSessionManager()
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await manager.deliver_attention(owner_id=OWNER, repository=self.repo,
                speech=None, worker_id="offline", receipt_seconds=1))
        self.assertEqual(self.count("no_session"), 1)
        session = self.session()
        sink = NativeAttentionSink(session, SimpleNamespace(ready=False), self.repo,
                                   session.try_attention, clock=lambda: self.db.now)
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await sink.try_attention(self.repo, worker_id="offline", receipt_seconds=1))
        self.assertEqual(self.count("route_not_ready"), 1)
        self.assertEqual(self.db.calls, [])
        self.assertEqual(self.count("quiet_denied"), 0)

    async def test_actual_foreground_admission_gate_does_not_manufacture_quiet_check(self):
        session = self.session()
        session._normal_turn_pending = True
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await session.try_attention(self.repo, None, worker_id="offline", receipt_seconds=1))
        self.assertEqual(self.db.calls, [])
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.count("admission_unclassified"), 0)

    async def test_actual_session_route_unready_precedes_foreground_without_claim_or_policy_read(self):
        for closed, ready in ((True, True), (False, False), (True, False)):
            with self.subTest(closed=closed, ready=ready):
                self.diagnostics = AdmissionDiagnostics()
                session = self.session()
                if closed:
                    session._closed.set()
                if not ready:
                    session._realtime_context_ready.clear()
                session._normal_turn_pending = True
                with patch.object(self.repo, "claim_next", new_callable=AsyncMock) as claim:
                    with observe_admission(self.diagnostics, OWNER):
                        self.assertIsNone(await session.try_attention(self.repo, None,
                            worker_id="offline", receipt_seconds=1))
                claim.assert_not_awaited()
                self.assertEqual(self.count("route_not_ready"), 1)
                self.assertEqual(self.count("foreground_busy"), 0)
                self.assertEqual(self.count("admission_unclassified"), 0)
                self.assertEqual(self.count("quiet_denied"), 0)
                self.assertEqual(self.count("quiet_unavailable"), 0)
                self.assertEqual(self.diagnostics.snapshot()["quiet_denials"], [])
                self.assertEqual(self.db.calls, [])
                self.assertFalse(session._response_admission_lock.locked())

    async def test_every_actual_foreground_occupancy_is_classified_without_quiet_checks(self):
        for field, value in (("_attention_delivery", object()), ("_user_audio_pending", True),
                             ("_normal_turn_pending", True), ("_pending_response", object()),
                             ("_active_responses", {"response": object()}),
                             ("_native_playback", {"item": object()})):
            with self.subTest(field=field):
                self.diagnostics = AdmissionDiagnostics()
                session = self.session()
                setattr(session, field, value)
                with patch.object(self.repo, "claim_next", new_callable=AsyncMock) as claim:
                    with observe_admission(self.diagnostics, OWNER):
                        self.assertIsNone(await session.try_attention(self.repo, None,
                            worker_id="offline", receipt_seconds=1))
                claim.assert_not_awaited()
                self.assertEqual(self.count("foreground_busy"), 1)
                self.assertEqual(self.count("route_not_ready"), 0)
                self.assertEqual(self.count("admission_unclassified"), 0)
                self.assertEqual(self.count("quiet_denied"), 0)
                self.assertEqual(self.count("quiet_unavailable"), 0)
                self.assertEqual(self.diagnostics.snapshot()["quiet_denials"], [])
                self.assertEqual(self.db.calls, [])
                self.assertIs(getattr(session, field), value)
                self.assertFalse(session._response_admission_lock.locked())

    async def test_actual_early_return_hook_is_poll_and_task_scoped(self):
        session = self.session()
        session._normal_turn_pending = True

        async def attempt():
            return await session.try_attention(self.repo, None, worker_id="offline", receipt_seconds=1)

        self.assertIsNone(await attempt())
        self.assertEqual(self.diagnostics.sequence, 0)
        with observe_admission(self.diagnostics, OWNER):
            child = asyncio.create_task(attempt())
            self.assertIsNone(await asyncio.wait_for(child, 1))
            self.assertTrue(child.done())
            self.assertEqual(self.count("foreground_busy"), 0)
            self.assertIsNone(await attempt())
            self.assertEqual(self.count("foreground_busy"), 1)
        self.assertIsNone(await attempt())
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.count("admission_unclassified"), 0)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.db.calls, [])

    async def test_actual_early_return_foreign_owner_is_not_attributed_to_poll(self):
        session = self.session()
        session.context.owner_id = "other-owner"
        session._normal_turn_pending = True
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await session.try_attention(self.repo, None, worker_id="offline", receipt_seconds=1))
        self.assertEqual(self.count("foreground_busy"), 0)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.count("admission_unclassified"), 1)
        self.assertEqual(self.diagnostics.snapshot()["dropped"]["invalid"], 1)
        self.assertEqual(self.db.calls, [])

    async def test_actual_early_return_survives_observer_failure_without_admission(self):
        session = self.session()
        session._normal_turn_pending = True
        with patch.object(self.diagnostics, "add", side_effect=RuntimeError("PRIVATE_OBSERVER_DETAIL")):
            with observe_admission(self.diagnostics, OWNER):
                self.assertIsNone(await session.try_attention(self.repo, None,
                    worker_id="offline", receipt_seconds=1))
        self.assertEqual(self.count("foreground_busy"), 0)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.db.calls, [])
        self.assertTrue(session._normal_turn_pending)
        self.assertFalse(session._response_admission_lock.locked())
        self.assertNotIn("PRIVATE_OBSERVER_DETAIL", json.dumps(self.diagnostics.snapshot()))

    async def test_observational_hook_cannot_raise_or_change_direct_work(self):
        with observe_admission(self.diagnostics, OWNER):
            record_admission("foreground_busy", owner=OWNER, session_id=self.session_id)
            record_admission("foreground_busy", owner=OWNER, session_id=self.session_id)
        self.assertEqual(self.count("foreground_busy"), 1)
        self.diagnostics.add = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret"))
        with observe_admission(self.diagnostics, OWNER):
            record_admission("no_session")
        self.assertEqual(self.count("quiet_denied"), 0)

    async def test_episode_capacity_detached_snapshot_and_late_child_are_bounded(self):
        for _ in range(MAX_EPISODES + 3):
            with observe_admission(self.diagnostics, OWNER):
                record_admission("quiet_denied", owner=OWNER, session_id=self.session_id,
                                 episode_id=uuid4(), revision=1)
        snapshot = self.diagnostics.snapshot()
        self.assertEqual(len(snapshot["quiet_denials"]), MAX_EPISODES)
        self.assertEqual(snapshot["dropped"]["episodes"], 3)
        snapshot["quiet_denials"].clear()
        self.assertEqual(len(self.diagnostics.snapshot()["quiet_denials"]), MAX_EPISODES)
        release = asyncio.Event()
        async def late():
            await release.wait()
            record_admission("foreground_busy")
        with observe_admission(self.diagnostics, OWNER):
            task = asyncio.create_task(late())
        release.set()
        await task
        self.assertEqual(self.count("foreground_busy"), 0)

    async def test_invalid_identity_and_cross_owner_do_not_escape_or_mix(self):
        with observe_admission(self.diagnostics, OWNER):
            record_admission("quiet_denied", owner="other", episode_id=uuid4(), revision=1)
            record_admission("quiet_denied", session_id="Bearer secret", episode_id=uuid4(), revision=1)
            record_admission("Bearer secret")
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertNotIn("Bearer", json.dumps(self.diagnostics.snapshot()))

    async def test_actual_service_loop_join_then_shutdown_snapshot(self):
        await self.command("activate")
        service = AttentionService(self.repo)
        session = self.session()
        sink = NativeAttentionSink(session, SimpleNamespace(ready=True), self.repo,
                                   session.try_attention, clock=lambda: self.db.now)
        session.try_attention = sink.try_attention
        session._voice_route_runtime = sink.runtime
        manager = RealtimeSessionManager()
        manager._audio_ids_by_owner_id[OWNER] = ["offline-audio"]
        manager._sessions_by_audio_id["offline-audio"] = session
        real_deliver = manager.deliver_attention
        async def deliver(**kwargs):
            result = await real_deliver(**kwargs)
            if service.admission_diagnostics.sequence == 2:
                service._closed = True
            return result
        manager.deliver_attention = deliver
        service.start_voice(manager=manager, owner=OWNER, speech=None,
                            config=SimpleNamespace(poll_seconds=0, receipt_seconds=1), route="native_voice")
        try:
            await asyncio.wait_for(service._task, 1)
        finally:
            await service.close()
        report = {"attention_admission_diagnostics": shutdown_snapshot(service)}
        receipt = json.loads(json.dumps(report))["attention_admission_diagnostics"]
        self.assertTrue(receipt["final"])
        self.assertTrue(receipt["voice_loop_joined"])
        self.assertEqual(receipt["counters"]["quiet_denied"]["count"], 2)
        self.assertEqual(len(receipt["quiet_denials"][0]["samples"]), 2)
        self.assertEqual(receipt["authority"], "diagnostic_only_not_canonical_delivery_evidence")
        self.assertFalse(shutdown_snapshot(None)["final"])


class DirectAdmissionDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.db = ClaimLedger()
        self.repo = AttentionRepository(TaskStore(self.db))
        self.diagnostics = AdmissionDiagnostics()
        self.session = DirectRealtimeAudioSession.__new__(DirectRealtimeAudioSession)
        self.session.context = SimpleNamespace(owner_id=OWNER, jarvis_session_id=uuid4())
        self.session._direct_output_pending = {"offline-response"}
        self.session._response_admission_lock = asyncio.Lock()
        self.session._closed = asyncio.Event()
        self.session._realtime_context_ready = asyncio.Event()
        self.session._realtime_context_ready.set()
        self.session._attention_delivery = None
        self.session._user_audio_pending = False
        self.session._normal_turn_pending = False
        self.session._pending_response = None
        self.session._active_responses = {}
        self.session._native_playback = {}

    async def call_direct(self):
        return await self.session.try_attention(self.repo, None, worker_id="offline", receipt_seconds=1)

    def count(self, reason):
        return self.diagnostics.snapshot()["counters"][reason]["count"]

    async def test_real_direct_pending_branch_is_foreground_not_quiet_or_unclassified(self):
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await self.call_direct())
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.count("admission_unclassified"), 0)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.diagnostics.snapshot()["quiet_denials"], [])
        self.assertEqual(self.db.calls, [])
        self.assertEqual(self.session._direct_output_pending, {"offline-response"})
        self.assertFalse(self.session._response_admission_lock.locked())

    async def test_pending_branch_keeps_its_existing_precedence_over_base_readiness(self):
        self.session._closed.set()
        self.session._realtime_context_ready.clear()
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await self.call_direct())
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.count("route_not_ready"), 0)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.db.calls, [])

    async def test_empty_direct_set_still_delegates_to_real_base_policy_and_transaction(self):
        source = ToolInvocationContext(request_event_id=uuid4(), request_revision=1)
        self.db.authorize(OWNER, self.session.context.jarvis_session_id, source)
        await OwnerQuietRepository(self.db).save(OWNER, self.session.context.jarvis_session_id,
            source, QuietCommand(action="activate", expected_revision=0))
        self.db.calls.clear()
        self.session._direct_output_pending.clear()
        for _ in range(2):
            with observe_admission(self.diagnostics, OWNER):
                self.assertIsNone(await self.call_direct())
        self.assertEqual(self.count("foreground_busy"), 0)
        self.assertEqual(self.count("quiet_denied"), 2)
        self.assertEqual(len(self.diagnostics.snapshot()["quiet_denials"][0]["samples"]), 2)
        self.assertEqual(sum("attention_quiet:current" in sql for sql, _ in self.db.calls), 2)
        self.assertEqual(self.db.attempts, [])

    async def test_broken_observer_does_not_raise_fall_through_or_mutate_pending_output(self):
        def broken(*args, **kwargs):
            raise RuntimeError("private diagnostic failure")
        self.diagnostics.add = broken
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await self.call_direct())
        self.assertEqual(self.db.calls, [])
        self.assertEqual(self.session._direct_output_pending, {"offline-response"})
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertNotIn("private diagnostic failure", json.dumps(self.diagnostics.snapshot()))

    async def test_real_direct_hook_is_owner_task_and_observation_scoped(self):
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await self.call_direct())
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertIsNone(await self.call_direct())  # Scope has ended.
        self.session.context.owner_id = "other-offline-owner"
        with observe_admission(self.diagnostics, OWNER):
            self.assertIsNone(await self.call_direct())
        self.session.context.owner_id = OWNER
        with observe_admission(self.diagnostics, OWNER):
            child = asyncio.create_task(self.call_direct())
            self.assertIsNone(await child)  # Inherited ContextVar is not task ownership.
        self.assertTrue(child.done())
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.count("quiet_denied"), 0)
        self.assertEqual(self.db.calls, [])

    async def test_direct_early_return_introduces_no_suspension(self):
        call = self.call_direct()
        try:
            with observe_admission(self.diagnostics, OWNER):
                with self.assertRaises(StopIteration) as completed:
                    call.send(None)
                self.assertIsNone(completed.exception.value)
        finally:
            call.close()
        self.assertEqual(self.count("foreground_busy"), 1)
        self.assertEqual(self.db.calls, [])


if __name__ == "__main__":
    unittest.main()
