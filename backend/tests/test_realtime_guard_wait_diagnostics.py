"""Cancelled native writer guard through real canonical/source guards; no live I/O."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.guard_diagnostics import GuardCancellationDiagnostics, guard_context, guard_wait_stage
from app.knowledge.models import SourceError
from app.knowledge.sources import SourceRepository
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import GuardedPayloadBinding, SessionScope
from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.source_context import LiveResponseSources
from app.realtime.telemetry import RealtimeTrace
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.settings import JarvisSettings


def diagnostic_rows(logs):
    return [row for record in logs.records if "realtime_trace " in record.getMessage()
            for row in [json.loads(record.getMessage().split("realtime_trace ", 1)[1])]
            if row["event"].startswith("provider_guard_wait")]


class Storage:
    """Only external DB results/waits are controlled; original guard order runs."""
    def __init__(self, target=None):
        self.target = target
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.steps = []
        self.pool_depth = self.transaction_depth = self.source_calls = 0
        self.available = self.scope_available = True

    async def step(self, name):
        self.steps.append(name)
        if name == self.target:
            self.reached.set()
            await self.release.wait()

    @asynccontextmanager
    async def acquire(self):
        await self.step("pool_acquire")
        self.pool_depth += 1
        try:
            yield self
        finally:
            self.pool_depth -= 1

    @asynccontextmanager
    async def transaction(self):
        await self.step("transaction_enter")
        self.transaction_depth += 1
        try:
            yield self
        finally:
            self.transaction_depth -= 1

    def is_in_transaction(self):
        return self.transaction_depth > 0

    async def execute(self, statement, *args):
        assert self.is_in_transaction()
        assert "pg_advisory_xact_lock" in statement
        await self.step("memory_owner_lock")

    async def lock_owner(self, connection, owner):
        assert connection is self and self.is_in_transaction()
        await self.step("task_owner_lock")

    async def dependencies(self, connection, dependencies):
        assert connection is self and self.is_in_transaction()
        await self.step("dependency_revalidation")
        return ()

    async def fetchval(self, statement, *args):
        assert self.is_in_transaction()
        if statement.startswith("select true from jarvis_sessions"):
            await self.step("scope_revalidation")
            return self.scope_available
        assert "lock_jarvis_copy_sources" in statement
        self.source_calls += 1
        await self.step("canonical_source_lock" if self.source_calls == 1 else "response_source_lock")
        return self.available


class GuardWaitTests(unittest.IsolatedAsyncioTestCase):
    def setup_writer(self, target=None, *, trace=True):
        store = Storage(target)
        scope = SessionScope(owner_id="guard-fixture", session_id=uuid4())
        binding = GuardedPayloadBinding(scope=scope, payload_root_ids=(uuid4(),), dependencies=())
        binder = CanonicalContextBinder(SourceRepository(store), scope,
            clock=lambda: datetime.now(timezone.utc))
        binder.tasks = SimpleNamespace(lock_owner=store.lock_owner, lock_owner_shared=store.lock_owner)
        binder._validate_dependencies = store.dependencies
        sources = LiveResponseSources(store, owner_id=scope.owner_id, session_id=scope.session_id,
                                      audio_session_id="guard-audio")
        sources.observe((uuid4(),))
        runtime = object.__new__(RealtimeVoiceRouteRuntime)
        runtime.session = SimpleNamespace(_source_context=sources)
        runtime._payload_roots = lambda payload: binding.payload_root_ids
        runtime._atomic_guard = lambda payload: binder.guard(binding)
        runtime._readiness_diagnostic = lambda *args, **kwargs: None
        provider = OpenAIRealtimeConnection(settings=JarvisSettings(_env_file=None,
            OPENAI_API_KEY="fixture-only-not-a-real-api-key"), owner_id=scope.owner_id)
        wire = SimpleNamespace(send=AsyncMock(), recv=AsyncMock(), close=AsyncMock())
        provider._websocket = wire
        if trace:
            provider.bind_trace(RealtimeTrace(provider="openai", audio_session_id="guard-audio",
                                             jarvis_session_id=scope.session_id))
        self.addAsyncCleanup(provider.close)
        payload = {"type": "conversation.item.create", "item": {
            "id": "replacement", "content": [{"text": "PRIVATE_CONTEXT"}]}}
        return store, scope, binding, binder, sources, runtime, provider, wire, payload

    async def cancel_at(self, target, *, trace=True, delete_first=False):
        parts = self.setup_writer(target, trace=trace)
        store, _, _, _, sources, runtime, provider, wire, payload = parts
        if delete_first:
            await provider.send_event({"type": "conversation.item.delete", "item_id": "old-context"})
            wire.recv.return_value = json.dumps({"type": "conversation.item.deleted", "item_id": "old-context"})
            self.assertEqual((await provider.receive_event())["item_id"], "old-context")
        task = asyncio.create_task(provider.send_event(lambda: payload,
            write_guard=lambda: runtime._source_guard(SimpleNamespace())))
        try:
            await asyncio.wait_for(store.reached.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            store.release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(store.pool_depth, 0)
        self.assertEqual(store.transaction_depth, 0)
        self.assertIsNone(sources._current_connection())
        self.assertEqual(provider._foreground_waiters, 0)
        self.assertEqual([json.loads(call.args[0])["type"] for call in wire.send.await_args_list],
                         ["conversation.item.delete"] if delete_first else [])
        return parts

    async def test_each_real_nested_guard_wait_identifies_cancellation_without_a_write(self):
        for stage in ("pool_acquire", "transaction_enter", "memory_owner_lock", "task_owner_lock",
                      "scope_revalidation", "canonical_source_lock", "dependency_revalidation",
                      "response_source_lock"):
            with self.subTest(stage=stage), self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                store, *_ = await self.cancel_at(stage)
            rows = diagnostic_rows(logs)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["phase"], stage)
            self.assertEqual(rows[0]["status"], "await_cancelled")
            self.assertFalse(rows[0]["write_possible"])
            self.assertFalse(rows[0]["send_returned"])
            self.assertEqual(len(rows[0]["dispatch_id"]), 32)
            self.assertGreaterEqual(rows[0]["duration_ms"], 0)
            self.assertNotIn("PRIVATE", str(rows))
            self.assertNotIn("owner_id", rows[0])
            self.assertEqual(rows[0]["boundary"], "completed_stages_inclusive")
            self.assertEqual({key for key in rows[0] if key.startswith("guard_completed_")},
                             {f"guard_completed_{name}_ms" for name in store.steps[:-1]})

    async def test_acknowledged_delete_does_not_become_replacement_write_or_ack(self):
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            await self.cancel_at("memory_owner_lock", delete_first=True)
        rows = diagnostic_rows(logs)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["write_possible"])
        self.assertNotIn("item_id", rows[0])
        all_rows = [json.loads(r.getMessage().split("realtime_trace ", 1)[1]) for r in logs.records]
        writes = [r for r in all_rows if r.get("phase") == "send" and r.get("status") == "completed"]
        self.assertEqual([r["event_type"] for r in writes], ["conversation.item.delete"])
        self.assertNotEqual(writes[0]["dispatch_id"], rows[0]["dispatch_id"])

    async def test_success_preserves_original_lock_order_and_has_no_cancellation_receipt(self):
        store, _, _, _, sources, runtime, provider, wire, payload = self.setup_writer()
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            self.assertTrue(await provider.send_event(payload,
                write_guard=lambda: runtime._source_guard(SimpleNamespace())))
        self.assertEqual(store.steps, ["pool_acquire", "transaction_enter", "memory_owner_lock",
            "task_owner_lock", "scope_revalidation", "canonical_source_lock",
            "dependency_revalidation", "response_source_lock"])
        self.assertEqual(diagnostic_rows(logs), [])
        self.assertEqual(json.loads(wire.send.call_args.args[0]), payload)
        self.assertEqual((store.pool_depth, store.transaction_depth), (0, 0))
        self.assertIsNone(sources._current_connection())

    async def test_scope_and_source_rejection_still_raise_without_false_cancellation(self):
        for scope_ok, expected in ((False, "live_state_session_unavailable"), (True, "live_state_context_revoked")):
            with self.subTest(scope_ok=scope_ok):
                store, _, _, _, _, runtime, provider, wire, payload = self.setup_writer()
                store.scope_available, store.available = scope_ok, False
                with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                    with self.assertRaises(SourceError) as error:
                        await provider.send_event(payload, write_guard=lambda: runtime._source_guard(None))
                self.assertEqual(error.exception.code, expected)
                self.assertEqual(diagnostic_rows(logs), [])
                wire.send.assert_not_awaited()
                self.assertEqual((store.pool_depth, store.transaction_depth), (0, 0))

    async def test_no_opt_in_trace_means_no_diagnostic_but_same_cancellation(self):
        with self.assertNoLogs("app.realtime.telemetry", "INFO"):
            await self.cancel_at("memory_owner_lock", trace=False)

    async def test_child_task_cannot_supply_parent_guard_stage(self):
        *_, provider, wire, payload = self.setup_writer()
        @asynccontextmanager
        async def guard():
            async def child():
                with guard_wait_stage("response_source_lock"):
                    raise asyncio.CancelledError()
            await asyncio.gather(asyncio.create_task(child()), return_exceptions=True)
            raise asyncio.CancelledError()
            yield
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await provider.send_event(payload, write_guard=guard)
        self.assertEqual(diagnostic_rows(logs)[0]["phase"], "unclassified_guard")
        wire.send.assert_not_awaited()

    async def test_unknown_private_stage_redacted_and_diagnostic_limit_is_bounded(self):
        *_, provider, wire, payload = self.setup_writer()
        @asynccontextmanager
        async def guard():
            with guard_wait_stage("PRIVATE_SQL_OR_CONTENT"):
                raise asyncio.CancelledError()
            yield
        with patch("app.guard_diagnostics.MAX_GUARD_CANCELLATION_RECORDS", 2):
            with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
                for _ in range(5):
                    with self.assertRaises(asyncio.CancelledError):
                        await provider.send_event(payload, write_guard=guard)
        rows = diagnostic_rows(logs)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r.get("phase") for r in rows[:2]], ["unclassified_guard"] * 2)
        self.assertEqual(rows[-1]["event"], "provider_guard_wait_truncated")
        self.assertNotIn("PRIVATE", str(rows))
        wire.send.assert_not_awaited()

    async def test_logging_and_clock_failure_do_not_replace_cancellation(self):
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=RuntimeError("PRIVATE_LOG")), \
                patch("app.guard_diagnostics._clock", return_value=None):
            await self.cancel_at("scope_revalidation")

    async def test_inner_timeout_preserves_timeout_type_and_reports_actual_cancelled_wait(self):
        *_, provider, wire, payload = self.setup_writer()
        @asynccontextmanager
        async def guard():
            async with asyncio.timeout(0.01):
                with guard_wait_stage("task_owner_lock"):
                    await asyncio.Event().wait()
                yield
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(TimeoutError):
                await provider.send_event(payload, write_guard=guard)
        self.assertEqual(diagnostic_rows(logs)[0]["phase"], "task_owner_lock")
        wire.send.assert_not_awaited()

    async def test_cancel_after_guard_during_send_does_not_claim_prewrite_cancellation(self):
        *_, provider, wire, payload = self.setup_writer()
        @asynccontextmanager
        async def guard():
            with guard_wait_stage("scope_revalidation"):
                await asyncio.sleep(0)
            yield
        wire.send.side_effect = asyncio.CancelledError()
        with self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(asyncio.CancelledError):
                await provider.send_event(payload, write_guard=guard)
        self.assertEqual(diagnostic_rows(logs), [])
        wire.send.assert_awaited_once()

    async def test_context_wrapper_preserves_exit_arguments_and_suppression(self):
        context = SimpleNamespace(__aenter__=AsyncMock(return_value="value"),
                                  __aexit__=AsyncMock(return_value=True))
        async with guard_context(context, "transaction_enter") as value:
            self.assertEqual(value, "value")
            raise ValueError("PRIVATE")
        self.assertIs(context.__aexit__.call_args.args[0], ValueError)
        context.__aenter__.assert_awaited_once()
        context.__aexit__.assert_awaited_once()

    async def test_completed_stage_totals_distinguish_earlier_waits_from_cancelled_stage(self):
        diagnostics = GuardCancellationDiagnostics()
        trace = RealtimeTrace(provider="openai", audio_session_id="guard-audio",
                              jarvis_session_id=uuid4())
        now = [0.0]
        with patch("app.guard_diagnostics._clock", side_effect=lambda: now[0]), \
                self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(asyncio.CancelledError), diagnostics.observe(trace, "fixture-dispatch"):
                with guard_wait_stage("memory_owner_lock"):
                    now[0] += 1.2
                with guard_wait_stage("task_owner_lock"):
                    now[0] += 0.3
                with guard_wait_stage("memory_owner_lock"):
                    now[0] += 0.7
                with guard_wait_stage("scope_revalidation"):
                    now[0] += 0.1
                    raise asyncio.CancelledError()
        row, = diagnostic_rows(logs)
        self.assertAlmostEqual(row["guard_completed_memory_owner_lock_ms"], 1900)
        self.assertAlmostEqual(row["guard_completed_task_owner_lock_ms"], 300)
        self.assertAlmostEqual(row["duration_ms"], 100)
        self.assertNotIn("guard_completed_scope_revalidation_ms", row)

    async def test_completed_nested_stages_are_inclusive_and_child_stages_are_excluded(self):
        diagnostics = GuardCancellationDiagnostics()
        trace = RealtimeTrace(provider="openai", audio_session_id="guard-audio",
                              jarvis_session_id=uuid4())
        now = [0.0]

        async def child():
            with guard_wait_stage("task_owner_lock"):
                await asyncio.sleep(0)

        with patch("app.guard_diagnostics._clock", side_effect=lambda: now[0]), \
                self.assertLogs("app.realtime.telemetry", "INFO") as logs:
            with self.assertRaises(asyncio.CancelledError), diagnostics.observe(trace, "fixture-dispatch"):
                with guard_wait_stage("scope_revalidation"):
                    now[0] = 0.2
                    with guard_wait_stage("memory_owner_lock"):
                        now[0] = 0.5
                    now[0] = 1.0
                await asyncio.create_task(child())
                with guard_wait_stage("PRIVATE_SQL_OR_CONTENT"):
                    now[0] = 1.1
                with guard_wait_stage("pool_acquire"):
                    now[0] = 1.2
                    raise asyncio.CancelledError()
        row, = diagnostic_rows(logs)
        self.assertEqual(row["boundary"], "completed_stages_inclusive")
        self.assertAlmostEqual(row["guard_completed_scope_revalidation_ms"], 1000)
        self.assertAlmostEqual(row["guard_completed_memory_owner_lock_ms"], 300)
        self.assertAlmostEqual(row["guard_completed_unclassified_guard_ms"], 100)
        self.assertNotIn("guard_completed_task_owner_lock_ms", row)
        self.assertNotIn("PRIVATE", str(row))
