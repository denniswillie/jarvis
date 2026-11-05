"""Offline call-boundary diagnostics; no claims about individual SQL durations."""
import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.assistant.context import AssistantContextBuilder
from app.assistant.context_providers import (
    ContextBuildRequest, CurrentTimeContextProvider, SessionContextProvider,
)
from app.assistant.memory_context import WorkingMemoryContextProvider
from app.knowledge.current_context import CurrentSourceContext, SourceContextProvider
from app.knowledge.models import SourceError, SourceVersion


SOURCE_LOG = "app.knowledge.current_context"
MEMORY_LOG = "app.assistant.memory_context"
PRIVATE = "private-memory-and-credential-canary"
NOW = datetime(2026, 10, 31, tzinfo=UTC)


class ContextPreparationDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def request(self):
        return ContextBuildRequest(owner_id=PRIVATE, session=SimpleNamespace(id=uuid4()),
                                   recent_messages=(), requested_at_utc=NOW)

    def diagnostic(self, captured, *, stages):
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertNotIn(PRIVATE, message)
        self.assertLess(len(message), 2500)
        result = json.loads(message.split(" ", 1)[1])
        self.assertEqual(len(result["stages"]), stages)
        self.assertGreaterEqual(result["elapsed_ms"], 0)
        self.assertIsNone(result["pool_acquire_ms"])
        self.assertIsNone(result["sql_statement_ms"])
        self.assertEqual(result["timing_scope"], "service_calls_including_pool_wait")
        for stage in result["stages"]:
            self.assertGreaterEqual(stage["elapsed_ms"], 0)
        return result

    def source_provider(self, *, selected=True, timeout=1.5):
        area, root = uuid4(), uuid4()
        source = SourceVersion(owner_id=PRIVATE, area_id=area, source_id=uuid4(), revision=2,
            generation=2, payload_id=root, document_id=PRIVATE, bank_name=PRIVATE,
            content_digest="a" * 64, source_type="user_transcript", trust="user_input",
            source_time=NOW, time_basis=PRIVATE, recorded_at=NOW,
            content={"text": PRIVATE}, is_current=True, ingestion_status="pending")
        repository = SimpleNamespace(current_snapshot=AsyncMock(return_value={
            "observed_at": NOW, "total_sources": 1, "available_sources": 1,
            "not_searchable_sources": 1, "failed_ingestion_sources": 0,
            "uncertain_ingestion_sources": 0, "sources": [source],
        }))
        service = CurrentSourceContext(repository, timeout_seconds=timeout)
        service._provenance = SimpleNamespace(bind_context=AsyncMock(return_value=(root,)))
        selection = {"area_id": str(area), "selection_revision": 3, "private": PRIVATE}
        selections = SimpleNamespace(read=AsyncMock(return_value=selection)) if selected else None
        return SourceContextProvider(service, area, selections), repository, root

    def memory_service(self):
        state = SimpleNamespace(revision=4, state={"summary": PRIVATE}, updated_at=NOW,
            last_observation_ended_at=NOW, expires_at=NOW + timedelta(minutes=1))
        return SimpleNamespace(get_session_working_memory=AsyncMock(return_value=state),
                               get_owner_working_memory=AsyncMock(return_value=None))

    async def test_source_success_preserves_roots_guards_and_one_bounded_record(self):
        provider, repository, root = self.source_provider()
        request = self.request()
        # Deterministic clock affects diagnostics only, not asyncio deadlines.
        ticks = iter(range(10))
        with patch(SOURCE_LOG + ".monotonic", side_effect=lambda: next(ticks)), self.assertLogs(SOURCE_LOG, "INFO") as logs:
            section = await provider.build_section(request)
        result = self.diagnostic(logs, stages=4)
        self.assertEqual(result["elapsed_ms"], 9000)
        self.assertEqual([s["elapsed_ms"] for s in result["stages"]], [1000] * 4)
        self.assertEqual([s["stage"] for s in result["stages"]], ["selection_read",
            "source_snapshot_and_projection", "provenance_binding", "selection_revalidation"])
        self.assertEqual(result["outcome"], "available")
        self.assertEqual(result["deadline_ms"], 1500)
        self.assertIsNone(result["failed_stage"])
        self.assertEqual(section.source_payload_ids, (root,))
        self.assertEqual(section.payload["selection"]["selection_revision"], 3)
        self.assertEqual(provider.selections.read.await_count, 2)
        repository.current_snapshot.assert_awaited_once()
        provider.service._provenance.bind_context.assert_awaited_once()

    async def test_source_timeout_reports_each_existing_boundary_not_sql_duration(self):
        cases = ("selection_read", "source_snapshot_and_projection", "provenance_binding", "selection_revalidation")
        for target in cases:
            with self.subTest(target=target):
                provider, repository, _ = self.source_provider(timeout=0.01)
                joined = asyncio.Event()

                async def blocked(*args, **kwargs):
                    try:
                        await asyncio.Future()
                    finally:
                        joined.set()

                if target == "selection_read":
                    provider.selections.read.side_effect = blocked
                elif target == "source_snapshot_and_projection":
                    repository.current_snapshot.side_effect = blocked
                elif target == "provenance_binding":
                    provider.service._provenance.bind_context.side_effect = blocked
                else:
                    original = provider.selections.read.return_value
                    calls = 0

                    async def revalidate(*args, **kwargs):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            return original
                        return await blocked()

                    provider.selections.read.side_effect = revalidate
                with self.assertLogs(SOURCE_LOG, "WARNING") as logs:
                    with self.assertRaises(TimeoutError):
                        await provider.build_section(self.request())
                result = self.diagnostic(logs, stages=cases.index(target) + 1)
                self.assertEqual(result["failed_stage"], target)
                self.assertEqual(result["reason"], "preparation_deadline_exceeded")
                self.assertEqual(result["outcome"], "unavailable")
                self.assertTrue(joined.is_set())

    async def test_source_operation_timeout_is_not_mislabelled_outer_deadline(self):
        provider, repository, _ = self.source_provider(selected=False)
        repository.current_snapshot.side_effect = TimeoutError(PRIVATE)
        with self.assertLogs(SOURCE_LOG, "WARNING") as logs:
            with self.assertRaises(TimeoutError):
                await provider.build_section(self.request())
        self.assertEqual(self.diagnostic(logs, stages=1)["reason"], "operation_timeout")

    async def test_selection_change_remains_rejected_and_logs_no_selection_contents(self):
        provider, _, _ = self.source_provider()
        original = provider.selections.read.return_value
        provider.selections.read.side_effect = [original, {**original, "selection_revision": 4}]
        with self.assertLogs(SOURCE_LOG, "WARNING") as logs:
            with self.assertRaises(SourceError) as raised:
                await provider.build_section(self.request())
        self.assertEqual(raised.exception.code, "knowledge_selection_changed")
        result = self.diagnostic(logs, stages=4)
        self.assertEqual(result["reason"], "selection_changed")
        self.assertEqual(result["failed_stage"], "selection_revalidation")

    async def test_withdrawn_source_guard_and_generic_error_keep_existing_fallback(self):
        for error, reason in ((SourceError("knowledge_source_unavailable", PRIVATE), "source_guard_rejected"),
                              (RuntimeError(PRIVATE), "preparation_failed")):
            with self.subTest(reason=reason):
                provider, _, _ = self.source_provider(selected=False)
                provider.service._provenance.bind_context.side_effect = error
                builder = AssistantContextBuilder(providers=(
                    CurrentTimeContextProvider(), SessionContextProvider(), provider,
                ))
                with self.assertLogs(SOURCE_LOG, "WARNING") as logs, self.assertLogs("app.assistant.context", "WARNING"):
                    context = await builder.build(owner_id=PRIVATE, session=None, recent_messages=[])
                result = self.diagnostic(logs, stages=2)
                self.assertEqual(result["reason"], reason)
                self.assertEqual(context.knowledge_sources["status"], "unavailable")
                self.assertNotIn("available_sources", context.knowledge_sources)

    async def test_source_caller_cancellation_does_not_become_unavailable_success(self):
        provider, repository, _ = self.source_provider(selected=False)
        started, joined = asyncio.Event(), asyncio.Event()

        async def blocked(*args, **kwargs):
            started.set()
            try:
                await asyncio.Future()
            finally:
                joined.set()

        repository.current_snapshot.side_effect = blocked
        with self.assertLogs(SOURCE_LOG, "INFO") as logs:
            task = asyncio.create_task(provider.build_section(self.request()))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(joined.is_set())
        result = self.diagnostic(logs, stages=1)
        self.assertEqual((result["outcome"], result["reason"]), ("cancelled", "caller_cancelled"))

    async def test_memory_success_logs_only_revision_and_empty_is_not_unavailable(self):
        service = self.memory_service()
        with self.assertLogs(MEMORY_LOG, "INFO") as logs:
            section = await WorkingMemoryContextProvider(service).build_section(self.request())
        result = self.diagnostic(logs, stages=2)
        self.assertTrue(section.payload["available"])
        self.assertEqual(result["deadline_ms"], 350)
        self.assertEqual(result["stages"][0]["revision"], 4)
        self.assertEqual(result["stages"][1]["outcome"], "empty")
        service.get_session_working_memory.assert_awaited_once()
        service.get_owner_working_memory.assert_awaited_once()

    async def test_memory_timeout_records_completed_vs_cancelled_read_and_joins(self):
        service = self.memory_service()
        joined = asyncio.Event()

        async def blocked(**kwargs):
            try:
                await asyncio.Future()
            finally:
                joined.set()

        service.get_owner_working_memory.side_effect = blocked
        with self.assertLogs(MEMORY_LOG, "WARNING") as logs:
            section = await WorkingMemoryContextProvider(service, 0.01).build_section(self.request())
        result = self.diagnostic(logs, stages=2)
        self.assertEqual(result["reason"], "read_deadline_exceeded")
        self.assertEqual([s["outcome"] for s in result["stages"]], ["available", "cancelled"])
        self.assertEqual(section.payload, {"available": False, "session": None, "owner": None})
        self.assertTrue(joined.is_set())

    async def test_memory_reads_remain_parallel_and_caller_cancellation_joins_both(self):
        both, joined = asyncio.Event(), set()
        started = set()

        async def blocked(name):
            started.add(name)
            if len(started) == 2:
                both.set()
            try:
                await asyncio.Future()
            finally:
                joined.add(name)

        service = SimpleNamespace(get_session_working_memory=lambda **_: blocked("session"),
                                  get_owner_working_memory=lambda **_: blocked("owner"))
        with self.assertLogs(MEMORY_LOG, "INFO") as logs:
            task = asyncio.create_task(WorkingMemoryContextProvider(service).build_section(self.request()))
            await asyncio.wait_for(both.wait(), 0.2)
            # The caller can continue while both optional reads are suspended.
            self.assertFalse(task.done())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(joined, {"session", "owner"})
        result = self.diagnostic(logs, stages=2)
        self.assertEqual(result["reason"], "caller_cancelled")
        self.assertEqual([s["outcome"] for s in result["stages"]], ["cancelled", "cancelled"])

    async def test_memory_read_error_propagates_without_logging_exception_text(self):
        service = self.memory_service()
        service.get_session_working_memory.side_effect = RuntimeError(PRIVATE)
        with self.assertLogs(MEMORY_LOG, "WARNING") as logs:
            with self.assertRaises(ExceptionGroup):
                await WorkingMemoryContextProvider(service).build_section(self.request())
        result = self.diagnostic(logs, stages=2)
        self.assertEqual(result["reason"], "read_failed")
        self.assertEqual(result["stages"][0]["outcome"], "failed")

    async def test_missing_service_or_session_has_explicit_zero_read_reason(self):
        for service, session, reason in ((None, self.request().session, "service_unconfigured"),
                                         (self.memory_service(), None, "session_missing")):
            with self.subTest(reason=reason):
                request = ContextBuildRequest(PRIVATE, session, (), NOW)
                with self.assertLogs(MEMORY_LOG, "INFO") as logs:
                    section = await WorkingMemoryContextProvider(service).build_section(request)
                self.assertFalse(section.payload["available"])
                self.assertEqual(self.diagnostic(logs, stages=0)["reason"], reason)

    async def test_concurrent_source_attempts_do_not_share_stage_or_failure_state(self):
        provider, repository, _ = self.source_provider(selected=False)
        held, release = asyncio.Event(), asyncio.Event()
        snapshot = repository.current_snapshot.return_value
        calls = 0

        async def snapshot_read(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                held.set()
                await release.wait()
                raise RuntimeError(PRIVATE)
            return snapshot

        repository.current_snapshot.side_effect = snapshot_read
        first, second = self.request(), self.request()
        with self.assertLogs(SOURCE_LOG, "INFO") as logs:
            task = asyncio.create_task(provider.build_section(first))
            await held.wait()
            try:
                result = await provider.build_section(second)
            finally:
                release.set()
            with self.assertRaises(RuntimeError):
                await task
        self.assertEqual(result.payload["status"], "available")
        self.assertEqual(len(logs.records), 2)
        records = {json.loads(r.getMessage().split(" ", 1)[1])["session_id"]:
                   json.loads(r.getMessage().split(" ", 1)[1]) for r in logs.records}
        self.assertEqual(records[str(first.session.id)]["reason"], "preparation_failed")
        self.assertEqual(records[str(second.session.id)]["reason"], "complete")
        self.assertEqual(len(records[str(first.session.id)]["stages"]), 1)
        self.assertEqual(len(records[str(second.session.id)]["stages"]), 2)
