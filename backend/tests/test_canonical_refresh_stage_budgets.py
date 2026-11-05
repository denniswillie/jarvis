"""Bounded read/publication scheduling without changing source authority."""
import asyncio
from datetime import UTC, datetime
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.models import SessionScope
from app.live_state.refresh import CanonicalRefreshWorker
from app.live_state.runtime import LiveStateRuntime
from app.runtime.events import DomainEventBus, DomainEventName


class RefreshStageBudgetTests(unittest.IsolatedAsyncioTestCase):
    def worker(self, read, apply, invalidate, **options):
        return CanonicalRefreshWorker(scope=SessionScope(owner_id="stage-budget-owner", session_id=uuid4()),
            bus=DomainEventBus(), event_names=(DomainEventName.OWNER_WORKING_MEMORY_UPDATED,),
            read=read, apply=apply, invalidate=invalidate, interval_seconds=60,
            **({"timeout_seconds": 0.12} | options))

    async def run_case(self, *, separate, read_delay=0.08, apply_delay=0.10, error=None):
        terminal, released = asyncio.Event(), asyncio.Event()
        calls, reasons = [], []

        async def read():
            calls.append("read")
            try:
                await asyncio.sleep(read_delay)
                return 42
            finally:
                if "apply" not in calls:
                    released.set()

        async def apply(value):
            self.assertEqual(value, 42)
            calls.append("apply")
            released.clear()
            try:
                await asyncio.sleep(apply_delay)
                if error is not None:
                    raise error
                worker._closed = True
                terminal.set()
            finally:
                released.set()

        async def invalidate(reason):
            reasons.append(reason)
            terminal.set()

        worker = self.worker(read, apply, invalidate, apply_timeout_seconds=separate)
        worker.start()
        try:
            await asyncio.wait_for(terminal.wait(), 2)
            await asyncio.sleep(0)
            self.assertTrue(released.is_set())
            return worker, calls, reasons
        finally:
            # The success path closes the loop itself; failure is joined here.
            worker._closed = False
            await worker.close()
            self.assertTrue(worker._task.done())

    async def test_separate_publication_budget_finishes_cold_work(self):
        worker, calls, reasons = await self.run_case(separate=0.2)
        self.assertEqual(calls, ["read", "apply"])
        self.assertEqual((worker.refresh_count, worker.failure_count), (1, 0))
        self.assertIsNone(worker.last_error)
        self.assertEqual(reasons, [])

    async def test_default_combined_budget_is_unchanged(self):
        worker, calls, reasons = await self.run_case(separate=None)
        self.assertEqual(calls, ["read", "apply"])
        self.assertEqual((worker.refresh_count, worker.failure_count), (0, 1))
        self.assertEqual((worker.last_error, worker.last_error_stage), ("TimeoutError", "apply"))
        self.assertEqual(reasons, ["canonical_refresh_unavailable"])

    async def test_read_budget_does_not_borrow_from_publication(self):
        worker, calls, reasons = await self.run_case(separate=1, read_delay=1)
        self.assertEqual(calls, ["read"])
        self.assertEqual((worker.refresh_count, worker.failure_count), (0, 1))
        self.assertEqual(worker.last_error_stage, "read")
        self.assertEqual(reasons, ["canonical_refresh_unavailable"])

    async def test_stalled_publication_is_cancelled_and_withdrawn(self):
        worker, calls, reasons = await self.run_case(separate=0.05, read_delay=0, apply_delay=1)
        self.assertEqual((worker.refresh_count, worker.failure_count), (0, 1))
        self.assertEqual(worker.last_error_stage, "apply")
        self.assertEqual(reasons, ["canonical_refresh_unavailable"])

    async def test_source_rejection_still_invalidates_without_publishing_success(self):
        worker, _, reasons = await self.run_case(separate=1, read_delay=0, apply_delay=0,
            error=SourceError("live_state_context_revoked", "Synthetic removed source"))
        self.assertEqual((worker.refresh_count, worker.failure_count), (0, 1))
        self.assertEqual(worker.last_error_code, "live_state_context_revoked")
        self.assertEqual(reasons, ["canonical_refresh_unavailable"])

    async def test_close_during_apply_joins_without_reclassifying_failure(self):
        entered, released = asyncio.Event(), asyncio.Event()

        async def apply(value):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                released.set()

        invalidate = AsyncMock()
        worker = self.worker(AsyncMock(return_value=42), apply, invalidate, apply_timeout_seconds=1)
        worker.start()
        await asyncio.wait_for(entered.wait(), 1)
        await worker.close()
        self.assertTrue(released.is_set())
        self.assertTrue(worker._task.done())
        self.assertEqual((worker.refresh_count, worker.failure_count), (0, 0))
        invalidate.assert_not_awaited()

    def test_invalid_apply_deadlines_are_rejected(self):
        for deadline in (0, -1, float("inf"), float("nan")):
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                self.worker(AsyncMock(), AsyncMock(), AsyncMock(), apply_timeout_seconds=deadline)

    def test_runtime_uses_evidence_ttl_only_for_background_publication(self):
        sources = type("Sources", (), {"pool": object()})()
        runtime = LiveStateRuntime(scope=SessionScope(owner_id="stage-budget-owner", session_id=uuid4()),
            sources=sources, context=object(), area_selections=None, bus=DomainEventBus(),
            access_check=lambda owner: None, clock=lambda: datetime.now(UTC),
            timeout_seconds=2, ttl_seconds=10)
        self.assertEqual(runtime.worker.timeout_seconds, 2)
        self.assertEqual(runtime.worker.apply_timeout_seconds, 10)
        self.assertEqual(runtime.binder.timeout_seconds, 2)
        self.assertIsNone(runtime.episode_context.worker.apply_timeout_seconds)
