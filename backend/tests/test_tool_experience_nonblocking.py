"""Offline voice-tool bookkeeping; terminal receipts remain synchronous."""
import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.experience.segmenter import EventSegmentationPolicy, ExperienceEventSegmenter
from app.experience.service import ExperienceEventService
from app.tools.executor import JarvisToolExecutor
from tests.test_tool_executor_durable import _ExecutionRepository, _Registry


class NonblockingToolExperienceTests(IsolatedAsyncioTestCase):
    def service(self):
        repository = SimpleNamespace(get_open_event=AsyncMock(return_value=None))
        service = ExperienceEventService(repository=repository,
            segmenter=ExperienceEventSegmenter(EventSegmentationPolicy(12, 120)))
        self.addAsyncCleanup(service.close)
        return service

    def execution(self, *, status="completed", session_id=None):
        return SimpleNamespace(id=uuid4(), owner_id="fixture-owner",
            session_id=session_id or uuid4(), status=status, result=None, error=None)

    async def test_voice_result_returns_after_receipt_before_slow_attachment(self):
        service = self.service()
        entered, release = asyncio.Event(), asyncio.Event()

        async def attach(execution):
            self.assertEqual(execution.status, "completed")
            entered.set()
            await release.wait()

        service.attach_tool_execution = AsyncMock(side_effect=attach)
        execution = self.execution(status="started")
        repository = _ExecutionRepository(execution, created=True)
        finalize = AsyncMock(wraps=repository.finalize_execution)
        repository.finalize_execution = finalize
        handler = AsyncMock(return_value={"status": "success", "value": "receipt"})
        executor = JarvisToolExecutor(_Registry(handler), execution_repository=repository,
            experience_event_service=service, owner_id=execution.owner_id,
            session_id=execution.session_id, defer_experience_attachment=True)
        try:
            result = await asyncio.wait_for(executor.execute("mutate", {},
                provider_call_id="fixture-call"), 0.5)
            self.assertEqual(result, {"status": "success", "value": "receipt"})
            finalize.assert_awaited_once()
            handler.assert_awaited_once()
            await asyncio.wait_for(entered.wait(), 0.5)
            self.assertTrue(await service._has_active_work((execution.owner_id, execution.session_id)))
            self.assertFalse(release.is_set())
        finally:
            release.set()
        await service.close()
        self.assertEqual(service._active_counts, {})
        self.assertEqual(service._tool_attachment_tasks, {})

    async def test_failed_finalization_never_queues_auxiliary_work(self):
        service = self.service()
        service.defer_tool_execution = AsyncMock()
        execution = self.execution(status="started")
        handler = AsyncMock(return_value={"status": "success"})
        executor = JarvisToolExecutor(_Registry(handler),
            execution_repository=_ExecutionRepository(execution, created=True, fail_finalize=True),
            experience_event_service=service, owner_id=execution.owner_id,
            session_id=execution.session_id, defer_experience_attachment=True)
        result = await executor.execute("mutate", {}, provider_call_id="fixture-call")
        self.assertEqual(result["error"]["code"], "tool_execution_persistence_failed")
        service.defer_tool_execution.assert_not_awaited()
        handler.assert_awaited_once()

    async def test_reservation_blocks_session_close_before_task_starts(self):
        service = self.service()
        release = asyncio.Event()
        service.attach_tool_execution = AsyncMock(side_effect=lambda _: release.wait())

        async def attach(_):
            await release.wait()

        service.attach_tool_execution = AsyncMock(side_effect=attach)
        execution = self.execution()
        await service.defer_tool_execution(execution)
        service.schedule_session_close(owner_id=execution.owner_id,
            session_id=execution.session_id, closed_at=datetime.now(UTC))
        await asyncio.sleep(0)
        service.repository.get_open_event.assert_not_awaited()
        release.set()
        await service.close()
        service.repository.get_open_event.assert_awaited_once()

    async def test_queue_is_bounded_deduplicates_pending_and_rejects_nonterminal(self):
        service = self.service()
        release = asyncio.Event()

        async def attach(_):
            await release.wait()

        service.attach_tool_execution = AsyncMock(side_effect=attach)
        execution = self.execution()
        with patch("app.experience.service.MAX_BACKGROUND_TOOL_ATTACHMENTS", 1):
            self.assertTrue(await service.defer_tool_execution(execution))
            self.assertTrue(await service.defer_tool_execution(execution))
            self.assertFalse(await service.defer_tool_execution(self.execution()))
            self.assertFalse(await service.defer_tool_execution(self.execution(status="started")))
        release.set()
        await service.close()
        service.attach_tool_execution.assert_awaited_once_with(execution)
        self.assertFalse(await service.defer_tool_execution(self.execution()))

    async def test_attachment_error_releases_activity_without_replaying_handler(self):
        service = self.service()
        service.attach_tool_execution = AsyncMock(side_effect=RuntimeError("private payload"))
        execution = self.execution()
        with self.assertLogs("app.experience.service", "WARNING") as logs:
            await service.defer_tool_execution(execution)
            await service.close()
        self.assertNotIn("private payload", "\n".join(logs.output))
        service.attach_tool_execution.assert_awaited_once()
        self.assertEqual(service._active_counts, {})
        self.assertEqual(service._tool_attachment_tasks, {})

    async def test_three_cancelled_close_callers_join_one_owned_drain(self):
        service = self.service()
        entered, release = asyncio.Event(), asyncio.Event()

        async def attach(_):
            entered.set()
            await release.wait()

        service.attach_tool_execution = AsyncMock(side_effect=attach)
        await service.defer_tool_execution(self.execution())
        await asyncio.wait_for(entered.wait(), 0.5)
        callers = [asyncio.create_task(service.close()) for _ in range(3)]
        await asyncio.sleep(0)
        for caller in callers:
            caller.cancel()
        await asyncio.sleep(0)
        self.assertTrue(all(not caller.done() for caller in callers))
        self.assertFalse(service._shutdown_task.done())
        release.set()
        results = await asyncio.gather(*callers, return_exceptions=True)
        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
        await service.close()
        self.assertTrue(service._shutdown_task.done())
        service.attach_tool_execution.assert_awaited_once()
        self.assertEqual(service._active_counts, {})

    async def test_shutdown_deadline_cancels_and_joins_owned_attachment(self):
        service = self.service()
        cancelled = asyncio.Event()

        async def attach(_):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        service.attach_tool_execution = AsyncMock(side_effect=attach)
        with patch("app.experience.service.SESSION_CLOSE_DRAIN_TIMEOUT_SECONDS", 0.02):
            await service.defer_tool_execution(self.execution())
            await asyncio.wait_for(service.close(), 0.5)
        self.assertTrue(cancelled.is_set())
        self.assertEqual(service._active_counts, {})
        self.assertEqual(service._tool_attachment_tasks, {})
