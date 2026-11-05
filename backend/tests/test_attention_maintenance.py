"""Durable attention maintenance does not require a TTS key or voice session."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.attention.service import AttentionService


class AttentionMaintenanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = SimpleNamespace(execute=AsyncMock())
        self.entered = []
        @asynccontextmanager
        async def transaction(owner):
            self.entered.append(owner)
            yield self.connection
        self.repository = SimpleNamespace(tasks=SimpleNamespace(transaction=transaction),
            _recover=AsyncMock(), reconcile_notifications=AsyncMock())
        self.service = AttentionService(self.repository)

    async def asyncTearDown(self):
        await self.service.close()

    async def test_cycle_recovers_quiet_windows_and_notifications_without_voice(self):
        await self.service._maintenance_cycle("owner")
        self.repository._recover.assert_awaited_once_with(self.connection, "owner")
        self.connection.execute.assert_awaited_once_with(
            "select refresh_jarvis_responsibility_quiet($1,clock_timestamp())", "owner")
        self.repository.reconcile_notifications.assert_awaited_once_with("owner")
        self.assertIsNone(self.service._client)
        self.assertIsNone(self.service._task)
        self.assertEqual(self.service.voice_status()["state"], "unavailable")

    async def test_worker_runs_without_provider_and_shutdown_joins_it(self):
        tick = asyncio.Event()
        async def reconcile(owner):
            tick.set()
        self.repository.reconcile_notifications.side_effect = reconcile
        self.service.start_maintenance(owner="owner", poll_seconds=0.01)
        await asyncio.wait_for(tick.wait(), 1)
        self.assertIsNone(self.service.maintenance_issue)
        task = self.service._maintenance_task
        await self.service.close()
        self.assertTrue(task.done())
        self.assertIsNone(self.service._client)

    async def test_recovery_failure_does_not_disable_future_maintenance(self):
        recovered = asyncio.Event()
        self.repository._recover.side_effect = [RuntimeError("fixture"), None]
        async def reconcile(owner):
            recovered.set()
        self.repository.reconcile_notifications.side_effect = reconcile
        self.service.start_maintenance(owner="owner", poll_seconds=0.01)
        await asyncio.wait_for(recovered.wait(), 1)
        self.assertEqual(self.repository._recover.await_count, 2)
        self.assertIsNone(self.service.maintenance_issue)

    async def test_duplicate_start_does_not_create_another_authority(self):
        self.service.start_maintenance(owner="owner")
        with self.assertRaises(RuntimeError):
            self.service.start_maintenance(owner="owner")

    async def test_closed_service_cannot_restart(self):
        await self.service.close()
        with self.assertRaises(RuntimeError):
            self.service.start_maintenance(owner="owner")

    async def test_poll_interval_is_bounded(self):
        for value in (0, -1, 61, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.start_maintenance(owner="owner", poll_seconds=value)
