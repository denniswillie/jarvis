"""Offline refresh lifetime and failure tests; no provider, SQL or native process."""
import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from pydantic import SecretStr

from app.plugins import media_probe
from app.tasks.models import TaskError
from app.tasks.worker import TaskWorkerRole


class MediaRefreshTests(unittest.IsolatedAsyncioTestCase):
    def refresher(self):
        return media_probe.WorkerMediaReadinessRefresh("/explicit/operator.json",
            owner_id="offline-owner", api_key=SecretStr("offline-secret"))

    async def test_failure_retries_only_after_cadence_without_claiming_a_new_lease(self):
        stop = asyncio.Event()
        attempts, waits = [], []

        async def refresh(*args, **kwargs):
            attempts.append((args, kwargs))
            if len(attempts) == 1:
                raise RuntimeError("provider-private-content-offline-secret")
            stop.set()

        async def elapsed(waiter, timeout):
            waiter.close()
            waits.append(timeout)
            raise TimeoutError()

        with patch.object(media_probe, "refresh_observations", refresh), \
                patch.object(media_probe.asyncio, "wait_for", elapsed), \
                self.assertLogs(media_probe.LOGGER, level="WARNING") as logs:
            await self.refresher()(stop)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(waits, [120])
        self.assertEqual(attempts[0][1]["owner_id"], "offline-owner")
        self.assertNotIn("offline-secret", " ".join(logs.output))
        self.assertNotIn("provider-private-content", " ".join(logs.output))

    async def test_cancel_propagates_and_wait_is_stop_interruptible(self):
        stop = asyncio.Event()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def in_flight(*args, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch.object(media_probe, "refresh_observations", in_flight):
            task = asyncio.create_task(self.refresher()(stop))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(cancelled.is_set())
        with patch.object(media_probe, "refresh_observations", AsyncMock()) as probe:
            task = asyncio.create_task(self.refresher()(stop))
            await asyncio.sleep(0)
            stop.set()
            await asyncio.wait_for(task, 1)
        probe.assert_awaited_once()

    async def test_already_stopped_does_not_probe(self):
        stop = asyncio.Event()
        stop.set()
        with patch.object(media_probe, "refresh_observations", AsyncMock()) as probe:
            await self.refresher()(stop)
        probe.assert_not_awaited()

    def test_test_shims_and_missing_runtime_never_enable_refresh(self):
        sandbox = SimpleNamespace(provider=Mock())
        for runtime in (None, SimpleNamespace(observe=AsyncMock()), SimpleNamespace()):
            self.assertIsNone(media_probe.worker_media_refresh(runtime, sandbox, environ={}))
        self.assertEqual(sandbox.provider.mock_calls, [])


class WorkerRefreshLifetimeTests(unittest.IsolatedAsyncioTestCase):
    async def make_role(self, directory, refresh, stop, started):
        repository = SimpleNamespace(pool=object(), configure_scheduler_limits=AsyncMock())
        configuration = SimpleNamespace(host_id="offline", work_root=Path(directory), heartbeat_seconds=1)
        runner = SimpleNamespace(run=AsyncMock(side_effect=lambda: None), stop=AsyncMock())

        async def runner_wait():
            await asyncio.Event().wait()

        runner.run = runner_wait

        async def advertised(**kwargs):
            await asyncio.wait_for(started.wait(), 1)
            stop.set()

        registry = SimpleNamespace(advertise=AsyncMock(side_effect=advertised), withdraw=AsyncMock())
        with patch("app.tasks.worker.TaskRunner", return_value=runner), \
                patch("app.tasks.worker.TaskWorkerRegistry", return_value=registry):
            return TaskWorkerRole(repository, {}, owner_id="offline-owner", configuration=configuration,
                                  readiness_refresh=refresh), registry

    async def test_real_worker_lock_owns_and_awaits_one_refresh_child(self):
        with tempfile.TemporaryDirectory() as directory:
            stop, started, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
            calls = []

            async def refresh(child_stop):
                calls.append(child_stop)
                self.assertTrue((Path(directory) / "worker.json").exists())
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.assertTrue(child_stop.is_set())
                    closed.set()

            role, registry = await self.make_role(directory, refresh, stop, started)
            with patch("app.tasks.worker.ensure_private_runtime_root", return_value=Path(directory)):
                await asyncio.wait_for(role.run(stop), 1)
            self.assertEqual(len(calls), 1)
            self.assertTrue(closed.is_set())
            self.assertFalse((Path(directory) / "worker.json").exists())
            registry.withdraw.assert_awaited_once()
            self.assertFalse(any(t.get_name() == "managed-media-readiness" for t in asyncio.all_tasks()))

    async def test_contended_worker_lock_never_starts_a_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            stop, started = asyncio.Event(), asyncio.Event()
            refresh = AsyncMock()
            role, _ = await self.make_role(directory, refresh, stop, started)
            with patch("app.tasks.worker.ensure_private_runtime_root", return_value=Path(directory)), \
                    patch("app.tasks.worker.fcntl.flock", side_effect=BlockingIOError()):
                with self.assertRaises(TaskError) as caught:
                    await role.run(stop)
            self.assertEqual(caught.exception.code, "task_worker_root_in_use")
            refresh.assert_not_awaited()
