"""Private native copies must not outlive an unconfirmed cleanup unnoticed."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.tasks.hermes import HermesRuntimeConfiguration, HermesTaskProvider, _NativeRun
from app.tasks.models import TaskError, TaskStatus, WorkerOutcome
from test_second_brain_hermes import context


class HermesPrivateCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="jarvis-hermes-cleanup-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.configuration = HermesRuntimeConfiguration(
            Path("/usr/bin/python3"), self.root / "runs", stop_seconds=.1,
        )
        self.ctx = context()
        self.ctx.task.budget = SimpleNamespace(shutdown_seconds=10, poll_seconds=.1)
        self.gateway = SimpleNamespace(
            run_id="hermes-run:" + uuid4().hex, close=Mock(), images={},
            tool_receipts=[], revalidate=AsyncMock(),
        )

    def run_fixture(self):
        run = _NativeRun(self.configuration, self.ctx, self.gateway)
        (run.home / "synthetic-history.txt").write_text("Synthetic private task content.")
        return run

    def receipt(self, run):
        return json.loads((run.directory / "run.json").read_text())

    async def test_cleanup_error_is_sanitized_persisted_and_retryable(self):
        run = self.run_fixture()
        sibling = self.configuration.work_root / uuid4().hex
        sibling.mkdir(mode=0o700)
        untouched = sibling / "owned-by-another-run.txt"
        untouched.write_text("Keep this run.")
        with patch("app.tasks.hermes.shutil.rmtree", side_effect=PermissionError("PRIVATE CONTENT")):
            with self.assertRaises(TaskError) as raised:
                await run.stop()
        self.assertEqual(raised.exception.code, "hermes_private_cleanup_pending")
        marker = self.receipt(run)
        self.assertEqual(marker["phase"], "exited")
        self.assertEqual(marker["private_home_cleanup"], {"state":"failed", "error_type":"PermissionError"})
        self.assertNotIn("PRIVATE CONTENT", json.dumps(marker))
        self.assertTrue(run.home.exists())
        self.assertFalse(run.finished.is_set())
        self.gateway.close.assert_called_once()
        await run.stop()
        self.assertFalse(run.home.exists())
        self.assertTrue(run.finished.is_set())
        self.assertEqual(self.receipt(run)["private_home_cleanup"], {"state":"complete"})
        self.assertEqual(untouched.read_text(), "Keep this run.")
        self.assertEqual((run.directory / "run.json").stat().st_mode & 0o777, 0o600)

    async def test_absent_home_and_repeated_stop_are_idempotent(self):
        run = self.run_fixture()
        await run.stop()
        before = self.receipt(run)
        await run.stop()
        self.assertEqual(self.receipt(run), before)
        self.assertFalse(run.home.exists())

    async def test_missing_descendant_is_not_mistaken_for_removed_home(self):
        run = self.run_fixture()
        with patch("app.tasks.hermes.shutil.rmtree", side_effect=FileNotFoundError("A child disappeared")):
            with self.assertRaises(TaskError):
                await run.stop()
        self.assertTrue(run.home.exists())
        self.assertEqual(self.receipt(run)["private_home_cleanup"]["state"], "failed")
        await run.stop()

    async def test_home_symlink_is_unlinked_without_touching_its_target(self):
        run = self.run_fixture()
        outside = self.root / "unrelated"
        run.home.rename(outside)
        run.home.symlink_to(outside, target_is_directory=True)
        await run.stop()
        self.assertFalse(run.home.is_symlink())
        self.assertEqual((outside / "synthetic-history.txt").read_text(), "Synthetic private task content.")
        self.assertEqual(self.receipt(run)["private_home_cleanup"]["state"], "complete")

    async def test_run_directory_symlink_cannot_redirect_cleanup_or_receipts(self):
        run = self.run_fixture()
        outside = self.root / "moved-run"
        run.directory.rename(outside)
        original = (outside / "run.json").read_bytes()
        run.directory.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(TaskError):
            await run.stop()
        self.assertEqual((outside / "run.json").read_bytes(), original)
        self.assertTrue((outside / "home" / "synthetic-history.txt").exists())
        self.assertFalse(run.finished.is_set())

    async def test_leftover_receipt_symlink_does_not_overwrite_its_target(self):
        run = self.run_fixture()
        outside = self.root / "untouched.txt"
        outside.write_text("Keep this.")
        (run.directory / "run.next").symlink_to(outside)
        await run.stop()
        self.assertEqual(outside.read_text(), "Keep this.")
        self.assertEqual(self.receipt(run)["private_home_cleanup"]["state"], "complete")

    async def test_home_is_not_removed_until_the_owned_process_has_exited(self):
        run = self.run_fixture()
        entered, release = asyncio.Event(), asyncio.Event()
        async def wait():
            entered.set()
            await release.wait()
            run.process.returncode = 0
            return 0
        run.process = SimpleNamespace(
            returncode=None, stdin=SimpleNamespace(write=Mock(), drain=AsyncMock()),
            wait=AsyncMock(side_effect=wait),
        )
        stopping = asyncio.create_task(run.stop())
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(run.home.exists())
            self.assertFalse(run.finished.is_set())
        finally:
            release.set()
            await stopping
        self.assertFalse(run.home.exists())
        self.assertEqual(self.receipt(run)["exit_code"], 0)

    def provider_fixture(self):
        self.ctx.tasks.start_runtime = AsyncMock()
        prepared = SimpleNamespace(
            manifest=SimpleNamespace(
                owner_id=self.ctx.task.owner_id, task_id=self.ctx.task.id,
                task_revision=self.ctx.claim.task_revision, id=uuid4(),
            ),
            tools=SimpleNamespace(context=self.ctx, catalog=lambda: []),
        )
        upstream = SimpleNamespace(aclose=AsyncMock())
        provider = HermesTaskProvider(
            SimpleNamespace(model=self.ctx.task.model), self.configuration,
            prepare=AsyncMock(return_value=prepared), upstream_factory=lambda: upstream,
        )
        parsed = SimpleNamespace(verified=AsyncMock(return_value=WorkerOutcome(
            TaskStatus.SUCCEEDED, {"answer":"Synthetic answer."},
        )))
        self.enterContext(patch("app.tasks.hermes.TaskModelGateway", return_value=self.gateway))
        self.enterContext(patch.object(_NativeRun, "launch", new=AsyncMock()))
        self.enterContext(patch.object(_NativeRun, "result", new=AsyncMock(return_value={
            "completed":True, "failed":False, "partial":False, "interrupted":False,
            "final_response":"synthetic",
        })))
        self.enterContext(patch("app.tasks.hermes.HermesTaskResult.model_validate_json", return_value=parsed))
        return provider, upstream

    async def test_failed_cleanup_still_closes_upstream_and_preserves_terminal_checkpoint(self):
        provider, upstream = self.provider_fixture()
        with patch("app.tasks.hermes.shutil.rmtree", side_effect=PermissionError("synthetic")):
            with self.assertRaises(TaskError):
                await provider.execute(self.ctx)
        upstream.aclose.assert_awaited_once()
        self.assertFalse(provider._runs)
        checkpoint = self.ctx.checkpoint.await_args.args[0]
        self.assertEqual(checkpoint["phase"], "terminal")
        self.assertEqual(checkpoint["outcome"]["payload"]["answer"], "Synthetic answer.")
        self.ctx.tasks.start_runtime.assert_awaited_once()

    async def test_cancellation_during_shutdown_waits_for_cleanup_and_closes_upstream(self):
        provider, upstream = self.provider_fixture()
        entered, release = asyncio.Event(), asyncio.Event()
        original = _NativeRun.stop
        async def delayed_stop(run):
            entered.set()
            await release.wait()
            await original(run)
        with patch.object(_NativeRun, "stop", new=delayed_stop):
            job = asyncio.create_task(provider.execute(self.ctx))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                job.cancel()
                await asyncio.sleep(0)
                self.assertFalse(job.done())
            finally:
                release.set()
                with self.assertRaises(asyncio.CancelledError):
                    await job
        upstream.aclose.assert_awaited_once()
        self.assertFalse(provider._runs)
        directory = self.configuration.work_root / self.gateway.run_id.removeprefix("hermes-run:")
        self.assertFalse((directory / "home").exists())

    async def test_reconcile_refuses_live_or_unreadable_process_without_removing_home(self):
        run = self.run_fixture()
        self.ctx.claim.provider_handle = run.run_id
        run.marker.update(phase="running", pid=os.getpid())
        run.write_marker()
        provider = HermesTaskProvider(SimpleNamespace(model=self.ctx.task.model), self.configuration)
        for error in (None, PermissionError("synthetic")):
            with self.subTest(error=type(error).__name__), patch("app.tasks.hermes.os.kill", side_effect=error):
                result = await provider.reconcile(self.ctx, cancel=True)
                self.assertEqual(result.state, "unknown")
                self.assertTrue(run.home.exists())
        await run.stop()

    async def test_reconcile_refuses_changed_scope_or_nonobject_receipt(self):
        run = self.run_fixture()
        self.ctx.claim.provider_handle = run.run_id
        provider = HermesTaskProvider(SimpleNamespace(model=self.ctx.task.model), self.configuration)
        original = dict(run.marker)
        for change in (
            {"host_id":"another-host"}, {"owner_digest":"another-owner"},
            {"task_id":str(uuid4())}, {"attempt_id":str(uuid4())},
            {"run_id":"hermes-run:" + uuid4().hex},
        ):
            with self.subTest(change=change):
                (run.directory / "run.json").write_text(json.dumps(original | change))
                self.assertEqual((await provider.reconcile(self.ctx, cancel=False)).state, "unknown")
                self.assertTrue(run.home.exists())
        (run.directory / "run.json").write_text("[]")
        self.assertEqual((await provider.reconcile(self.ctx, cancel=False)).state, "unknown")
        self.assertTrue(run.home.exists())
        run.write_marker()
        await run.stop()
