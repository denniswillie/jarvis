"""Real private files and asyncio cancellation; no SQL, child or model calls."""
import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import urlencode
from uuid import uuid4

from app.tasks.models import TaskClaim
from app.tasks.runner import WorkerContext
from evals.responsive_voice import native_task_release_control as release
from evals.responsive_voice import native_task_worker_entry as entry
from evals.responsive_voice.native_task_result_hold import ResultHoldProvider, ResultHoldTimeout


class NativeTaskReleaseControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="offline-result-control-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.root.chmod(0o700)
        run = uuid4()
        self.config = release.create_control(self.root, run_id=run,
            owner_id=f"direct-image-eval-second-brain-{run.hex}-a",
            database_schema=f"jarvis_sb_eval_{run.hex}_0", worker_host_id=f"native-task-{run.hex}-0123456789ab",
            model="fixture-model", expires_at=datetime.now(UTC) + timedelta(minutes=2),
            max_hold_seconds=1.0, poll_seconds=.005)
        (self.root / "native").mkdir(mode=0o700)
        self.worker_id = "frontier-" + uuid4().hex
        marker = {"format": "jarvis.task_worker.v1", "worker_id": self.worker_id,
            "owner_digest": hashlib.sha256(self.config.scope.owner_id.encode()).hexdigest(),
            "host_id": self.config.scope.worker_host_id, "pid": os.getpid()}
        release.write_private(self.root / "native", "worker.json", json.dumps(marker).encode(), exclusive=True)
        query = urlencode({"search_path": f'"{self.config.scope.database_schema}",pg_catalog,extensions'})
        release.write_private(self.root, "db.secret", ("postgresql://fixture:PRIVATE@127.0.0.1/db?" + query).encode(), exclusive=True)
        self.env = {"JARVIS_NATIVE_RESULT_HOLD_CONFIG": str(self.config.config_path),
            "JARVIS_OWNER_ID": self.config.scope.owner_id, "SECOND_BRAIN_WORKER_HOST_ID": self.config.scope.worker_host_id,
            "SECOND_BRAIN_FRONTIER_MODEL": self.config.scope.model,
            "SECOND_BRAIN_HERMES_WORK_ROOT": str(self.root / "native"), "JARVIS_FRONTIER_DB_URL_FILE": str(self.root / "db.secret")}

    def context(self):
        attempt = uuid4()
        task = SimpleNamespace(id=uuid4(), owner_id=self.config.scope.owner_id, session_id=uuid4(),
            model=self.config.scope.model, backend_kind="frontier", current_attempt_id=attempt, fence=1, revision=1)
        claim = TaskClaim(task=task, attempt_id=attempt, task_revision=1, fence=1,
            lease_until=datetime.now(UTC) + timedelta(seconds=10), provider_state="live",
            provider_handle=None, checkpoint_payload_id=None)
        return WorkerContext(claim, Mock(), Mock(), Mock())

    async def started(self, context=None, config=None):
        context, config = context or self.context(), config or self.config
        result = object()
        provider = SimpleNamespace(model=config.scope.model, prepare=object(), reconcile=AsyncMock(), execute=AsyncMock(return_value=result))
        wrapper = ResultHoldProvider(provider, release.BoundReleaseWaiter(config), max_hold_seconds=config.max_hold_seconds)
        job = asyncio.create_task(wrapper.execute(context))
        async def cleanup():
            if not job.done():
                job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        self.addAsyncCleanup(cleanup)
        async with asyncio.timeout(.5):
            while True:
                if job.done():
                    await job
                try:
                    held = release.read_hold(config, attempt_id=context.claim.attempt_id, fence=context.claim.fence)
                    break
                except FileNotFoundError:
                    await asyncio.sleep(.001)
        return context, provider, wrapper, result, job, held

    def command(self, held, **changes):
        now = datetime.now(UTC)
        return release.ReleaseCommand(scope=held.scope, identity=held.identity, hold_id=held.hold_id,
            command_id=uuid4(), issued_at=now, expires_at=min(now + timedelta(seconds=.2), held.expires_at)).model_copy(update=changes)

    def raw_command(self, held, command):
        release.write_private(self.config.control_root, f"release-{held.hold_id.hex}.json", command.model_dump_json().encode())

    async def test_real_return_held_until_exact_release_and_metadata_only(self):
        context, provider, wrapper, result, job, held = await self.started()
        self.assertFalse(job.done())
        self.assertEqual(held.phase, "entered")
        self.assertEqual(held.identity.attempt_id, context.claim.attempt_id)
        self.assertEqual(held.identity.worker_id, self.worker_id)
        self.assertIs(wrapper.prepare, provider.prepare)
        self.assertIs(wrapper.reconcile, provider.reconcile)
        command = release.publish_release(self.config, held, command_id=uuid4())
        self.assertIs(await asyncio.wait_for(job, .5), result)
        provider.execute.assert_awaited_once_with(context)
        final = release.read_hold(self.config, attempt_id=context.claim.attempt_id, fence=1)
        self.assertEqual(final.phase, "released")
        self.assertEqual(final.command_id, command.command_id)
        self.assertNotIn("PRIVATE", final.model_dump_json())
        self.assertEqual(set(final.model_dump()), {"format", "scope", "identity", "hold_id", "phase",
            "entered_at", "expires_at", "updated_at", "command_id", "rejected_commands", "last_rejection"})
        context.tasks.assert_not_called()

    async def test_foreign_old_expired_and_observer_receipts_cannot_release(self):
        cases = ("scope", "task", "attempt", "fence", "worker", "session", "hold", "expired", "early", "observer")
        for case in cases:
            with self.subTest(case=case):
                context, provider, wrapper, result, job, held = await self.started()
                command = self.command(held)
                if case == "scope":
                    command = command.model_copy(update={"scope": held.scope.model_copy(update={"control_id": uuid4()})})
                elif case in {"task", "attempt", "fence", "worker", "session"}:
                    key, value = {"task": ("task_id", uuid4()), "attempt": ("attempt_id", uuid4()),
                        "fence": ("fence", 2), "worker": ("worker_id", "frontier-" + uuid4().hex),
                        "session": ("session_id", uuid4())}[case]
                    command = command.model_copy(update={"identity": held.identity.model_copy(update={key: value})})
                elif case == "hold":
                    command = command.model_copy(update={"hold_id": uuid4()})
                elif case == "expired":
                    command = command.model_copy(update={"issued_at": held.entered_at,
                        "expires_at": held.entered_at + timedelta(microseconds=1)})
                elif case == "early":
                    command = command.model_copy(update={"issued_at": held.entered_at - timedelta(seconds=1)})
                if case == "observer":
                    release.write_private(self.config.control_root, f"release-{held.hold_id.hex}.json",
                                          b'{"format":"jarvis.native_task_barrier.v1","status":"eligible"}')
                else:
                    self.raw_command(held, command)
                await asyncio.sleep(.025)
                self.assertFalse(job.done())
                release.publish_release(self.config, held, command_id=uuid4())
                self.assertIs(await asyncio.wait_for(job, .5), result)
                final = release.read_hold(self.config, attempt_id=context.claim.attempt_id, fence=1)
                self.assertEqual(final.rejected_commands, 1)

    async def test_command_id_consumption_persists_across_distinct_holds(self):
        first = await self.started()
        command_id = uuid4()
        release.publish_release(self.config, first[-1], command_id=command_id)
        await first[-2]
        second = await self.started()
        release.publish_release(self.config, second[-1], command_id=command_id)
        await asyncio.sleep(.025)
        self.assertFalse(second[-2].done())
        release.publish_release(self.config, second[-1], command_id=uuid4())
        await second[-2]
        final = release.read_hold(self.config, attempt_id=second[0].claim.attempt_id, fence=1)
        self.assertEqual(final.last_rejection, "release_replayed")

    async def test_cancellation_propagates_and_never_releases_original_result(self):
        context, provider, wrapper, result, job, held = await self.started()
        job.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await job
        final = release.read_hold(self.config, attempt_id=context.claim.attempt_id, fence=1)
        self.assertEqual(final.phase, "cancelled")
        self.assertIsNone(final.command_id)
        with self.assertRaises(release.ReleaseControlError):
            release.publish_release(self.config, held, command_id=uuid4())
        provider.execute.assert_awaited_once()

    async def test_time_limit_records_timeout_without_result_or_lease_changes(self):
        config = self.config.model_copy(update={"max_hold_seconds": .15})
        context, provider, wrapper, result, job, held = await self.started(config=config)
        original_lease = context.claim.lease_until
        with self.assertRaises(ResultHoldTimeout):
            await asyncio.wait_for(job, .5)
        final = release.read_hold(config, attempt_id=context.claim.attempt_id, fence=1)
        self.assertEqual(final.phase, "timeout")
        self.assertEqual(context.claim.lease_until, original_lease)
        self.assertEqual(context.tasks.mock_calls, [])

    async def test_claim_and_actual_worker_marker_mismatch_fail_before_entered(self):
        for case in ("owner", "revision", "attempt", "fence", "recovering", "marker"):
            with self.subTest(case=case):
                context = self.context()
                if case == "owner": context.task.owner_id = "unrelated-owner"
                elif case == "revision": context.task.revision = 2
                elif case == "attempt": context.task.current_attempt_id = uuid4()
                elif case == "fence": context.task.fence = 2
                elif case == "recovering": object.__setattr__(context.claim, "recovering", True)
                else:
                    release.write_private(self.root / "native", "worker.json", b'{"format":"wrong"}')
                with self.assertRaises(release.ReleaseControlError):
                    await release.BoundReleaseWaiter(self.config)(context)
                self.assertFalse((self.config.control_root / f"hold-{context.claim.attempt_id.hex}-1.json").exists())

    async def test_entry_uses_existing_run_command_instance_transform_and_actual_context(self):
        async def run(command, *, provider_transform):
            self.assertEqual(command, "run")
            provider = SimpleNamespace(model=self.config.scope.model, execute=AsyncMock())
            wrapped = provider_transform(provider)
            self.assertIsInstance(wrapped, ResultHoldProvider)
            self.assertIs(wrapped._provider, provider)
            self.assertIsInstance(wrapped._wait_for_release, release.BoundReleaseWaiter)
            self.assertEqual(wrapped._wait_for_release.config, self.config)
            return 0
        with patch.object(entry, "run_command", new=AsyncMock(side_effect=run)) as run_command:
            self.assertEqual(await entry.run_entry(environ=self.env), 0)
            run_command.assert_awaited_once()

    async def test_entry_rejects_missing_foreign_or_wrong_schema_configuration_before_worker(self):
        for key in ("JARVIS_NATIVE_RESULT_HOLD_CONFIG", "JARVIS_OWNER_ID", "SECOND_BRAIN_WORKER_HOST_ID",
                    "SECOND_BRAIN_FRONTIER_MODEL", "SECOND_BRAIN_HERMES_WORK_ROOT", "JARVIS_FRONTIER_DB_URL_FILE"):
            with self.subTest(key=key):
                env = {**self.env, key: "wrong"}
                with patch.object(entry, "run_command", new_callable=AsyncMock) as run:
                    with self.assertRaises(ValueError):
                        await entry.run_entry(environ=env)
                    run.assert_not_awaited()
        release.write_private(self.root, "db.secret", b"postgresql://fixture:PRIVATE@127.0.0.1/db?search_path=public")
        with self.assertRaises(release.ReleaseControlError):
            entry.configured_control(self.env)

    def test_strict_release_schema_and_private_files(self):
        for payload in (b'{"format":1,"format":2}', b'null', b'{}', b'x' * (release.MAX_BYTES + 1)):
            with self.subTest(payload_size=len(payload)):
                with self.assertRaises(release.ReleaseControlError):
                    release.decode(release.ReleaseCommand, payload)
        self.assertEqual(self.config.control_root.stat().st_mode & 0o777, 0o700)
        self.assertEqual(self.config.config_path.stat().st_mode & 0o777, 0o600)
        outside = self.root / "outside.json"
        release.write_private(self.root, outside.name, b"{}", exclusive=True)
        link = self.config.control_root / "symlink.json"
        link.symlink_to(outside)
        with self.assertRaises(OSError):
            release.read_private(self.config.control_root, link.name)
