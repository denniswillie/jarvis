"""Offline supervisor qualification: fake process and SQL, never native/provider work."""
import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from evals.responsive_voice import native_task_worker as worker


def private_file(path, value):
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


class FakeConnection:
    def __init__(self, pool):
        self.pool = pool

    @asynccontextmanager
    async def transaction(self, *, readonly):
        assert readonly is True
        self.pool.read_transactions += 1
        yield self

    async def fetchval(self, query, *args):
        pool = self.pool
        if query == "select current_schema()":
            return pool.schema
        if query == "select current_schemas(false)":
            return [pool.schema, "pg_catalog", "extensions"]
        assert "select exists(select 1 from jarvis_task_workers" in query
        pool.registry_reads.append(args)
        owner, kind, model, identity, capabilities = args
        return (pool.state == "active" and owner == pool.owner and kind == "frontier"
                and model == pool.model and identity == pool.worker_id and capabilities == [])

    async def fetchrow(self, query, *args):
        assert "jarvis_memory_banks" in query
        assert args == (self.pool.owner, self.pool.area)
        return {"bank_name": self.pool.bank, "status": "active"}

    async def fetch(self, query, *args):
        assert "jarvis_task_workers" in query
        assert args == (self.pool.owner, self.pool.host_id)
        if self.pool.state == "absent":
            return []
        return [{"worker_id": self.pool.worker_id, "model": self.pool.model,
                 "draining": self.pool.state == "withdrawn", "unexpired": self.pool.state != "expired"}]


class FakePool:
    def __init__(self, scope, model):
        self.owner, self.schema, self.bank = scope["owner_id"], scope["database_schema"], scope["memory_bank_id"]
        from uuid import UUID
        self.area = UUID(scope["area_id"])
        self.model, self.host_id, self.worker_id = model, None, None
        self.state = "absent"
        self.registry_reads, self.read_transactions = [], 0
        self.closed, self.terminated = False, False

    @asynccontextmanager
    async def acquire(self):
        yield FakeConnection(self)

    async def close(self):
        self.closed = True

    def terminate(self):
        self.terminated = True


class FakeProcess:
    def __init__(self, pool):
        self.pool = pool
        self.pid, self.returncode = 46001, None
        self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
        self.exited = asyncio.Event()

    async def wait(self):
        await self.exited.wait()
        return self.returncode

    def finish(self, code=0, *, state="withdrawn"):
        if self.returncode is None:
            self.returncode = code
            self.pool.state = state
            self.stdout.feed_eof()
            self.stderr.feed_eof()
            self.exited.set()


class NativeTaskWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="offline-native-worker-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = uuid4().hex
        self.owner = f"direct-image-eval-second-brain-{self.run}-a"
        self.scope = {"run_id": self.run, "owner_id": self.owner,
            "database_schema": f"jarvis_sb_eval_{self.run}_0", "area_id": str(uuid4()),
            "memory_bank_id": "jarvis_" + uuid4().hex,
            "expires_at": (datetime.now(UTC) + timedelta(minutes=3)).isoformat()}
        fixture = {"format": "jarvis.second_brain.fixture.v1", "run_id": self.run,
            "owners": [self.owner, f"direct-image-eval-second-brain-{self.run}-b"],
            "schemas": [self.scope["database_schema"]], "status": "prepared"}
        private_file(self.root / "database.json", json.dumps(fixture))
        private_file(self.root / "scope.json", json.dumps(self.scope))
        private_file(self.root / "openai-source.key", "fixture-private-openai-key")
        private_file(self.root / "hindsight-source.key", "fixture-private-hindsight-key")
        self.resources = {"format": "jarvis.frontend_host.resources.v1", "status": "prepared",
            "capability_profile": "native_realtime_media", **self.scope, "bank_name": self.scope["memory_bank_id"],
            "fixture_manifest": str(self.root / "database.json"), "scope_file": str(self.root / "scope.json"),
            "hindsight_url": "http://127.0.0.1:18888",
            "hindsight_api_key_file": str(self.root / "hindsight-source.key")}
        self.options = worker.NativeTaskWorkerOptions(model="fixture-frontier", artifact_dir=self.root,
            api_key_file=self.root / "openai-source.key", hermes_python=Path(sys.executable),
            readiness_timeout_seconds=.15, poll_seconds=.005, stop_timeout_seconds=.03,
            kill_timeout_seconds=.03, withdrawal_timeout_seconds=.03, io_timeout_seconds=.1, pipe_cap_bytes=32)
        self.pool = FakePool(self.scope, self.options.model)
        self.process = None
        self.instance = None
        self.launched = asyncio.Event()
        self.signals = []
        self.advertise, self.marker_valid, self.ignore_term = True, True, False
        self._patches = [patch.object(worker, "_open_pool", new_callable=AsyncMock, return_value=self.pool),
                        patch.object(worker, "_launch", new_callable=AsyncMock, side_effect=self.launch),
                        patch.object(worker.os, "getpgid", side_effect=lambda pid: pid),
                        patch.object(worker.os, "killpg", side_effect=self.signal)]
        self.open_pool, self.launch_mock, self.getpgid, self.killpg = [p.start() for p in self._patches]
        for item in self._patches:
            self.addCleanup(item.stop)
        self.dsn = "postgresql://fixture:fixture-db-secret@127.0.0.1/fixture"

    async def asyncTearDown(self):
        if self.instance is not None:
            await self.instance.close()
        if self.process is not None and self.process.returncode is None:
            self.process.finish()
            await asyncio.sleep(0)

    async def launch(self, *args, **kwargs):
        self.launch_arguments, self.launch_options = args, kwargs
        env = kwargs["env"]
        self.pool.host_id = env["SECOND_BRAIN_WORKER_HOST_ID"]
        self.pool.worker_id = "frontier-" + uuid4().hex
        self.pool.state = "active" if self.advertise else "absent"
        self.process = FakeProcess(self.pool)
        native = Path(env["SECOND_BRAIN_HERMES_WORK_ROOT"])
        native.mkdir(mode=0o700)
        private_file(native / "worker.json", json.dumps({"format": "jarvis.task_worker.v1",
            "worker_id": self.pool.worker_id, "host_id": self.pool.host_id if self.marker_valid else "foreign-host",
            "owner_digest": hashlib.sha256(self.owner.encode()).hexdigest(), "pid": self.process.pid}))
        self.process.stdout.feed_data(b"fixture-private-openai-key" * 100)
        self.process.stderr.feed_data(b"fixture-private-hindsight-key" * 100)
        self.launched.set()
        return self.process

    def signal(self, pid, sig):
        self.assertEqual(pid, self.process.pid)
        self.signals.append(sig)
        if sig == signal.SIGTERM and not self.ignore_term:
            self.process.finish()
        elif sig == signal.SIGKILL:
            self.process.finish(-9, state="expired")

    async def start(self):
        self.instance = await worker.start(self.options, self.resources, self.dsn)
        return self.instance

    async def test_real_registry_predicate_exact_scope_cli_and_private_files(self):
        with patch.dict(os.environ, {"JARVIS_OWNER_ID": "personal-owner", "OPENAI_API_KEY": "global-secret",
                                     "SECOND_BRAIN_SANDBOX_ENABLED": "true"}):
            instance = await self.start()
        receipt = instance.readiness_receipt
        self.assertEqual(receipt["status"], "ready")
        self.assertEqual(receipt["owner_id"], self.owner)
        self.assertEqual(receipt["worker_id"], self.pool.worker_id)
        self.assertTrue(receipt["registry_ready"])
        self.assertGreater(len(self.pool.registry_reads), 0)
        self.assertTrue(all(args[3] == self.pool.worker_id for args in self.pool.registry_reads))
        self.assertEqual(self.launch_arguments, (str(self.options.worker_python), "-m", "app.tasks.worker", "run"))
        self.assertTrue(self.launch_options["start_new_session"])
        env = self.launch_options["env"]
        self.assertEqual(env["JARVIS_OWNER_ID"], self.owner)
        self.assertEqual(env["SECOND_BRAIN_KNOWLEDGE_AREA_ID"], self.scope["area_id"])
        self.assertEqual(env["SECOND_BRAIN_HINDSIGHT_URL"], self.resources["hindsight_url"])
        self.assertNotIn("SUPABASE_DB_URL", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        for name in ("SANDBOX", "ARTIFACTS", "PUSH", "MAIL"):
            self.assertEqual(env[f"SECOND_BRAIN_{name}_ENABLED"], "false")
        self.assertEqual(env["SECOND_BRAIN_HERMES_PYTHON"], str(self.options.hermes_python))
        database_secret = Path(env["JARVIS_FRONTIER_DB_URL_FILE"])
        scoped = database_secret.read_text()
        self.assertEqual(scoped, self.open_pool.await_args.args[0])
        self.assertEqual(parse_qs(urlsplit(scoped).query)["search_path"],
                         [f'"{self.scope["database_schema"]}",pg_catalog,extensions'])
        for path in instance._secrets:
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(instance.root.stat().st_mode & 0o777, 0o700)
        closed = await instance.close()
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["registry_state"], "withdrawn")
        self.assertTrue(closed["worker_joined"] and closed["readiness_inactive"] and closed["secrets_removed"])
        self.assertTrue(self.pool.closed)
        self.assertEqual(self.signals, [signal.SIGTERM])
        self.assertTrue(all(not path.exists() for path in instance._secrets))
        self.assertEqual((self.root / "openai-source.key").read_text(), "fixture-private-openai-key")
        self.assertTrue((self.root / "hindsight-source.key").exists())
        self.assertTrue((self.root / "scope.json").exists())

    async def test_pipe_drains_are_capped_complete_and_never_export_raw_output(self):
        instance = await self.start()
        receipt = await instance.close()
        self.assertTrue(receipt["pipes_complete"])
        for pipe in receipt["pipes"]:
            self.assertGreater(pipe["bytes_seen"], 32)
            self.assertEqual(pipe["retained_bytes"], 32)
            self.assertTrue(pipe["truncated"])
        manifest = instance.resource_manifest.read_text()
        for secret in ("fixture-private-openai-key", "fixture-private-hindsight-key", "fixture-db-secret"):
            self.assertNotIn(secret, manifest)
        self.assertTrue(all(len(drain.tail) == 0 for drain in instance._drains))

    async def test_cli_failure_before_marker_survives_failed_start_cleanup(self):
        failure = {"ready": False, "error_type": "ValidationError", "code": "task_worker_startup_failed"}
        await self.assert_cli_failure_survives_cleanup(failure)

    async def test_cli_failure_frame_survives_failed_start_cleanup(self):
        failure = {"ready": False, "error_type": "ValueError", "code": "task_worker_startup_failed",
                   "diagnostic": {"version": 1, "stage": "knowledge_ingestion",
                       "frame": {"source": "ingestion.py", "function": "__init__", "line": 14},
                       "basis": "exception_origin", "truncated": False}}
        await self.assert_cli_failure_survives_cleanup(failure)

    async def assert_cli_failure_survives_cleanup(self, failure):

        async def failed_launch(*args, **kwargs):
            self.pool.host_id = kwargs["env"]["SECOND_BRAIN_WORKER_HOST_ID"]
            self.pool.state = "absent"
            self.process = FakeProcess(self.pool)
            self.process.stdout.feed_data((json.dumps(failure) + "\n").encode())
            self.process.finish(1, state="absent")
            return self.process

        self.launch_mock.side_effect = failed_launch
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        error = caught.exception
        self.assertEqual(error.code, "fixture_worker_exited_before_ready")
        self.assertEqual(error.receipt["status"], "failed")
        self.assertEqual(error.receipt["returncode"], 1)
        self.assertTrue(error.receipt["worker_joined"] and error.receipt["pipes_complete"])
        self.assertEqual(error.receipt["pipes"][0]["startup_failure"], failure)
        self.assertIsNone(error.receipt["pipes"][1]["startup_failure"])
        persisted = json.loads(error.resource_manifest.read_text())
        self.assertIsNone(persisted["readiness"])
        self.assertEqual(persisted["close"]["pipes"][0]["startup_failure"], failure)
        self.assertEqual(persisted["close"]["status"], "failed")

    async def test_close_is_idempotent_and_concurrent_callers_join_one_stop(self):
        instance = await self.start()
        first, second = await asyncio.gather(instance.close(), instance.close())
        self.assertEqual(first, second)
        self.assertEqual(await instance.close(), first)
        self.assertEqual(self.signals, [signal.SIGTERM])

    async def test_missing_advertisement_never_declares_ready_and_start_failure_joins(self):
        self.advertise = False
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        self.assertEqual(caught.exception.code, "fixture_worker_readiness_timeout")
        self.assertTrue(caught.exception.receipt["worker_joined"])
        self.assertEqual(caught.exception.receipt["registry_state"], "withdrawn")
        self.assertEqual(self.process.returncode, 0)

    async def test_cancelled_start_finishes_cleanup_before_reraising(self):
        self.advertise = False
        pending = asyncio.create_task(worker.start(self.options, self.resources, self.dsn))
        await self.launched.wait()
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError) as caught:
            await pending
        self.assertTrue(caught.exception.receipt["worker_joined"])
        self.assertTrue(caught.exception.receipt["secrets_removed"])
        self.assertIsNotNone(caught.exception.resource_manifest)
        self.assertEqual(self.process.returncode, 0)

    async def test_foreign_marker_cannot_use_an_existing_owner_readiness_lease(self):
        self.marker_valid = False
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        self.assertEqual(caught.exception.code, "fixture_worker_marker_mismatch")
        self.assertTrue(caught.exception.receipt["worker_joined"])

    async def test_manifest_owner_bank_model_and_public_dsn_fail_before_spawn(self):
        for change in ({"owner_id": "personal-owner"}, {"bank_name": "jarvis_" + uuid4().hex},
                       {"worker_model": "different-model"}, {"status": "cleaned"}):
            with self.subTest(change=tuple(change)):
                with self.assertRaises(worker.NativeTaskWorkerError):
                    await worker.start(self.options, self.resources | change, self.dsn)
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await worker.start(self.options, self.resources, self.dsn + "?search_path=public")
        self.assertEqual(caught.exception.code, "fixture_database_path_mismatch")
        self.launch_mock.assert_not_awaited()
        self.open_pool.assert_not_awaited()

    async def test_actual_database_binding_is_checked_before_any_child(self):
        self.pool.bank = "jarvis_" + uuid4().hex
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        self.assertEqual(caught.exception.code, "fixture_actual_bank_mismatch")
        self.assertTrue(caught.exception.receipt["worker_joined"])
        self.assertTrue(caught.exception.receipt["secrets_removed"])
        self.launch_mock.assert_not_awaited()

    async def test_short_or_expired_host_lease_never_spawns(self):
        self.scope["expires_at"] = datetime.now(UTC).isoformat()
        private_file(self.root / "scope.json", json.dumps(self.scope))
        self.resources["expires_at"] = self.scope["expires_at"]
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        self.assertEqual(caught.exception.code, "fixture_worker_lease_insufficient")
        self.launch_mock.assert_not_awaited()

    async def test_host_lease_stops_worker_without_a_caller_close(self):
        self.scope["expires_at"] = (datetime.now(UTC) + timedelta(seconds=2.6)).isoformat()
        private_file(self.root / "scope.json", json.dumps(self.scope))
        self.resources["expires_at"] = self.scope["expires_at"]
        instance = await self.start()
        await asyncio.wait_for(self.process.exited.wait(), 3)
        receipt = await instance.close()
        self.assertEqual(receipt["reason"], "host_lease")
        self.assertEqual(receipt["status"], "closed")
        self.assertLess(datetime.now(UTC), datetime.fromisoformat(self.scope["expires_at"]))

    async def test_forced_kill_is_joined_but_not_a_clean_native_shutdown(self):
        self.ignore_term = True
        instance = await self.start()
        receipt = await instance.close()
        self.assertEqual(self.signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertTrue(receipt["worker_joined"])
        self.assertEqual(receipt["registry_state"], "expired")
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["native_cleanup"], "unverified_after_forced_stop")

    async def test_foreign_process_group_is_never_signalled_or_called_joined(self):
        instance = await self.start()
        self.getpgid.side_effect = lambda pid: pid + 1
        receipt = await instance.close()
        self.assertEqual(receipt["status"], "failed")
        self.assertFalse(receipt["worker_joined"])
        self.assertFalse(receipt["secrets_removed"])
        self.killpg.assert_not_called()

    async def test_public_secret_permissions_are_rejected_without_copy_or_spawn(self):
        self.options.api_key_file.chmod(0o644)
        with self.assertRaises(worker.NativeTaskWorkerError) as caught:
            await self.start()
        self.assertEqual(caught.exception.code, "fixture_private_permissions_required")
        self.launch_mock.assert_not_awaited()

    async def test_host_environment_is_explicit_and_has_no_credentials_or_owner_defaults(self):
        env = worker.expected_host_environment(self.options)
        self.assertEqual(env["SECOND_BRAIN_FRONTIER_MODEL"], "fixture-frontier")
        self.assertEqual(env["SECOND_BRAIN_FRONTIER_REASONING_EFFORT"], "low")
        self.assertEqual(env["SECOND_BRAIN_ATTENTION_VOICE_ENABLED"], "true")
        self.assertEqual(env["REMINDER_SCHEDULER_ENABLED"], "true")
        self.assertEqual(env["SECOND_BRAIN_GLOBAL_CONCURRENCY"], "1")
        self.assertNotIn("JARVIS_OWNER_ID", env)
        self.assertNotIn("SUPABASE_DB_URL", env)
        self.assertNotIn("OPENAI_API_KEY", env)


class PipeStartupFailureTests(unittest.IsolatedAsyncioTestCase):
    def diagnostic_failure(self):
        return {"ready": False, "error_type": "ValueError", "code": "task_worker_startup_failed",
                "diagnostic": {"version": 1, "stage": "knowledge_ingestion",
                    "frame": {"source": "ingestion.py", "function": "__init__", "line": 14},
                    "basis": "exception_origin", "truncated": False}}

    async def drain(self, chunks, *, cap=8):
        pieces = iter(chunks)

        class Stream:
            async def read(self, limit):
                value = next(pieces, b"")
                assert len(value) <= limit
                return value

        drain = worker._PipeDrain(Stream(), cap)
        await drain.run()
        self.assertTrue(drain.eof)
        self.assertIsNone(drain.error)
        self.assertLessEqual(len(drain.tail), cap)
        self.assertEqual(len(drain._startup_line), 0)
        return drain

    async def test_recognized_fragmented_receipts_survive_tail_clear(self):
        for error_type, code in (
            ("ValidationError", "task_worker_startup_failed"),
            ("ValueError", "task_worker_startup_failed"),
            ("TaskError", "task_worker_not_configured"),
        ):
            with self.subTest(error_type=error_type):
                expected = {"ready": False, "error_type": error_type, "code": code}
                data = (json.dumps(expected) + "\n").encode()
                drain = await self.drain([data[index:index+1] for index in range(len(data))])
                self.assertTrue(drain.receipt()["truncated"])
                self.assertEqual(drain.receipt()["startup_failure"], expected)
                drain.tail.clear()
                self.assertEqual(drain.receipt()["startup_failure"], expected)
                exposed = drain.receipt()
                exposed["startup_failure"]["code"] = "changed_by_caller"
                self.assertEqual(drain.receipt()["startup_failure"], expected)

    async def test_unknown_strings_are_redacted_not_treated_as_safe_identifiers(self):
        private = "PRIVATE_SECRET_Bearer_sk_secret_transcript"
        for error_type, code, expected_type, expected_code in (
            (private, private, "unrecognized_error_type", "unrecognized_code"),
            ("TaskError", private, "TaskError", "unrecognized_code"),
            (private, "task_schema_not_ready", "unrecognized_error_type", "task_schema_not_ready"),
            ("TaskError PRIVATE", "task_schema_not_ready PRIVATE", "unrecognized_error_type", "unrecognized_code"),
        ):
            with self.subTest(expected_type=expected_type, expected_code=expected_code):
                data = json.dumps({"ready": False, "error_type": error_type, "code": code}).encode()
                drain = await self.drain([data])
                receipt = drain.receipt()
                self.assertEqual(receipt["startup_failure"], {
                    "ready": False, "error_type": expected_type, "code": expected_code})
                self.assertNotIn("PRIVATE", json.dumps(receipt))

    async def test_fragmented_diagnostic_survives_clear_and_nested_caller_mutation(self):
        expected = self.diagnostic_failure()
        data = (json.dumps(expected) + "\n").encode()
        self.assertLessEqual(len(data), worker._STARTUP_FAILURE_MAX_LINE_BYTES)
        drain = await self.drain([data[index:index+1] for index in range(len(data))])
        drain.tail.clear()
        self.assertEqual(drain.receipt()["startup_failure"], expected)
        drain.receipt()["startup_failure"]["diagnostic"]["frame"]["line"] = 99
        self.assertEqual(drain.receipt()["startup_failure"], expected)

    async def test_invalid_or_private_diagnostic_is_rejected_not_downgraded(self):
        valid = self.diagnostic_failure()
        diagnostic = valid["diagnostic"]
        invalid = [None, [], diagnostic | {"version": True}, diagnostic | {"version": 2},
            diagnostic | {"stage": "PRIVATE"}, diagnostic | {"stage": "worker"},
            diagnostic | {"basis": "PRIVATE"}, diagnostic | {"truncated": 0},
            diagnostic | {"truncated": True}, diagnostic | {"message": "PRIVATE"},
            diagnostic | {"frame": None}, diagnostic | {"frame": []}]
        invalid += [diagnostic | {"frame": diagnostic["frame"] | change} for change in (
            {"line": True}, {"line": 0}, {"line": 1_000_001}, {"line": 1.5},
            {"source": "/private/ingestion.py"}, {"function": "PRIVATE"},
            {"locals": "PRIVATE"}, {"source": []})]
        for index, value in enumerate(invalid):
            with self.subTest(case=index):
                drain = await self.drain([(json.dumps(valid | {"diagnostic": value}) + "\n").encode()])
                self.assertIsNone(drain.receipt()["startup_failure"])
                self.assertNotIn("PRIVATE", json.dumps(drain.receipt()))

    async def test_unknown_or_bounded_origin_remains_explicit(self):
        for diagnostic in (
            {"version": 1, "stage": "unavailable", "frame": None, "basis": "unavailable", "truncated": False},
            self.diagnostic_failure()["diagnostic"] | {"basis": "filtered_caller", "truncated": True},
        ):
            with self.subTest(basis=diagnostic["basis"]):
                expected = self.diagnostic_failure() | {"diagnostic": diagnostic}
                drain = await self.drain([json.dumps(expected).encode()])
                self.assertEqual(drain.receipt()["startup_failure"], expected)

    async def test_diagnostic_duplicate_and_incomplete_fields_rejected(self):
        data = json.dumps(self.diagnostic_failure()).encode()
        for blob in (data[:-1], data.replace(b'"line": 14', b'"line": 14, "line": 99')):
            with self.subTest(size=len(blob)):
                drain = await self.drain([blob])
                self.assertIsNone(drain.receipt()["startup_failure"])

    async def test_exact_keys_types_and_unique_fields_are_required(self):
        valid = {"ready": False, "error_type": "TaskError", "code": "task_schema_not_ready"}
        invalid = [
            valid | {"ready": True}, valid | {"ready": 0}, valid | {"ready": None},
            valid | {"error_type": None}, valid | {"error_type": ["TaskError"]},
            valid | {"code": {"private": "PRIVATE"}}, valid | {"code": 1},
            valid | {"message": "PRIVATE_EXCEPTION"}, {"ready": False, "error_type": "TaskError"},
            [valid],
        ]
        blobs = [json.dumps(value).encode() for value in invalid]
        blobs.append(b'{"ready":true,"ready":false,"error_type":"TaskError","code":"task_schema_not_ready"}')
        for index, blob in enumerate(blobs):
            with self.subTest(case=index):
                drain = await self.drain([blob + b"\n"])
                self.assertIsNone(drain.receipt()["startup_failure"])
                self.assertNotIn("PRIVATE", json.dumps(drain.receipt()))

    async def test_oversized_noise_is_discarded_until_real_line_boundary(self):
        expected = {"ready": False, "error_type": "TaskError", "code": "task_model_not_accessible"}
        valid = (json.dumps(expected) + "\n").encode()
        # Do not recognize a valid-looking suffix of an oversized/noisy line.
        noise = b"PRIVATE" * 5000 + valid
        chunks = [noise[index:index+8192] for index in range(0, len(noise), 8192)]
        rejected = await self.drain(chunks)
        self.assertIsNone(rejected.receipt()["startup_failure"])
        recovered = await self.drain(chunks + [valid])
        self.assertEqual(recovered.receipt()["startup_failure"], expected)
        self.assertNotIn("PRIVATE", json.dumps(recovered.receipt()))

    async def test_partial_or_invalid_lines_do_not_fabricate_receipt(self):
        valid = b'{"ready":false,"error_type":"TaskError","code":"task_schema_not_ready"}'
        for blob in (valid[:-1], b"noise " + valid, b"\xff" + valid,
                     b"[" * 240 + b"]" * 240):
            with self.subTest(size=len(blob)):
                drain = await self.drain([blob])
                self.assertIsNone(drain.receipt()["startup_failure"])
        complete = await self.drain([valid])
        self.assertEqual(complete.receipt()["startup_failure"]["code"], "task_schema_not_ready")

    async def test_first_failure_only_is_retained(self):
        first = {"ready": False, "error_type": "ValueError", "code": "task_worker_startup_failed"}
        second = {"ready": False, "error_type": "TaskError", "code": "task_memory_not_configured"}
        drain = await self.drain([(json.dumps(first) + "\n" + json.dumps(second) + "\n").encode()])
        self.assertEqual(drain.receipt()["startup_failure"], first)

    async def test_cancelled_drain_keeps_complete_receipt_but_clears_partial_line(self):
        blocked = asyncio.Event()
        expected = {"ready": False, "error_type": "TaskError", "code": "task_worker_not_configured"}

        class Stream:
            sent = False

            async def read(self, _):
                if not self.sent:
                    self.sent = True
                    return (json.dumps(expected) + '\n{"private":"PRIVATE_PARTIAL').encode()
                blocked.set()
                await asyncio.Event().wait()

        drain = worker._PipeDrain(Stream(), 8)
        task = asyncio.create_task(drain.run())
        await asyncio.wait_for(blocked.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(drain.eof)
        self.assertEqual(drain.error, "drain_cancelled")
        self.assertEqual(drain.receipt()["startup_failure"], expected)
        self.assertEqual(len(drain._startup_line), 0)
        self.assertNotIn("PRIVATE", json.dumps(drain.receipt()))
