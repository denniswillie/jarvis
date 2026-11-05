"""Finite probe-control checks with real local snapshots and provider doubles."""
import asyncio
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.files.local_file import snapshot_file
from app.sandbox.models import SandboxError
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.sandbox_media_input_probe import exercise, read_key, validate


class ProbeControlTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.original = self.root / "synthetic.mp4"
        self.original.write_bytes(b"synthetic-not-a-decoding-test" * 8000)
        self.digest = hashlib.sha256(self.original.read_bytes()).hexdigest()
        self.args = SimpleNamespace(live=True, synthetic=True, credential_file=self.root / "never-read.env", api_key_file=None,
            manifest_dir=self.root / "new", input_path=self.original, input_sha256=self.digest,
            input_size=self.original.stat().st_size, media_policy="mp4_capture_v1",
            deployment_limit_bytes=20_000_000, inventory=False)
        self.alive = False
        self.input_file = self.output_file = None
        async def create(scope, policy):
            self.alive = True
            return SimpleNamespace(sandbox_id="only-vm", template_id="base", scope=scope, policy=policy)
        async def get_info(*args, **kwargs):
            if not self.alive:
                raise SandboxError("sandbox_not_found")
            return SimpleNamespace(allow_internet_access=False, network={}, lifecycle={})
        async def request(fn, *args, **kwargs):
            return await fn(*args, **kwargs)
        async def kill(*args):
            self.alive = False
        async def stage(handle, identity, local, *, policy):
            self.input_file = local
            self.identity = identity
            return {"path": "/home/jarvis/inputs/" + identity, "sha256": local.content_sha256,
                    "size_bytes": local.size_bytes}
        async def download(*args, **kwargs):
            self.output_file = await snapshot_file(self.original, max_bytes=20_000_000)
            return self.output_file
        self.provider = SimpleNamespace(create=AsyncMock(side_effect=create), prepare=AsyncMock(),
            discover=AsyncMock(side_effect=lambda *_: ["only-vm"] if self.alive else []),
            kill=AsyncMock(side_effect=kill), _request=AsyncMock(side_effect=request), _options=lambda: {},
            sdk=SimpleNamespace(AsyncSandbox=SimpleNamespace(get_info=AsyncMock(side_effect=get_info))),
            stage_input_file=AsyncMock(side_effect=stage), read_workspace_file_snapshot=AsyncMock(side_effect=download))
        self.probe = Probe(self.provider, self.root)
        self.actual_cleanup = self.probe.cleanup
        self.probe.cleanup = AsyncMock(side_effect=self.actual_cleanup)
        async def read_copy(handle, code):
            self.assertNotIn(str(self.original), code)
            return SimpleNamespace(exit_code=0, stdout=json.dumps({
                "path": "/home/jarvis/inputs/" + self.identity, "sha256": self.digest,
                "size_bytes": self.args.input_size, "uid": 1000, "input_uid": 0, "input_mode": 0o444,
                "write_permission_denied": True, "output_path": "media-input-roundtrip.mp4"}))
        self.probe.python = AsyncMock(side_effect=read_copy)

    async def test_invalid_exact_input_never_creates_vm_and_still_cleans(self):
        self.args.input_sha256 = "0" * 64
        with self.assertRaises(ValueError):
            await exercise(self.probe, self.args)
        self.provider.create.assert_not_awaited()
        self.provider.stage_input_file.assert_not_awaited()
        self.probe.cleanup.assert_awaited_once()
        self.assertFalse(self.probe.record["qualification"]["passed"])

    async def test_one_vm_one_upload_copy_download_and_individual_absence(self):
        await exercise(self.probe, self.args)
        self.provider.create.assert_awaited_once()
        policy = self.provider.create.call_args.args[1]
        self.assertEqual(policy.lifetime_seconds, 300)
        self.assertFalse(policy.networked)
        self.provider.stage_input_file.assert_awaited_once()
        self.probe.python.assert_awaited_once()
        self.provider.read_workspace_file_snapshot.assert_awaited_once()
        self.probe.cleanup.assert_awaited_once()
        self.provider.kill.assert_awaited_once()
        self.assertTrue(self.probe.record["qualification"]["passed"])
        self.assertEqual(self.probe.record["individual_absence"], [
            {"sandbox_id": "only-vm", "absent": True, "code": "sandbox_not_found"}])
        self.assertEqual(self.probe.record["private_files"], {
            "input": {"closed": True, "removed": True}, "output": {"closed": True, "removed": True}})
        self.assertEqual(hashlib.sha256(self.original.read_bytes()).hexdigest(), self.digest)

    async def test_staging_uncertainty_stops_without_copy_download_or_retry(self):
        self.provider.stage_input_file.side_effect = SandboxError("sandbox_input_cleanup_uncertain")
        with self.assertRaisesRegex(SandboxError, "sandbox_input_cleanup_uncertain"):
            await exercise(self.probe, self.args)
        self.provider.create.assert_awaited_once()
        self.provider.stage_input_file.assert_awaited_once()
        self.probe.python.assert_not_awaited()
        self.provider.read_workspace_file_snapshot.assert_not_awaited()
        self.probe.cleanup.assert_awaited_once()
        self.provider.kill.assert_awaited_once()
        self.assertFalse(self.alive)
        self.assertTrue(self.probe.record["private_files"]["input"]["removed"])

    async def test_actual_task_cancel_at_upload_runs_cleanup_and_closes_local(self):
        started = asyncio.Event()
        async def blocked(*args, **kwargs):
            self.input_file = args[2]
            started.set()
            await asyncio.Event().wait()
        self.provider.stage_input_file.side_effect = blocked
        task = asyncio.create_task(exercise(self.probe, self.args))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.input_file._closed)
        self.assertFalse(self.input_file.path.exists())
        self.probe.cleanup.assert_awaited_once()
        self.provider.kill.assert_awaited_once()
        self.assertFalse(self.probe.record["qualification"]["passed"])

    async def test_unknown_create_preserves_cleanup_unresolved_and_no_replacement(self):
        self.provider.create.side_effect = SandboxError("sandbox_provider_outcome_uncertain")
        with self.assertRaises(SandboxError):
            await exercise(self.probe, self.args)
        self.provider.create.assert_awaited_once()
        self.probe.cleanup.assert_awaited_once()
        self.assertEqual(self.probe.record["status"], "cleanup_unresolved")
        self.assertEqual(self.probe.record["cleanup"]["failures"][0]["phase"], "unresolved_create")

    async def test_main_deadline_at_upload_stops_and_cleans_one_allocation(self):
        async def blocked(*args, **kwargs):
            await asyncio.Event().wait()
        self.provider.stage_input_file.side_effect = blocked
        with patch("evals.second_brain.sandbox_media_input_probe.MAIN_SECONDS", 0.05):
            with self.assertRaises(TimeoutError):
                await exercise(self.probe, self.args)
        self.provider.create.assert_awaited_once()
        self.provider.stage_input_file.assert_awaited_once()
        self.provider.kill.assert_awaited_once()
        self.probe.cleanup.assert_awaited_once()
        self.assertFalse(self.alive)

    async def test_individual_get_info_uncertainty_cannot_be_claimed_absent(self):
        actual = self.provider.sdk.AsyncSandbox.get_info.side_effect
        async def info(*args, **kwargs):
            if not self.alive:
                raise SandboxError("sandbox_provider_outcome_uncertain")
            return await actual(*args, **kwargs)
        self.provider.sdk.AsyncSandbox.get_info.side_effect = info
        with self.assertRaisesRegex(SandboxError, "probe_cleanup_unresolved"):
            await exercise(self.probe, self.args)
        self.assertFalse(self.probe.record["individual_absence"][0]["absent"])
        self.assertEqual(self.probe.record["status"], "cleanup_unresolved")
        self.assertFalse(self.probe.record["qualification"]["passed"])

    async def test_cleanup_deadline_is_recorded_without_retry(self):
        async def stuck():
            await asyncio.Event().wait()
        self.probe.cleanup.side_effect = stuck
        with patch("evals.second_brain.sandbox_media_input_probe.CLEANUP_SECONDS", 0.05):
            with self.assertRaises(TimeoutError):
                await exercise(self.probe, self.args)
        self.probe.cleanup.assert_awaited_once()
        self.assertEqual(self.probe.record["status"], "cleanup_unresolved")
        self.assertEqual(self.probe.record["cleanup_error"], "TimeoutError")

    def test_scope_and_manifest_admission(self):
        for key, value in (("live", False), ("synthetic", False), ("deployment_limit_bytes", 20_000_001),
                           ("input_path", Path("relative.mp4")), ("media_policy", "document_v1")):
            args = SimpleNamespace(**vars(self.args))
            setattr(args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(args, new_directory=True)
        self.args.manifest_dir.mkdir()
        with self.assertRaisesRegex(ValueError, "new manifest"):
            validate(self.args, new_directory=True)

    def test_explicit_raw_or_dotenv_source_has_no_environment_fallback(self):
        raw = self.root / "synthetic-key.txt"
        raw.write_text("fixture-only-key\n")
        self.args.api_key_file, self.args.credential_file = raw, None
        validate(self.args, new_directory=True)
        self.assertEqual(read_key(self.args), "fixture-only-key")
        env = self.root / "synthetic.env"
        env.write_text("E2B_API_KEY=fixture-dotenv-key\n")
        self.args.api_key_file, self.args.credential_file = None, env
        self.assertEqual(read_key(self.args), "fixture-dotenv-key")
        env.write_text("OTHER=unused\n")
        with patch.dict("os.environ", {"E2B_API_KEY": "must-not-be-read"}):
            self.assertIsNone(read_key(self.args))
        self.args.api_key_file = raw
        with self.assertRaises(ValueError):
            validate(self.args, new_directory=True)


if __name__ == "__main__":
    unittest.main()
