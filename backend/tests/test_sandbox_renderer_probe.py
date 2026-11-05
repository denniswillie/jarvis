"""Source-only renderer-probe lifecycle checks. No encoder/VM/provider/SQL."""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.files.local_file import snapshot_file
from app.sandbox.models import SandboxError, SandboxPolicy
from evals.second_brain.e2b_probe import Probe
from evals.second_brain.sandbox_renderer_probe import (
    PROTECTED_SOFTWARE_PATHS, RENDER_ENTRYPOINT, SYSTEM_TRUST_FORMAT, RendererBinding, exercise, fixed_props, main,
)


class RendererProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "synthetic-input.mp4"
        self.source.write_bytes(b"inert source fixture" * 8000)
        self.source_sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.source_size = self.source.stat().st_size
        self.encoded = self.root / "fake-encoder-output.mp4"
        self.encoded.write_bytes(b"inert output requires actual decode inspection")
        self.output_sha = hashlib.sha256(self.encoded.read_bytes()).hexdigest()
        self.args = SimpleNamespace(live=True, synthetic=True, input_path=self.source)
        for name, value in (("SOURCE_SHA256", self.source_sha), ("SOURCE_SIZE", self.source_size)):
            change = patch("evals.second_brain.sandbox_renderer_probe." + name, value)
            change.start()
            self.addCleanup(change.stop)
        self.alive = False
        self.software = "f" * 64
        self.transport_files = []
        async def create(scope, policy):
            self.alive = True
            return SimpleNamespace(sandbox_id="only-vm", template_id="candidate-template", scope=scope, policy=policy)
        async def info(*args, **kwargs):
            if not self.alive:
                raise SandboxError("sandbox_not_found")
            return SimpleNamespace(allow_internet_access=False, network={}, lifecycle={})
        async def request(fn, *args, **kwargs):
            return await fn(*args, **kwargs)
        async def kill(*args):
            self.alive = False
        async def stage(handle, identity, local, *, policy):
            self.transport_files.append(local)
            return {"path": "/home/jarvis/inputs/" + identity, "sha256": local.content_sha256, "size_bytes": local.size_bytes}
        async def download(*args, **kwargs):
            local = await snapshot_file(self.encoded, max_bytes=20_000_000)
            self.transport_files.append(local)
            return local
        self.provider = SimpleNamespace(create=AsyncMock(side_effect=create), prepare=AsyncMock(),
            discover=AsyncMock(side_effect=lambda *_: ["only-vm"] if self.alive else []),
            kill=AsyncMock(side_effect=kill), _request=AsyncMock(side_effect=request), _options=lambda: {},
            sdk=SimpleNamespace(AsyncSandbox=SimpleNamespace(get_info=AsyncMock(side_effect=info))),
            stage_input_file=AsyncMock(side_effect=stage), read_workspace_file_snapshot=AsyncMock(side_effect=download))
        self.probe = Probe(self.provider, self.root)
        cleanup = self.probe.cleanup
        self.probe.cleanup = AsyncMock(side_effect=cleanup)
        self.commands = []
        async def run(handle, command):
            self.commands.append(command)
            if command.startswith("/usr/bin/timeout --signal=KILL 20s "):
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"format": SYSTEM_TRUST_FORMAT, "passed": True}))
            if "inventory.py --verify" in command:
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"software_sha256": self.software, "render_qualified": False}))
            return SimpleNamespace(exit_code=0, stdout="encoder returned")
        async def python(handle, code):
            self.assertNotIn(str(self.source), code)
            if "def receipt():" in code:
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"software_sha256": self.software, "uid": 1000,
                    "protection": [{"path": path, "root_owned": True, "readonly": True,
                                    "regular": True, "write_denied": True} for path in PROTECTED_SOFTWARE_PATHS]}))
            props_sha = hashlib.sha256(json.dumps(fixed_props(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            return SimpleNamespace(exit_code=0, stdout=json.dumps({"sha256": self.source_sha,
                "size_bytes": self.source_size, "props_sha256": props_sha}))
        self.probe.run = AsyncMock(side_effect=run)
        self.probe.python = AsyncMock(side_effect=python)
        receipt = self.root / "actual-build-receipt.json"
        receipt.write_text(json.dumps({"template_id": "candidate-template", "build_id": "candidate-build", "status": "ready"}))
        async def confirm(probe, handle):
            return {"sandbox_id": handle.sandbox_id, "template_id": handle.template_id,
                    "state": "running", "allow_internet_access": False}
        async def acquire(probe, policy):
            return await probe.create(policy)
        self.binding = RendererBinding("candidate-template", "candidate-build", self.software, receipt,
            hashlib.sha256(receipt.read_bytes()).hexdigest(), AsyncMock(side_effect=confirm),
            AsyncMock(side_effect=acquire))

    async def test_missing_binding_never_creates_or_renders(self):
        with self.assertRaisesRegex(SandboxError, "reviewed_renderer_binding_required"):
            await exercise(self.probe, self.args)
        self.provider.create.assert_not_awaited()
        self.probe.run.assert_not_awaited()
        self.probe.cleanup.assert_awaited_once()

    async def test_invalid_source_or_build_receipt_fails_before_vm(self):
        self.source.write_bytes(b"wrong exact bytes")
        with self.assertRaises(ValueError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.create.assert_not_awaited()
        self.binding.build_receipt.write_text("changed build record")
        with self.assertRaisesRegex(SandboxError, "renderer_build_receipt_changed"):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.create.assert_not_awaited()

    async def test_one_attempt_retains_private_output_but_never_claims_semantic_pass(self):
        await exercise(self.probe, self.args, binding=self.binding)
        self.provider.create.assert_awaited_once()
        self.provider.stage_input_file.assert_awaited_once()
        self.provider.read_workspace_file_snapshot.assert_awaited_once()
        self.assertEqual(sum(command.startswith(RENDER_ENTRYPOINT + " ") for command in self.commands), 1)
        self.assertEqual(self.provider.create.call_args.args[1].lifetime_seconds, 300)
        self.assertFalse(self.provider.create.call_args.args[1].networked)
        artifact = self.probe.record["inspection_artifact"]
        retained = Path(artifact["path"])
        self.assertEqual(hashlib.sha256(retained.read_bytes()).hexdigest(), self.output_sha)
        self.assertEqual(stat.S_IMODE(retained.stat().st_mode), 0o400)
        self.assertEqual(artifact["semantic_status"], "uninspected")
        self.assertEqual(artifact["status"], "retained_uninspected")
        self.assertTrue(self.probe.record["qualification"]["execution_and_transfer_passed"])
        self.assertFalse(self.probe.record["qualification"]["semantic_passed"])
        self.assertFalse(self.probe.record["qualification"]["overall_passed"])
        for local in self.transport_files:
            self.assertTrue(local._closed)
            self.assertFalse(local.path.exists())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_sha)
        self.assertNotIn("build_id", self.probe.record["native_binding"])
        self.assertEqual(self.probe.record["selected_build_provenance"]["build_id"], "candidate-build")
        self.provider.kill.assert_awaited_once()
        self.assertTrue(self.probe.record["individual_absence"][0]["absent"])

    async def test_supplied_owned_handle_does_not_create_second_vm(self):
        handle = await self.probe.create(SandboxPolicy(template="base", lifetime_seconds=300,
                                                       command_seconds=180, output_bytes=16384))
        await exercise(self.probe, self.args, binding=replace(self.binding, create=None), handle=handle)
        self.provider.create.assert_awaited_once()
        self.binding.create.assert_not_awaited()

    async def test_software_mismatch_cleans_without_upload_or_render(self):
        self.software = "e" * 64
        with self.assertRaises(AssertionError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.stage_input_file.assert_not_awaited()
        self.assertEqual(sum(command.startswith(RENDER_ENTRYPOINT + " ") for command in self.commands), 0)
        self.provider.kill.assert_awaited_once()

    async def test_native_network_or_template_mismatch_is_not_build_proof(self):
        self.binding.confirm.return_value = {"sandbox_id": "only-vm", "template_id": "candidate-template",
                                              "state": "running", "allow_internet_access": True}
        self.binding.confirm.side_effect = None
        with self.assertRaises(AssertionError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.probe.run.assert_not_awaited()
        self.provider.kill.assert_awaited_once()

    async def test_missing_launcher_protection_stops_before_media_upload(self):
        original = self.probe.python.side_effect
        async def wrong_entrypoint(handle, code):
            result = await original(handle, code)
            if "def receipt():" in code:
                observed = json.loads(result.stdout)
                for row in observed["protection"]:
                    if row["path"] == "render-clip":
                        row["path"] = "/usr/local/bin/jarvis-render-clip"
                return SimpleNamespace(exit_code=0, stdout=json.dumps(observed))
            return result
        self.probe.python.side_effect = wrong_entrypoint
        with self.assertRaises(AssertionError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.stage_input_file.assert_not_awaited()
        self.provider.read_workspace_file_snapshot.assert_not_awaited()
        self.assertFalse(any(command.startswith(RENDER_ENTRYPOINT + " ") for command in self.commands))
        self.provider.kill.assert_awaited_once()

    async def test_system_trust_failure_is_retained_before_any_task_or_upload(self):
        observed = {"format": SYSTEM_TRUST_FORMAT, "passed": False,
                    "reason": "writable_ancestor", "path": "/usr/lib"}
        self.probe.run.side_effect = None
        self.probe.run.return_value = SimpleNamespace(exit_code=1, stdout=json.dumps(observed))
        with self.assertRaisesRegex(SandboxError, "renderer_probe_command_failed"):
            await exercise(self.probe, self.args, binding=self.binding)
        self.probe.run.assert_awaited_once()
        self.assertTrue(self.probe.run.call_args.args[1].startswith("/usr/bin/timeout --signal=KILL 20s /usr/bin/python3 -I -S -B -c "))
        self.provider.stage_input_file.assert_not_awaited()
        self.provider.read_workspace_file_snapshot.assert_not_awaited()
        self.probe.python.assert_not_awaited()
        self.provider.kill.assert_awaited_once()
        receipt = self.probe.record["command_receipts"]["verify_system_trust"]
        self.assertEqual(receipt["exit_code"], 1)
        self.assertEqual(json.loads(json.loads(Path(receipt["path"]).read_text())["stdout"]), observed)

    async def test_cancel_during_render_cleans_without_retry_or_success(self):
        started = asyncio.Event()
        original = self.probe.run.side_effect
        async def blocked(handle, command):
            if command.startswith(RENDER_ENTRYPOINT + " "):
                self.commands.append(command)
                started.set()
                await asyncio.Event().wait()
            return await original(handle, command)
        self.probe.run.side_effect = blocked
        task = asyncio.create_task(exercise(self.probe, self.args, binding=self.binding))
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(sum(command.startswith(RENDER_ENTRYPOINT + " ") for command in self.commands), 1)
        self.provider.read_workspace_file_snapshot.assert_not_awaited()
        self.provider.kill.assert_awaited_once()
        self.assertFalse(self.probe.record["qualification"]["execution_and_transfer_passed"])

    async def test_staging_uncertainty_never_renders_or_retries(self):
        self.provider.stage_input_file.side_effect = SandboxError("sandbox_input_cleanup_uncertain")
        with self.assertRaises(SandboxError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.stage_input_file.assert_awaited_once()
        self.assertEqual(sum(command.startswith(RENDER_ENTRYPOINT + " ") for command in self.commands), 0)
        self.provider.kill.assert_awaited_once()

    async def test_render_nonzero_stops_without_download(self):
        async def failed(handle, command):
            if command.startswith("/usr/bin/timeout --signal=KILL 20s "):
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"format": SYSTEM_TRUST_FORMAT, "passed": True}))
            if "inventory.py --verify" in command:
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"software_sha256": self.software}))
            return SimpleNamespace(exit_code=1, stdout="private failure")
        self.probe.run.side_effect = failed
        with self.assertRaises(AssertionError):
            await exercise(self.probe, self.args, binding=self.binding)
        self.provider.read_workspace_file_snapshot.assert_not_awaited()
        self.provider.kill.assert_awaited_once()

    async def test_unresolved_cleanup_keeps_retained_artifact_and_failure(self):
        self.provider.kill.side_effect = SandboxError("sandbox_provider_outcome_uncertain")
        with self.assertRaises(Exception):
            await exercise(self.probe, self.args, binding=self.binding)
        self.assertTrue(Path(self.probe.record["inspection_artifact"]["path"]).exists())
        self.assertEqual(self.probe.record["status"], "cleanup_unresolved")
        self.assertFalse(self.probe.record["qualification"]["execution_and_transfer_passed"])

    async def test_nonzero_inventory_diagnostic_survives_cleanup_privately(self):
        stdout = "inventory verification started\n"
        stderr = "Traceback (most recent call last):\nValueError: software_inventory_changed\n"
        async def fail_inventory(handle, command):
            if command.startswith("/usr/bin/timeout --signal=KILL 20s "):
                return SimpleNamespace(exit_code=0, stdout=json.dumps({"format": SYSTEM_TRUST_FORMAT, "passed": True}))
            return SimpleNamespace(exit_code=23, stdout=stdout, stderr=stderr)
        self.probe.run.side_effect = fail_inventory
        with self.assertRaisesRegex(SandboxError, "renderer_probe_command_failed"):
            await exercise(self.probe, self.args, binding=self.binding)
        summary = self.probe.record["command_receipts"]["verify_installed_software"]
        path = Path(summary["path"])
        receipt = json.loads(path.read_text())
        self.assertEqual(receipt["stage"], "verify_installed_software")
        self.assertEqual(receipt["exit_code"], 23)
        self.assertEqual(receipt["stdout"], stdout)
        self.assertEqual(receipt["stderr"], stderr)
        self.assertEqual(receipt["truncated"], {"stdout": False, "stderr": False})
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
        self.assertNotIn("software_inventory_changed", self.probe.path.read_text())
        self.provider.stage_input_file.assert_not_awaited()
        self.provider.kill.assert_awaited_once()
        self.assertTrue(self.probe.record["individual_absence"][0]["absent"])
        self.assertFalse(self.probe.record["qualification"]["execution_and_transfer_passed"])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_sha)

    async def test_command_diagnostics_have_combined_utf8_bound_and_explicit_truncation(self):
        self.probe.run.side_effect = None
        self.probe.run.return_value = SimpleNamespace(exit_code=1, stdout="é" * 10000, stderr="診" * 10000)
        with self.assertRaisesRegex(SandboxError, "renderer_probe_command_failed"):
            await exercise(self.probe, self.args, binding=self.binding)
        path = Path(self.probe.record["command_receipts"]["verify_system_trust"]["path"])
        receipt = json.loads(path.read_text())
        self.assertLessEqual(sum(receipt["captured_utf8_bytes"].values()), 16384)
        self.assertEqual(receipt["original_utf8_bytes"], {"stdout": 20000, "stderr": 30000})
        self.assertEqual(receipt["truncated"], {"stdout": True, "stderr": True})
        self.assertGreater(receipt["captured_utf8_bytes"]["stderr"], 0)
        self.assertLess(path.stat().st_size, 16384 * 6 + 4096)
        self.provider.kill.assert_awaited_once()

    def test_cli_live_flag_never_enables_creation(self):
        self.assertEqual(main(["--live"]), 2)
        self.provider.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
