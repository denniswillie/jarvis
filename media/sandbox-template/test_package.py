"""Offline package/manifest tests. No installation, browser or build invocation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from unittest.mock import Mock
from contextlib import redirect_stdout
from io import StringIO
import stat
import sys

from template import ROOT, make_template, preflight
from build import identity_observer, read_key, save, submit


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "package"
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns("__pycache__", "*.log"))

    def pins(self, change):
        path = self.root / "pins.json"
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_current_source_and_lock_preflight_complete_no_runtime_claim(self):
        value = preflight(self.root, require_complete=True)
        self.assertEqual(value["missing"], [])
        self.assertFalse(value["runtime_qualified"])
        self.assertEqual(value["source_bundle_sha256"], "15c5f6372789702c41a43a924611d647270df7669c295d99cd4bafee40ea5b51")
        self.assertEqual(value["recipe_sha256"], preflight(self.root)["recipe_sha256"])

    def test_modified_qualified_copy_rejected(self):
        path = self.root / "payload/clip/src/Clip.mjs"
        path.chmod(0o600)
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            preflight(self.root)

    def test_extra_file_cannot_enter_build_context(self):
        (self.root / "payload/.env").write_text("fixture-only")
        with self.assertRaisesRegex(ValueError, "unexpected_build_context_file"):
            preflight(self.root)

    def test_missing_browser_pin_never_admits_build(self):
        self.pins(lambda value: value["chrome"].update(sha256=None))
        self.assertEqual(preflight(self.root)["missing"], ["chrome_archive_sha256"])
        with self.assertRaisesRegex(ValueError, "required_before_build"):
            preflight(self.root, require_complete=True)

    def test_dependency_integrity_mismatch_rejected(self):
        self.pins(lambda value: value.update(compositor_integrity="sha512-wrong"))
        with self.assertRaisesRegex(ValueError, "locked_dependency_mismatch"):
            preflight(self.root)

    def test_sdk_recipe_uses_exact_base_only_explicit_context_and_bounded_commands(self):
        calls = []
        class Builder:
            def __init__(self, **kwargs):
                calls.append(("constructor", kwargs))
            def from_image(self, value):
                calls.append(("image", value)); return self
            def run_cmd(self, command, **kwargs):
                calls.append(("run", command, kwargs)); return self
            def copy(self, source, destination, **kwargs):
                calls.append(("copy", source, destination, kwargs)); return self
        make_template(Builder, self.root)
        self.assertIn("@sha256:", calls[1][1])
        self.assertEqual([item[1] for item in calls if item[0] == "copy"], ["payload/", "pins.json", "source-lock.json"])
        commands = [item[1] for item in calls if item[0] == "run"]
        self.assertTrue(commands[0].startswith("timeout 240 "))
        self.assertTrue(commands[1].startswith("timeout 420 "))
        self.assertIn("20261101T000000Z", commands[0])

    def test_symlink_qualified_source_rejected(self):
        target = self.root / "payload/clip/package.json"
        data = target.read_bytes()
        target.unlink()
        outside = self.root / "elsewhere"
        outside.write_bytes(data)
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
            preflight(self.root)

    def test_inventory_definition_excludes_jobs_and_detects_software_change(self):
        scope = {"__name__": "fixture"}
        exec((self.root / "payload/inventory.py").read_text(), scope)
        software = self.root / "software"
        software.mkdir()
        (software / "tool").write_bytes(b"installed-tool")
        first = scope["entries"](software, require_root=False)
        (self.root / "job.mp4").write_bytes(b"per-job-media")
        (software / "software-inventory.json").write_text("self-receipt")
        self.assertEqual(first, scope["entries"](software, require_root=False))
        (software / "tool").write_bytes(b"changed-tool")
        self.assertNotEqual(first, scope["entries"](software, require_root=False))
        (software / "escape").symlink_to(self.root / "job.mp4")
        with self.assertRaisesRegex(ValueError, "symlink_escape"):
            scope["entries"](software, require_root=False)

    def test_build_credentials_are_explicit_and_not_environment_fallback(self):
        path = self.root / "fixture.env"
        path.write_text("E2B_API_KEY=fixture-only-key\n")
        args = SimpleNamespace(api_key_file=None, credential_file=path)
        self.assertEqual(read_key(args), "fixture-only-key")
        path.write_text("OTHER=irrelevant\n")
        with patch.dict("os.environ", {"E2B_API_KEY": "must-not-use"}):
            with self.assertRaises(ValueError):
                read_key(args)
        args.api_key_file = path
        with self.assertRaises(ValueError):
            read_key(args)

    def test_atomic_receipt_replacements_remain_private(self):
        path = self.root / "receipt.json"
        for index in range(3):
            save(path, {"index": index})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(json.loads(path.read_text()), {"index": index})

    def test_identity_callback_ignores_every_other_log_and_preserves_conflict(self):
        path = self.root / "identity.json"
        record = {}
        observe = identity_observer(path, record)
        for message in ("unrelated secret-looking log", "prefix Template created with ID: t, Build ID: b",
                        "Template created with ID: t, Build ID: b\nprivate-tail"):
            observe(SimpleNamespace(level="info", message=message))
        self.assertEqual(record, {})
        self.assertFalse(path.exists())
        observe(SimpleNamespace(level="info", message="Template created with ID: first-t, Build ID: first-b"))
        with self.assertRaisesRegex(ValueError, "identity_conflict"):
            observe(SimpleNamespace(level="info", message="Template created with ID: second-t, Build ID: second-b"))
        self.assertEqual(record["template_id"], "first-t")
        self.assertEqual(record["build_id"], "first-b")
        self.assertEqual(record["identity_conflict"]["template_id"], "second-t")


class BuildControlTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, behavior):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            credential = root / "key"
            credential.write_text("fixture-only-key")
            args = SimpleNamespace(manifest=root / "manifest.json", api_key_file=credential,
                                   credential_file=None, name="jarvis-clip-candidate-fixture-only")
            candidate = {"recipe_sha256": "a" * 64, "pins": {"proposed_build": {"count": 1}}}
            async def build(*args_, **kwargs):
                kwargs["on_build_logs"](SimpleNamespace(level="info", message="Template created with ID: template-one, Build ID: build-one"))
                saved = json.loads(args.manifest.read_text())
                self.assertEqual(saved["template_id"], "template-one")
                self.assertEqual(stat.S_IMODE(args.manifest.stat().st_mode), 0o600)
                if behavior == "loss":
                    raise TimeoutError("provider-private-message")
                return SimpleNamespace(template_id="template-other" if behavior == "conflict" else "template-one", build_id="build-one")
            api = SimpleNamespace(exists=AsyncMock(return_value=False), build_in_background=AsyncMock(side_effect=build),
                get_build_status=AsyncMock(return_value=SimpleNamespace(status=SimpleNamespace(value="ready"))))
            fake_e2b = SimpleNamespace(AsyncTemplate=api, Template=object)
            with patch.dict(sys.modules, {"e2b": fake_e2b}), patch("build.version", return_value="2.49.1"), \
                    patch("build.make_template", return_value=object()), patch("build.preflight", return_value=candidate):
                if behavior == "success":
                    result = await submit(args, candidate)
                    self.assertTrue(result["returned_identity_confirmed"])
                    self.assertEqual(result["status"], "ready")
                else:
                    with self.assertRaises(TimeoutError if behavior == "loss" else ValueError):
                        await submit(args, candidate)
            saved = json.loads(args.manifest.read_text())
            self.assertEqual(saved["template_id"], "template-one")
            self.assertEqual(saved["build_id"], "build-one")
            self.assertNotIn("provider-private-message", args.manifest.read_text())
            self.assertEqual(stat.S_IMODE(args.manifest.stat().st_mode), 0o600)
            api.build_in_background.assert_awaited_once()
            if behavior != "success":
                self.assertTrue(saved["observation_lost"])
                self.assertTrue(saved["cleanup_required"])
                api.get_build_status.assert_not_awaited()
            if behavior == "conflict":
                self.assertEqual(saved["returned_identity_conflict"]["template_id"], "template-other")

    async def test_loss_after_early_identity_preserves_same_build(self):
        await self.exercise("loss")

    async def test_success_confirms_exact_early_identity(self):
        await self.exercise("success")

    async def test_returned_identity_conflict_preserves_original(self):
        await self.exercise("conflict")


class InventoryTests(unittest.TestCase):
    """Real local hashing; root metadata and dpkg are doubled, no tool execution."""
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "software"
        self.root.mkdir()
        self.addCleanup(lambda: self.root.chmod(0o755))
        spec = importlib.util.spec_from_file_location("inventory_fixture", ROOT / "payload/inventory.py")
        self.inventory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.inventory)
        self.inventory.ROOT = self.root
        (self.root / "pins.json").write_bytes((ROOT / "pins.json").read_bytes())
        (self.root / "render-clip").write_bytes(b"fixed launcher")
        (self.root / "tool").write_bytes(b"fixed payload")
        self.receipt = self.root / "software-inventory.json"
        self.receipt.write_text("placeholder excluded from payload digest")
        for path in self.root.iterdir():
            path.chmod(0o555 if path.name == "render-clip" else 0o444)
        self.core = self.root.parent / "core-command"
        self.core.write_bytes(b"fixed core command")
        self.core.chmod(0o555)
        self.inventory.CORE_COMMANDS = (str(self.core),)
        self.real_protected = self.inventory.protected
        def root_metadata(path, **kwargs):
            info = path.lstat()
            return self.real_protected(self.fake_path(path, uid=0, mode=info.st_mode), **kwargs)
        protection = patch.object(self.inventory, "protected", side_effect=root_metadata)
        protection.start()
        self.addCleanup(protection.stop)
        self.root.chmod(0o555)
        self.packages = "fixture-package\t1\tamd64\n"
        def package_command(args):
            self.assertEqual(args, ["/usr/bin/dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"])
            return self.packages
        command = patch.object(self.inventory, "command", side_effect=package_command)
        self.command = command.start()
        self.addCleanup(command.stop)
        forbidden = patch.object(self.inventory, "build_versions", side_effect=AssertionError("runtime version execution forbidden"))
        forbidden.start()
        self.addCleanup(forbidden.stop)
        self.versions = {"node": "v22.23.4", "npm": "fixture npm", "chrome": "Google Chrome for Testing 157.0.8080.0",
                         "ffmpeg": "fixture ffmpeg", "ffprobe": "fixture ffprobe", "remotion": "4.0.539"}
        software = self.inventory.software(versions=self.versions)
        self.saved = {"software": software, "software_sha256": hashlib.sha256(self.inventory.canonical(software)).hexdigest(),
                      "render_qualified": False}
        self.write_receipt(self.saved)
        self.command.reset_mock()

    @staticmethod
    def fake_path(path, *, uid, mode):
        class FakePath:
            def lstat(self):
                return SimpleNamespace(st_uid=uid, st_mode=mode)
            def __str__(self):
                return str(path)
        return FakePath()

    def write_receipt(self, value):
        self.receipt.chmod(0o600)
        self.receipt.write_text(json.dumps(value))
        self.receipt.chmod(0o444)

    def test_runtime_verifies_identity_with_only_existing_dpkg_query(self):
        with patch.object(self.inventory.subprocess, "run", side_effect=AssertionError("no real subprocess")):
            self.assertEqual(self.inventory.verify(), self.saved["software_sha256"])
        self.command.assert_called_once()
        self.assertEqual(self.saved["software"]["format"], "jarvis.clip.software.v2")
        self.assertEqual(self.saved["software"]["versions"], self.versions)

    def test_receipt_self_digest_fails_before_version_reuse_or_identity_scan(self):
        self.saved["software"]["versions"]["node"] = "forged"
        self.write_receipt(self.saved)
        with self.assertRaisesRegex(ValueError, "inventory_receipt_sha256_mismatch"):
            self.inventory.verify()
        self.command.assert_not_called()

    def test_changed_payload_reports_exact_sha_operand(self):
        path = self.root / "tool"
        path.chmod(0o600)
        path.write_bytes(b"changed payload")
        path.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "inventory_files_sha256_mismatch:tool:expected="):
            self.inventory.verify()

    def test_changed_core_hash_is_not_covered_by_reusing_build_versions(self):
        self.core.chmod(0o600)
        self.core.write_bytes(b"changed core command")
        self.core.chmod(0o555)
        with self.assertRaisesRegex(ValueError, "inventory_core_sha256_mismatch:"):
            self.inventory.verify()

    def test_system_package_comparison_remains_required(self):
        self.packages = "different-package\t2\tamd64\n"
        with self.assertRaisesRegex(ValueError, "inventory_system_packages_mismatch"):
            self.inventory.verify()

    def test_launcher_mode_and_protected_root_cannot_be_relaxed(self):
        launcher = self.root / "render-clip"
        launcher.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "software_mode_mismatch:.*render-clip:expected=0555:observed=0777"):
            self.inventory.verify()
        launcher.chmod(0o555)
        self.root.chmod(0o777)
        with self.assertRaisesRegex(ValueError, "software_mode_unprotected:.*software:observed=0777"):
            self.inventory.verify()

    def test_guard_reports_uid_and_rejects_symlink_ancestor(self):
        with self.assertRaisesRegex(ValueError, "software_uid_mismatch:/opt:expected=0:observed=1000"):
            self.real_protected(self.fake_path("/opt", uid=1000, mode=stat.S_IFDIR | 0o755), kind="directory")
        with self.assertRaisesRegex(ValueError, "software_type_not_directory:/opt"):
            self.real_protected(self.fake_path("/opt", uid=0, mode=stat.S_IFLNK | 0o777), kind="directory")
        with self.assertRaisesRegex(ValueError, "software_mode_unprotected:/opt:observed=0777"):
            self.real_protected(self.fake_path("/opt", uid=0, mode=stat.S_IFDIR | 0o777), kind="directory")

    def test_launcher_stays_in_sealed_tree_without_mutable_usr_local_links(self):
        script = (ROOT / "payload/install.sh").read_text()
        self.assertIn("chmod 0555 /opt/jarvis-render/render-clip", script)
        self.assertNotIn("install -m 0755 render-clip", script)
        self.assertNotIn("ln -s ", script)
        launcher = (ROOT / "payload/render-clip").read_text()
        self.assertNotIn("/usr/local", launcher)
        self.assertIn("/opt/jarvis-render/clip/node_modules/@remotion/compositor-linux-x64-gnu", launcher)

    def test_owner_only_file_and_untraversable_directory_rejected_even_for_root_inventory(self):
        for mode, kind in ((0o400, "file"), (0o444, "directory"), (0o4555, "file"), (0o2555, "directory")):
            typed = (stat.S_IFREG if kind == "file" else stat.S_IFDIR) | mode
            with self.subTest(mode=oct(mode)), self.assertRaisesRegex(ValueError, "software_mode_not_worker_readable"):
                self.real_protected(self.fake_path("/opt/jarvis-render/fixture", uid=0, mode=typed), kind=kind, readonly=True)


class PayloadAccessTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("package_access_fixture", ROOT / "payload/package_access.py")
        self.access = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.access)

    def receipt(self):
        return {"format": self.access.FORMAT, "uid": 65534, "euid": 65534, "gid": 65534,
                "egid": 65534, "groups": [], "passed": True, "software_sha256": "a" * 64,
                "access_paths_checked": 3, "render_qualified": False}

    def test_normalization_is_payload_wide_readable_traversable_preserves_exec_without_special_bits(self):
        for before, expected in ((stat.S_IFREG | 0o600, 0o444), (stat.S_IFREG | 0o400, 0o444),
                                 (stat.S_IFDIR | 0o700, 0o555), (stat.S_IFDIR | 0o000, 0o555),
                                 (stat.S_IFREG | 0o751, 0o555), (stat.S_IFREG | 0o4755, 0o555),
                                 (stat.S_IFREG | 0o2600, 0o444), (stat.S_IFDIR | 0o3777, 0o555)):
            with self.subTest(before=oct(before)):
                self.assertEqual(self.access.normalized_mode(before), expected)
        self.assertIsNone(self.access.normalized_mode(stat.S_IFLNK | 0o777))
        with self.assertRaises(ValueError):
            self.access.normalized_mode(stat.S_IFIFO | 0o644)

    def test_seal_plans_every_file_without_following_symlinks_or_changing_bytes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            data, executable = root / "upstream-private-metadata", root / "existing-tool"
            data.write_bytes(b"preserve original0600 metadata")
            data.chmod(0o600)
            executable.write_bytes(b"preserve tool")
            executable.chmod(0o755)
            (root / "alias").symlink_to("existing-tool")
            with patch.object(self.access.os, "geteuid", return_value=0), \
                    patch.object(self.access.os, "chown") as chown, patch.object(self.access.os, "chmod") as chmod:
                self.access.seal(root)
            modes = {call.args[0]: call.args[1] for call in chmod.call_args_list}
            self.assertEqual(modes, {data: 0o444, executable: 0o555, root: 0o555})
            self.assertEqual(chown.call_count, 4)
            self.assertTrue(all(call.kwargs == {"follow_symlinks": False} for call in chown.call_args_list + chmod.call_args_list))
            self.assertEqual(data.read_bytes(), b"preserve original0600 metadata")
            self.assertEqual(executable.read_bytes(), b"preserve tool")
            # Escaping symlink rejection occurs before any chmod/chown.
            (root / "escape").symlink_to(root.parent)
            with patch.object(self.access.os, "geteuid", return_value=0), \
                    patch.object(self.access.os, "chown") as chown, patch.object(self.access.os, "chmod") as chmod, \
                    self.assertRaisesRegex(ValueError, "payload_symlink_escape"):
                self.access.seal(root)
            chown.assert_not_called()
            chmod.assert_not_called()

    def test_gate_receipt_requires_actual_nonroot_ids_empty_groups_and_exact_hash(self):
        self.access.require_worker_receipt(self.receipt(), "a" * 64)
        for changes in ({"uid": 0}, {"euid": 0}, {"gid": 0}, {"egid": 0}, {"groups": [0]},
                        {"groups": [65534]}, {"passed": False}, {"software_sha256": "b" * 64},
                        {"render_qualified": True}, {"access_paths_checked": 0}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "receipt_invalid"):
                self.access.require_worker_receipt(self.receipt() | changes, "a" * 64)

    def test_child_inaccessible_path_produces_failed_receipt_without_root_fallback(self):
        class FakePath:
            def __init__(self, value): self.value = value
            def __truediv__(self, value): return FakePath(self.value + "/" + value)
            def __str__(self): return self.value
            def is_dir(self): return self.value == "/payload"
            def stat(self): return SimpleNamespace(st_mode=stat.S_IFREG | 0o444)
        namespace = {"ROOT": FakePath("/payload"), "verify": Mock(return_value="a" * 64),
                     "load_receipt": lambda: {"software": {"files": [{"path": "blocked"}]}}}
        output = StringIO()
        import runpy
        with patch.object(self.access.os, "getuid", return_value=65534), \
                patch.object(self.access.os, "geteuid", return_value=65534), \
                patch.object(self.access.os, "getgid", return_value=65534), \
                patch.object(self.access.os, "getegid", return_value=65534), \
                patch.object(self.access.os, "getgroups", return_value=[]), \
                patch.object(self.access.os, "access", side_effect=lambda path, *_args, **_kwargs: str(path) != "/payload/blocked"), \
                patch.object(runpy, "run_path", return_value=namespace), \
                patch.object(sys, "flags", SimpleNamespace(isolated=1, no_site=1, dont_write_bytecode=1)), \
                redirect_stdout(output), self.assertRaises(SystemExit) as result:
            exec(self.access.CHILD, {})
        self.assertEqual(result.exception.code, 1)
        record = json.loads(output.getvalue())
        self.assertFalse(record["passed"])
        self.assertIn("worker_payload_inaccessible:/payload/blocked", record["failure"]["detail"])
        namespace["verify"].assert_called_once()

    def test_supervisor_clears_groups_and_reaps_same_child_on_timeout(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "software-inventory.json").write_text(json.dumps({"software_sha256": "a" * 64}))
            out, err = Mock(), Mock()
            out.fileno.return_value, err.fileno.return_value = 101, 102
            process = SimpleNamespace(pid=23456, stdout=out, stderr=err, wait=Mock(), returncode=None)
            selector = SimpleNamespace(register=Mock(), get_map=lambda: {101: object()}, close=Mock())
            with patch.object(self.access, "ROOT", root), patch.object(self.access.os, "geteuid", return_value=0), \
                    patch.object(self.access.subprocess, "Popen", return_value=process) as popen, \
                    patch.object(self.access.selectors, "DefaultSelector", return_value=selector), \
                    patch.object(self.access.os, "set_blocking"), patch.object(self.access.os, "killpg") as kill, \
                    patch.object(self.access.time, "monotonic", side_effect=[0, 31]), \
                    self.assertRaisesRegex(ValueError, "gate_timeout"):
                self.access.worker_check(root)
            popen.assert_called_once()
            self.assertEqual(popen.call_args.kwargs["user"], 65534)
            self.assertEqual(popen.call_args.kwargs["group"], 65534)
            self.assertEqual(popen.call_args.kwargs["extra_groups"], ())
            self.assertEqual(popen.call_args.kwargs["cwd"], "/")
            self.assertEqual(popen.call_args.args[0][:5], ["/usr/bin/python3", "-I", "-S", "-B", "-c"])
            kill.assert_called_once_with(23456, self.access.signal.SIGKILL)
            process.wait.assert_called_once_with(timeout=1)

    def test_install_requires_worker_gate_inside_existing_budget(self):
        script = (ROOT / "payload/install.sh").read_text()
        self.assertLess(script.index("package_access.py --seal"), script.index("inventory.py --write"))
        self.assertLess(script.index("inventory.py --write"), script.index("package_access.py --worker-check"))
        self.assertIn("/usr/bin/timeout 35s", script)
        self.assertIn('"timeout 420 /bin/bash /opt/jarvis-render/install.sh"', (ROOT / "template.py").read_text())


if __name__ == "__main__":
    unittest.main()
