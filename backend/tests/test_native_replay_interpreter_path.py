"""CLI-to-worker interpreter path regression; no executable is launched.

Real temporary symlinks distinguish a requested environment executable from its
base-Python target. Only parsing, RunOptions and worker-options construction run;
settings, run_once, process/host construction and worker.start are stubbed.
This does not qualify Hermes imports, native startup, providers or SQL.
"""

import asyncio
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch, sentinel

from evals.responsive_voice import native_replay_run as replay
from evals.responsive_voice import native_task_worker as worker


class NativeReplayInterpreterPathTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="offline-replay-interpreter-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.python_target = Path(sys.executable).resolve()
        self.pinned = self.root / "venv" / "bin" / "python"
        self.pinned.parent.mkdir(parents=True)
        self.pinned.symlink_to(self.python_target)
        self.assertTrue(self.pinned.is_symlink())
        self.assertTrue(self.pinned.samefile(self.python_target))

        data = self.root / "actual-data"
        data.mkdir()
        (self.root / "linked-data").symlink_to(data, target_is_directory=True)
        self.data_paths = {}
        for key in ("input_manifest", "media_root", "speech_manifest", "artifact_dir",
                    "ledger_dir", "risk_authorization", "canonical_export_command",
                    "hindsight_api_key_file", "worker_api_key_file"):
            target = data / key
            if key in {"media_root", "artifact_dir", "ledger_dir"}:
                target.mkdir()
            else:
                target.write_text("offline fixture only\n", encoding="ascii")
            self.data_paths[key] = target

    def cli_options(self, requested_python):
        argv = ["--execute-once", "--hindsight-url", "https://offline.invalid",
                "--worker-model", "fixture-frontier", "--worker-reasoning-effort", "low"]
        for key in self.data_paths:
            argv.extend(["--" + key.replace("_", "-"), str(Path("linked-data") / key)])
        if requested_python is not None:
            argv.extend(["--worker-python", str(requested_python),
                         "--workload", replay.REASONING_REMINDER,
                         "--input-mode", replay.PREPARED])
        receipt = {"status": "lifecycle_complete", "journal": "offline-stub-only",
                   "launch_attempts": 0, "local_cleanup_complete": True,
                   "provider_evidence": "not_attempted_offline"}
        output = io.StringIO()
        with chdir(self.root), redirect_stdout(output), \
                patch("app.settings.JarvisSettings", return_value=SimpleNamespace(
                    SUPABASE_DB_URL="offline-dsn-no-connection")) as settings, \
                patch.object(replay, "HostBoundary", return_value=sentinel.host) as host, \
                patch.object(replay, "ProcessBoundary", return_value=sentinel.process) as process, \
                patch.object(replay, "run_once", new_callable=AsyncMock, return_value=receipt) as run_once:
            self.assertEqual(replay.main(argv), 0)
        settings.assert_called_once_with()
        host.assert_called_once_with("offline-dsn-no-connection")
        process.assert_called_once_with()
        run_once.assert_awaited_once()
        options = run_once.await_args.args[0]
        self.assertIsInstance(options, replay.RunOptions)
        self.assertEqual(run_once.await_args.kwargs, {"host": sentinel.host, "process": sentinel.process})
        self.assertEqual(json.loads(output.getvalue()), receipt)
        for key, target in self.data_paths.items():
            with self.subTest(data_path=key):
                self.assertEqual(getattr(options, key), target)
                self.assertTrue(getattr(options, key).is_absolute())
                self.assertNotIn("linked-data", getattr(options, key).parts)
        return options

    def assert_worker_handoff(self, options, expected_python):
        self.assertEqual(options.worker_python, expected_python)
        self.assertTrue(options.worker_python.is_absolute())
        self.assertNotEqual(options.worker_python, self.python_target)
        self.assertTrue(options.worker_python.samefile(self.python_target))
        resources = {"fixture_manifest": "offline-only-not-read"}
        with patch.object(worker, "start", new_callable=AsyncMock, return_value=sentinel.worker) as start:
            result = asyncio.run(replay.HostBoundary("offline-dsn-no-connection").start_worker(options, resources))
        self.assertIs(result, sentinel.worker)
        start.assert_awaited_once()
        passed = start.await_args.kwargs
        self.assertIs(passed["resources"], resources)
        self.assertEqual(passed["dsn"], "offline-dsn-no-connection")
        self.assertIsInstance(passed["options"], worker.NativeTaskWorkerOptions)
        self.assertEqual(passed["options"].hermes_python, expected_python)
        self.assertNotEqual(passed["options"].hermes_python, self.python_target)
        self.assertEqual(passed["options"].model, options.worker_model)
        self.assertEqual(passed["options"].api_key_file, self.data_paths["worker_api_key_file"])
        self.assertEqual(passed["options"].artifact_dir, self.data_paths["artifact_dir"])
        self.assertEqual(passed["options"].readiness_timeout_seconds, replay.WORKER_READINESS_SECONDS)

    def test_absolute_symlink_survives_cli_and_worker_handoff(self):
        options = self.cli_options(self.pinned)
        self.assert_worker_handoff(options, self.pinned)

    def test_relative_symlink_is_absolutized_without_dereferencing(self):
        requested = Path("venv/bin/python")
        options = self.cli_options(requested)
        self.assert_worker_handoff(options, self.root / requested)

    def test_relative_symlink_parent_traversal_keeps_executable_semantics(self):
        (self.root / "python-parent").symlink_to(self.pinned.parent, target_is_directory=True)
        requested = Path("python-parent/../bin/python")
        # Kernel traversal follows python-parent before applying '..'. Lexical
        # collapse or resolve would change the requested environment path.
        expected = self.root / requested
        self.assertTrue(expected.samefile(self.python_target))
        options = self.cli_options(requested)
        self.assertIn("..", options.worker_python.parts)
        self.assert_worker_handoff(options, expected)

    def test_optional_worker_python_stays_none_and_data_paths_still_resolve(self):
        options = self.cli_options(None)
        self.assertIsNone(options.worker_python)
        self.assertEqual(options.workload, replay.NO_WORKLOAD)


if __name__ == "__main__":
    unittest.main()
