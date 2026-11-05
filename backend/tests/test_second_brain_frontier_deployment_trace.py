import json
from contextlib import asynccontextmanager
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from evals.second_brain.frontier_deployment_trace import run


class FrontierDeploymentCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, logs_fail=False, down_fail=False):
        connection = AsyncMock()
        connection.fetchval.return_value = "jarvis_sb_eval_synthetic"
        pool = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
        @asynccontextmanager
        async def fixture(*args,**kwargs):
            yield SimpleNamespace(owner="direct-image-eval-second-brain-synthetic-a",pool=pool)
        commands = []
        def execute(arguments,**kwargs):
            commands.append(arguments)
            if "logs" in arguments and logs_fail:
                raise subprocess.TimeoutExpired(arguments,20)
            status = 1 if "up" in arguments or ("down" in arguments and down_fail) else 0
            return subprocess.CompletedProcess(arguments,status,stdout="",stderr="")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)/"jarvis-s04-frontier-deployment-failure"
            with patch("evals.second_brain.frontier_deployment_trace.source_fixture",fixture), patch(
                "evals.second_brain.frontier_deployment_trace.JarvisSettings",
                return_value=SimpleNamespace(SUPABASE_DB_URL="postgresql://synthetic@localhost/db",OPENAI_API_KEY="synthetic-key"),
            ), patch("evals.second_brain.frontier_deployment_trace.subprocess.run",side_effect=execute):
                with self.assertRaises(RuntimeError):
                    await run(root,"synthetic-image")
            report = json.loads((root/"deployment-resources.json").read_text())
            self.assertTrue(any("up" in command for command in commands))
            self.assertTrue(any("down" in command for command in commands))
            self.assertFalse((root/"secrets").exists())
            self.assertEqual(report["status"],"cleanup_failed" if down_fail else "cleaned")
            return report

    async def test_partial_startup_failure_still_tears_down_owned_resources(self):
        await self.exercise()

    async def test_log_capture_failure_does_not_skip_teardown(self):
        report = await self.exercise(logs_fail=True)
        self.assertEqual(report["log_capture"],"failed")

    async def test_failed_teardown_is_not_recorded_as_successful_cleanup(self):
        await self.exercise(down_fail=True)
