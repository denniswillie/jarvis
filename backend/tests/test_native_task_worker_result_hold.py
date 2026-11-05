"""Supervisor opt-in with existing fake child/SQL fixture, never a real worker."""
import json
import os
import unittest
from unittest.mock import patch

import test_native_task_worker as fixture_tests
from evals.responsive_voice.native_task_release_control import load_config
from evals.responsive_voice.native_task_worker import NativeTaskWorkerOptions


class NativeTaskWorkerResultHoldTests(unittest.IsolatedAsyncioTestCase):
    async def fixture(self):
        fixture = fixture_tests.NativeTaskWorkerTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        self.addAsyncCleanup(fixture.asyncTearDown)
        return fixture

    async def test_disabled_retains_exact_ordinary_cli_and_no_control_environment(self):
        fixture = await self.fixture()
        with patch.dict(os.environ, {"JARVIS_NATIVE_RESULT_HOLD_CONFIG": "/private/foreign/config.json"}):
            instance = await fixture.start()
        self.assertEqual(fixture.launch_arguments, (str(fixture.options.worker_python), "-m", "app.tasks.worker", "run"))
        self.assertNotIn("JARVIS_NATIVE_RESULT_HOLD_CONFIG", fixture.launch_options["env"])
        self.assertNotIn("result_hold_control", instance.readiness_receipt)
        self.assertFalse((instance.root / "result-control").exists())

    async def test_enabled_launches_explicit_entrypoint_and_binds_private_config(self):
        fixture = await self.fixture()
        fixture.options = fixture.options.model_copy(update={"result_hold_seconds": 15.0})
        instance = await fixture.start()
        self.assertEqual(fixture.launch_arguments, (str(fixture.options.worker_python), "-m",
                                                   "evals.responsive_voice.native_task_worker_entry"))
        env = fixture.launch_options["env"]
        config = load_config(env["JARVIS_NATIVE_RESULT_HOLD_CONFIG"])
        self.assertEqual(config.scope.run_id.hex, fixture.run)
        self.assertEqual(config.scope.owner_id, fixture.owner)
        self.assertEqual(config.scope.database_schema, fixture.scope["database_schema"])
        self.assertEqual(config.scope.worker_host_id, instance.host_id)
        self.assertEqual(config.worker_root, instance.root.resolve())
        self.assertEqual(config.scope.expires_at, instance.stop_at)
        self.assertEqual(config.max_hold_seconds, 15)
        self.assertEqual(instance.readiness_receipt["result_hold_control"], config.readiness())
        manifest = json.loads(instance.resource_manifest.read_text())
        self.assertEqual(manifest["result_hold_control"], config.readiness())
        self.assertNotIn("fixture-private", config.model_dump_json())
        self.assertEqual((await instance.close())["status"], "closed")
        self.assertTrue(config.config_path.exists())

    async def test_optional_budget_rejects_bool_nonfinite_and_out_of_range(self):
        fixture = await self.fixture()
        base = fixture.options.model_dump()
        for value in (True, False, 0, -1, 61, float("inf"), float("nan"), "5"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    NativeTaskWorkerOptions(**{**base, "result_hold_seconds": value})
        for value in (None, .1, 60):
            with self.subTest(value=value):
                options = NativeTaskWorkerOptions(**{**base, "result_hold_seconds": value})
                self.assertEqual(options.result_hold_seconds, value)
        fixture.launch_mock.assert_not_awaited()
