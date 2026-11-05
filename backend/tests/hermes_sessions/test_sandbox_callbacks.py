import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.hermes_sessions.sandbox_callbacks import BACKGROUND_INSTRUCTIONS, SandboxCallbacks
from app.sandbox.models import SandboxError


class CallbackInstallationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        scope = SimpleNamespace(logical_session_id="session-one", allocation_id="allocation-one")
        self.handle = SimpleNamespace(sandbox_id="sandbox-one", scope=scope)
        self.events = SimpleNamespace(_resource=lambda session_id: self.handle,
                                      credential=lambda session_id: "fixture-callback-secret")
        self.installer = SandboxCallbacks(events=self.events,
            endpoint="https://callback.example/hermes/background-wake")
        self.staged, self.commands = [], []

        async def stage(handle, identity, content):
            self.staged.append((handle, identity, content))
            return {"path": "/home/jarvis/inputs/" + identity}

        async def start(handle, command, **kwargs):
            self.commands.append(command)
            return SimpleNamespace(wait=AsyncMock(return_value=SimpleNamespace(
                exit_code=0, stdout="jarvis.callback.ready.v1")))

        self.provider = SimpleNamespace(stage_input=stage, start=start)

    async def test_stages_bytes_not_secret_shell_arguments_and_caches_success(self):
        await self.installer.install(self.provider, self.handle)
        self.assertEqual(len(self.staged), 2)
        config = json.loads(self.staged[1][2])
        self.assertEqual(config["session_id"], "session-one")
        self.assertEqual(config["token"], "fixture-callback-secret")
        self.assertNotIn(config["token"], self.commands[0])
        self.assertNotIn(config["endpoint"], self.commands[0])
        for _, identity, content in self.staged:
            self.assertEqual(identity, hashlib.sha256(content).hexdigest())
        await self.installer.install(self.provider, self.handle)
        self.assertEqual((len(self.staged), len(self.commands)), (2, 1))

    async def test_rotated_credential_updates_config_without_relaunching_user_program(self):
        await self.installer.install(self.provider, self.handle)
        self.events.credential = lambda session_id: "rotated-callback-secret"
        await self.installer.install(self.provider, self.handle)
        self.assertNotEqual(self.staged[1][1], self.staged[3][1])
        self.assertEqual(self.staged[0][1], self.staged[2][1])
        self.assertNotIn("rotated-callback-secret", self.commands[1])

    async def test_wrong_resource_never_receives_config(self):
        wrong = SimpleNamespace(sandbox_id="sandbox-other", scope=self.handle.scope)
        with self.assertRaisesRegex(SandboxError, "scope_mismatch"):
            await self.installer.install(self.provider, wrong)
        self.assertEqual(self.staged, [])

    async def test_failed_setup_is_not_cached(self):
        start = self.provider.start
        self.provider.start = AsyncMock(side_effect=OSError("lost setup acknowledgement"))
        with self.assertRaises(OSError):
            await self.installer.install(self.provider, self.handle)
        self.provider.start = start
        await self.installer.install(self.provider, self.handle)
        self.assertEqual(self.staged[1][1], self.staged[3][1])
        self.assertEqual(len(self.commands), 1)

    def test_endpoint_and_instructions_do_not_grant_provider_access(self):
        for endpoint in ("http://example/hermes/background-wake", "https://example/other",
                         "https://example/hermes/background-wake?token=secret"):
            with self.assertRaises(ValueError):
                SandboxCallbacks(events=self.events, endpoint=endpoint)
        self.assertIn("same conversation", BACKGROUND_INSTRUCTIONS)
        self.assertIn("inspect existing", BACKGROUND_INSTRUCTIONS)
        self.assertNotIn("fixture-callback-secret", BACKGROUND_INSTRUCTIONS)


if __name__ == "__main__":
    unittest.main()
