import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

from app.hermes_sessions.runtime import RetainedProfileRuntime
from app.sandbox.models import SandboxPolicy
from test_sandbox_keepalive import ClockProvider
from test_service import NativeFixture


class RuntimeNative(NativeFixture):
    change_sequence = 0

    async def __aenter__(self):
        self.connection_generation += 1
        return self

    async def __aexit__(self, *args):
        pass

    async def wait_for_change(self, sequence):
        await asyncio.Event().wait()


class ProfileRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-profile-runtime-")
        self.home = Path(self.temp.name).resolve()
        (self.home / "jarvis-adapter.json").write_text(json.dumps({"owner_id": "owner",
            "launch": {"source": "cli", "cwd": str(self.home), "toolsets": []}}))
        self.native = RuntimeNative(self.home)
        self.provider = ClockProvider(time.time)
        self.running = []

    async def asyncTearDown(self):
        for runtime, stop, task in self.running:
            stop.set()
            await asyncio.wait_for(task, timeout=5)
        self.temp.cleanup()

    async def start(self):
        runtime = RetainedProfileRuntime(self.home, provider=self.provider,
            sandbox_policy=SandboxPolicy(), callback_endpoint="https://callback.example/hermes/background-wake",
            signing_key=b"test-profile-runtime-signing-key-32", client_factory=lambda home: self.native)
        stop = asyncio.Event()
        task = asyncio.create_task(runtime.run(stop))
        self.running.append((runtime, stop, task))
        await runtime.wait_ready(timeout=5)
        return runtime, stop, task

    async def test_empty_start_is_ready_without_model_input_or_sandbox(self):
        runtime, stop, task = await self.start()
        self.assertIsNotNone(runtime.current_events())
        self.assertEqual(self.native.inputs, {})
        self.assertEqual(self.provider.creates, 0)
        self.assertIs(runtime.transport.terminal, runtime.terminal)
        self.assertIsNotNone(runtime.transport.sandbox_controls)
        await runtime.send(input_id="user-one", text="Hello")
        self.assertEqual(len(self.native.inputs), 1)
        stop.set()
        await task
        self.assertIsNone(runtime.current_events())
        self.assertEqual((self.provider.pauses, self.provider.resumes), (0, 0))

    async def test_restart_notifies_preexisting_sandbox_owner_once_with_same_binding(self):
        first, stop, task = await self.start()
        child = await first.bridge.service.create_work(first.master.logical_id, request_id="child", label="Child")
        resource = await first.sandboxes.ensure(child.logical_id)
        token = first.events.credential(child.logical_id)
        stop.set()
        await task
        second, _, _ = await self.start()
        # Startup pass and explicit retry share one recovery identity.
        await second.retry_recovery()
        self.assertEqual(len(self.native.inputs), 1)
        (native_id, _), (text, _) = next(iter(self.native.inputs.items()))
        self.assertEqual(native_id, child.native_id)
        self.assertIn("backend_restarted", text)
        self.assertEqual(second.resources.get(child.logical_id).sandbox_id, resource.sandbox_id)
        self.assertEqual(second.events.credential(child.logical_id), token)
        self.assertEqual(second.recovery_errors, {})
        self.assertEqual(self.provider.creates, 1)
        self.assertEqual((self.provider.pauses, self.provider.resumes), (0, 0))

    async def test_shutdown_does_not_delete_native_history(self):
        runtime, stop, task = await self.start()
        await runtime.send(input_id="retained", text="Remember the original input")
        stop.set()
        await task
        self.assertEqual(len(self.native.inputs), 1)
        self.assertIn("Remember the original input", next(iter(self.native.inputs.values()))[0])


if __name__ == "__main__":
    unittest.main()
