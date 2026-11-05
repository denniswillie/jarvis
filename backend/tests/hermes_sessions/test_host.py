"""Continuous delivery-owner lifecycle, without executing models or sandboxes."""

import asyncio
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from app.hermes_sessions.client import HermesGatewayError
from app.hermes_sessions.host import HostAlreadyRunning, RetainedProfileHost


class FakeClient:
    def __init__(self, home):
        self.profile_home = home
        self.change_sequence = 0
        self.closed = False
        self.broken = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    async def wait_for_change(self, _after):
        await self.broken.wait()
        raise HermesGatewayError("connection_lost")


class FakeBridge:
    def __init__(self):
        self.pumps = 0
        self.inputs = []
        self.master_calls = 0
        self.failure = None
        self.service = self

    async def master(self):
        self.master_calls += 1
        return SimpleNamespace(logical_id="retained-master")

    async def send(self, session_id, *, input_id, text):
        self.inputs.append((session_id, input_id, text))
        return {"status": "queued"}

    async def pump(self):
        self.pumps += 1
        if self.failure:
            failure, self.failure = self.failure, None
            raise failure
        return []


class HostTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jarvis-profile-host-")
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name).resolve()
        self.clients = []
        self.bridge = FakeBridge()
        self.stop = asyncio.Event()
        self.tasks = []

    async def asyncTearDown(self):
        self.stop.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    def client(self, home):
        client = FakeClient(home)
        self.clients.append(client)
        return client

    def make_host(self):
        return RetainedProfileHost(self.home, poll_seconds=0.02, retry_seconds=0.02,
                                   client_factory=self.client, bridge_factory=lambda *_: self.bridge)

    async def start(self, host):
        task = asyncio.create_task(host.run(self.stop))
        self.tasks.append(task)
        await host.wait_ready(timeout=1)
        return task

    async def test_idle_host_never_creates_master_or_model_input(self):
        host = self.make_host()
        await self.start(host)
        await asyncio.sleep(0.06)
        self.assertGreater(self.bridge.pumps, 1)
        self.assertEqual(self.bridge.master_calls, 0)
        self.assertEqual(self.bridge.inputs, [])

    async def test_send_admits_original_input_without_waiting_for_execution(self):
        host = self.make_host()
        await self.start(host)
        result = await host.send(input_id="source-message", text="Original user words")
        self.assertEqual(result, {"status": "queued"})
        self.assertEqual(self.bridge.inputs, [("retained-master", "source-message", "Original user words")])

    async def test_reconnect_preserves_delivery_owner_and_does_not_resubmit_user(self):
        host = self.make_host()
        await self.start(host)
        await host.send(input_id="one", text="Only once")
        self.clients[0].broken.set()
        async with asyncio.timeout(1):
            while host.connections < 2:
                await asyncio.sleep(0.01)
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(len(self.bridge.inputs), 1)

    async def test_stop_closes_transport_without_cancelling_native_work(self):
        host = self.make_host()
        task = await self.start(host)
        self.stop.set()
        await asyncio.wait_for(task, timeout=1)
        self.assertTrue(self.clients[0].closed)
        self.assertEqual(self.bridge.inputs, [])

    async def test_single_local_owner_and_release_on_shutdown(self):
        first = self.make_host()
        task = await self.start(first)
        with self.assertRaises(HostAlreadyRunning):
            await self.make_host().run(self.stop)
        self.stop.set()
        await task
        self.stop = asyncio.Event()
        await self.start(self.make_host())

    async def test_storage_failure_reconnects_without_new_model_turn(self):
        self.bridge.failure = OSError("private path must not enter status")
        host = self.make_host()
        task = asyncio.create_task(host.run(self.stop))
        self.tasks.append(task)
        async with asyncio.timeout(1):
            while host.connections < 2:
                await asyncio.sleep(0.01)
        self.assertEqual(self.bridge.inputs, [])
        self.assertIsNone(host.last_error)

    async def test_not_ready_does_not_start_a_gateway_or_create_input(self):
        host = self.make_host()
        with self.assertRaises(TimeoutError):
            await host.wait_ready(timeout=0.02)
        self.assertEqual(self.clients, [])
        self.assertEqual(self.bridge.inputs, [])


if __name__ == "__main__":
    unittest.main()
