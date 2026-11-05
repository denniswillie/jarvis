"""Portable transport tests; native integration is the separate gateway probe."""

import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from websockets.asyncio.server import serve

from app.hermes_sessions.client import HermesGatewayClient, HermesGatewayError, HermesOutcomeUnknown


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="jh0-client-")
        self.home = Path(self.temp.name).resolve()
        self.requests = []
        self.mode = "normal"
        self.socket = None
        self.server = await serve(self.peer, "127.0.0.1", 0, subprotocols=["hermes-gateway-v1"])
        self.port = self.server.sockets[0].getsockname()[1]
        self.control = await asyncio.start_unix_server(self.bootstrap, path=str(self.home / "gateway.sock"))
        self.client = HermesGatewayClient(self.home, timeout=0.5)

    async def asyncTearDown(self):
        await self.client.close()
        self.control.close()
        await self.control.wait_closed()
        self.server.close()
        await self.server.wait_closed()
        self.temp.cleanup()

    async def bootstrap(self, reader, writer):
        request = json.loads(await reader.readline())
        if request["verb"] == "identify":
            result = {"state": "ready", "instance_id": "fixture", "api_origin": f"http://127.0.0.1:{self.port}",
                      "capabilities": ["session-authority-v1"], "served_profiles": [{"profile_id": str(self.home)}]}
        else:
            self.assertEqual(request["params"]["profile_id"], str(self.home))
            self.assertEqual(request["params"]["instance_id"], "fixture")
            result = {"ticket": "fixture-ticket"}
        writer.write(json.dumps({"ok": True, "result": result}).encode() + b"\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def peer(self, socket):
        self.socket = socket
        async for wire in socket:
            request = json.loads(wire)
            self.requests.append(request)
            if self.mode == "silent":
                continue
            if self.mode == "disconnect":
                await socket.close()
                return
            if self.mode == "error":
                reply = {"id": request["id"], "error": {"message": "permission_denied"}}
            else:
                await socket.send(json.dumps({"method": "session.event", "params": {"seq": 1}}))
                reply = {"id": request["id"], "result": {"status": "queued", "echo": request["params"]}}
            await socket.send(json.dumps(reply))

    async def test_original_input_and_native_identity_are_unchanged(self):
        await self.client.open()
        text = 'User: use the earlier file.\nVoice: which one?\nUser: "the red one".'
        result = await self.client.submit("native-master", "original-event-id", text)
        self.assertEqual(result["echo"], {"session_id": "native-master", "input_id": "original-event-id", "text": text})
        self.assertEqual(len(self.requests), 1)

    async def test_notifications_do_not_require_a_payload_queue(self):
        await self.client.open()
        before = self.client.change_sequence
        await self.client.snapshot("master")
        changed = await self.client.wait_for_change(before, timeout=1)
        self.assertGreater(changed, before)
        self.assertEqual(len(self.client._pending), 0)

    async def test_native_refusal_is_not_retried(self):
        self.mode = "error"
        await self.client.open()
        with self.assertRaises(HermesGatewayError) as raised:
            await self.client.submit("master", "event", "hello")
        self.assertNotIsInstance(raised.exception, HermesOutcomeUnknown)
        self.assertEqual(raised.exception.reason, "permission_denied")
        self.assertEqual(len(self.requests), 1)

    async def test_submit_timeout_is_unknown_and_never_retried(self):
        self.mode = "silent"
        await self.client.open()
        with self.assertRaises(HermesOutcomeUnknown) as raised:
            await self.client.submit("master", "retain-this-id", "hello")
        self.assertEqual(raised.exception.request_id, "retain-this-id")
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(self.client._pending)

    async def test_disconnect_after_send_is_unknown(self):
        self.mode = "disconnect"
        await self.client.open()
        with self.assertRaises(HermesOutcomeUnknown):
            await self.client.submit("master", "event", "hello")
        self.assertEqual(len(self.requests), 1)

    async def test_slow_request_does_not_block_another_reply(self):
        self.mode = "silent"
        await self.client.open()
        first = asyncio.create_task(self.client.snapshot("slow"))
        while not self.requests:
            await asyncio.sleep(0)
        self.mode = "normal"
        fast = await self.client.snapshot("fast")
        self.assertEqual(fast["echo"]["session_id"], "fast")
        self.assertFalse(first.done())
        with self.assertRaises(HermesGatewayError):
            await first

    async def test_remote_descriptor_and_destructive_operation_are_refused(self):
        for origin in ("https://example.com", "http://example.com:9000", "http://127.0.0.1:9000/other",
                       "http://user:secret@127.0.0.1:9000", "http://127.0.0.1:9000?query=1"):
            with self.assertRaises(HermesGatewayError):
                self.client._websocket_url(origin)
        with self.assertRaises(ValueError):
            await self.client.request("session.delete", session_id="master")
        with self.assertRaises(ValueError):
            await self.client.request("prompt.submit", text="missing identity")

    async def test_redirect_uses_native_generation_without_local_id_on_wire(self):
        await self.client.open()
        result = await self.client.request("session.redirect", _operation_id="followup",
            session_id="master", execution_generation=3, text="Open the first result instead")
        self.assertEqual(result["echo"], {"session_id": "master", "execution_generation": 3,
                                           "text": "Open the first result instead"})

    async def test_redirect_timeout_is_unknown_not_retried(self):
        self.mode = "silent"
        await self.client.open()
        with self.assertRaises(HermesOutcomeUnknown):
            await self.client.request("session.redirect", _operation_id="followup",
                session_id="master", execution_generation=3, text="Different page please")
        self.assertEqual(len(self.requests), 1)

    async def test_direct_steer_uses_the_native_control_contract(self):
        await self.client.open()
        result = await self.client.request("session.steer", _operation_id="followup",
            session_id="master", execution_generation=3, text="Also check the date")
        self.assertEqual(result["echo"], {"session_id": "master", "execution_generation": 3,
                                         "text": "Also check the date"})


if __name__ == "__main__":
    unittest.main()
