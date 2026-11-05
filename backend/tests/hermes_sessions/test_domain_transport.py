import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from app.hermes_sessions.domain_transport import (
    DomainToolServer, FILE_TOOL_TIMEOUT_SECONDS, invoke_domain_tool, write_domain_catalog,
)


class DomainTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.home = Path(self.temporary.name).resolve()
        self.adapter = SimpleNamespace(invoke=AsyncMock(return_value=SimpleNamespace(
            payload={"status": "success", "value": "source-backed fixture"}, image=None)))
        self.server = await DomainToolServer(self.home, self.adapter).start()

    async def asyncTearDown(self):
        await self.server.close()
        self.temporary.cleanup()

    async def call(self, **changes):
        kwargs = dict(native_session_id="native", input_id="input", admission_id="admission",
                      call_id="call", name="search_memory", arguments={"query": "fixture"})
        kwargs.update(changes)
        return await asyncio.to_thread(invoke_domain_tool, self.home, **kwargs)

    async def test_private_roundtrip_keeps_model_arguments_separate(self):
        result = await self.call(arguments={"owner_id": "not-authority", "query": "fixture"})
        self.assertEqual(result["status"], "success")
        received = self.adapter.invoke.await_args.kwargs
        self.assertEqual(received["input_id"], "input")
        self.assertNotIn("owner_id", received)
        self.assertEqual(received["arguments"]["owner_id"], "not-authority")
        self.assertEqual(self.server.socket_path.stat().st_mode & 0o777, 0o600)

    async def test_different_profile_descriptor_is_refused_before_invoke(self):
        path = self.home / "jarvis/domain-api.json"
        descriptor = json.loads(path.read_text())
        descriptor["profile_home"] = "/other/profile"
        path.write_text(json.dumps(descriptor))
        self.assertEqual((await self.call())["error"]["code"], "domain_protocol_error")
        self.adapter.invoke.assert_not_called()

    async def test_unsafe_descriptor_permissions_are_rejected(self):
        (self.home / "jarvis/domain-api.json").chmod(0o644)
        self.assertEqual((await self.call())["error"]["code"], "domain_protocol_error")
        self.adapter.invoke.assert_not_called()

    async def test_no_second_delivery_owner_and_no_descriptor_damage(self):
        other = DomainToolServer(self.home, self.adapter)
        with self.assertRaises(BlockingIOError):
            await other.start()
        await other.close()
        self.assertEqual((await self.call())["status"], "success")

    async def test_missing_scope_is_bounded_and_no_exception_detail_leaks(self):
        self.adapter.invoke.side_effect = ValueError("private source path and content")
        result = await self.call()
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("private source", json.dumps(result))
        self.adapter.invoke.assert_awaited_once()

    async def test_no_automatic_retry_on_transport_loss(self):
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.adapter.invoke.side_effect = blocked
        pending = asyncio.create_task(self.call())
        await entered.wait()
        async with asyncio.timeout(2):
            await self.server.close()
        result = await pending
        self.assertEqual(result["error"]["code"], "domain_transport_unavailable")
        self.adapter.invoke.assert_awaited_once()

    async def test_deadline_and_capacity_do_not_make_duplicate_calls(self):
        await self.server.close()
        self.server = await DomainToolServer(self.home, self.adapter, max_in_flight=1, timeout=.15).start()
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.adapter.invoke.side_effect = blocked
        first = asyncio.create_task(self.call())
        await entered.wait()
        second = await self.call(call_id="second")
        self.assertEqual(second["error"]["code"], "domain_service_busy")
        self.assertEqual((await first)["error"]["code"], "domain_request_timeout")
        self.adapter.invoke.assert_awaited_once()

    async def test_only_file_tools_receive_the_extended_client_default(self):
        self.assertEqual(FILE_TOOL_TIMEOUT_SECONDS, 900)
        with patch("app.hermes_sessions.domain_transport._invoke_private",
                   return_value={"status": "success"}) as invoke:
            for name, expected in (("sandbox_stage_file", 905), ("sandbox_publish_file", 905),
                                   ("search_memory", 65), ("sandbox_view_image", 65)):
                with self.subTest(name=name):
                    await self.call(name=name, arguments={"name": "sandbox_publish_file"})
                    self.assertEqual(invoke.call_args.kwargs["timeout"], expected)
        self.adapter.invoke.assert_not_awaited()

    async def test_client_deadline_ceiling_is_finite_and_file_only(self):
        for name, ceiling in (("sandbox_stage_file", 905), ("sandbox_publish_file", 905),
                              ("search_memory", 180)):
            with self.subTest(name=name):
                self.assertEqual((await self.call(name=name, timeout=ceiling))["status"], "success")
                self.adapter.invoke.reset_mock()
                for invalid in (ceiling + 1, 0, float("inf"), float("nan")):
                    result = await self.call(name=name, timeout=invalid)
                    self.assertEqual(result["error"]["code"], "domain_protocol_error")
                self.adapter.invoke.assert_not_awaited()

    async def test_file_tools_outlive_control_deadline_without_extending_other_tools(self):
        await self.server.close()
        self.server = await DomainToolServer(self.home, self.adapter, timeout=.03).start()
        deadlines, remaining = [], []
        original_timeout = asyncio.timeout
        outcome = self.adapter.invoke.return_value

        def tracked_timeout(delay):
            deadline = original_timeout(delay)
            deadlines.append(deadline)
            return deadline

        async def delayed(**kwargs):
            remaining.append(deadlines[-1].when() - asyncio.get_running_loop().time())
            await asyncio.sleep(.06)
            return outcome

        self.adapter.invoke.side_effect = delayed
        with patch("app.hermes_sessions.domain_transport.asyncio.timeout", side_effect=tracked_timeout):
            for name in ("sandbox_stage_file", "sandbox_publish_file"):
                self.assertEqual((await self.call(name=name))["status"], "success")
                self.assertGreater(remaining[-1], 899)
                self.assertLessEqual(remaining[-1], 900)
            result = await self.call(arguments={"name": "sandbox_publish_file"})
            self.assertEqual(result["error"]["code"], "domain_request_timeout")
            self.assertLessEqual(remaining[-1], .03)
        self.assertEqual(self.adapter.invoke.await_count, 3)

    async def test_file_deadline_still_expires_without_retry(self):
        async def blocked(**kwargs):
            await asyncio.Event().wait()

        self.adapter.invoke.side_effect = blocked
        with patch("app.hermes_sessions.domain_transport.FILE_TOOL_TIMEOUT_SECONDS", .03):
            result = await self.call(name="sandbox_publish_file")
        self.assertEqual(result["error"]["code"], "domain_request_timeout")
        self.adapter.invoke.assert_awaited_once()

    async def test_incomplete_file_frame_keeps_short_ingress_deadline(self):
        await self.server.close()
        self.server = await DomainToolServer(self.home, self.adapter, timeout=.03).start()
        reader, writer = await asyncio.open_unix_connection(str(self.server.socket_path))
        try:
            writer.write(b'{"name":"sandbox_publish_file"')
            await writer.drain()
            async with asyncio.timeout(1):
                reply = json.loads(await reader.readline())
            self.assertEqual(reply["payload"]["error"]["code"], "domain_request_timeout")
            self.adapter.invoke.assert_not_awaited()
        finally:
            writer.close()
            await writer.wait_closed()

    async def test_unsupported_images_are_not_claimed_as_seen(self):
        self.adapter.invoke.return_value.image = object()
        result = await self.call()
        self.assertEqual(result["error"]["code"], "domain_image_transport_unavailable")

    async def test_catalog_is_derived_schema_only(self):
        tools = [{"name": "read", "description": "Read a source.", "parameters": {"type": "object"}}]
        write_domain_catalog(self.home, tools)
        self.assertEqual(json.loads((self.home / "jarvis/domain-tools.json").read_text()),
                         {"version": 1, "tools": tools})
        with self.assertRaises(ValueError):
            write_domain_catalog(self.home, tools * 2)

    async def test_protocol_does_not_accept_extra_scope_fields(self):
        reader, writer = await asyncio.open_unix_connection(str(self.server.socket_path))
        try:
            writer.write(json.dumps(dict(version=1, native_session_id="native", input_id="input",
                admission_id="admission", call_id="call", name="read", arguments={}, owner_id="other")).encode() + b"\n")
            await writer.drain()
            result = json.loads(await reader.readline())
            self.assertEqual(result["payload"]["error"]["code"], "domain_protocol_error")
            self.adapter.invoke.assert_not_called()
        finally:
            writer.close()
            await writer.wait_closed()


if __name__ == "__main__":
    unittest.main()
