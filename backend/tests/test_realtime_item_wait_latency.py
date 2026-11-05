"""Owned item ACK timing, not output/hearing or fabricated provider acceptance."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.realtime.openai_client import OpenAIRealtimeConnection
from app.realtime.telemetry import RealtimeTrace
from app.voice_routes.realtime_adapter import RealtimeVoiceRoute, _ConversationItem


class RealtimeItemWaitLatencyTests(unittest.IsolatedAsyncioTestCase):
    def make_route(self):
        trace = RealtimeTrace(provider="openai", audio_session_id=str(uuid4()), jarvis_session_id=uuid4())
        provider = OpenAIRealtimeConnection(
            settings=SimpleNamespace(OPENAI_REALTIME_WRITE_TIMEOUT_SECONDS=1), owner_id="fixture")
        provider.bind_trace(trace)
        route = object.__new__(RealtimeVoiceRoute)
        route._bindings = SimpleNamespace(provider=provider, audio_session_id=trace.context["audio_session_id"], identity=object())
        route._closed = False
        route._io_timeout = 1
        route._tasks, route._data_tasks = set(), set()
        route._items = {"owned-item": _ConversationItem("owned-item", "owned-create", state="created")}
        sent = []

        async def send(payload):
            sent.append(json.loads(payload))

        provider._websocket = SimpleNamespace(send=send)
        return route, sent

    async def wait_started(self, records, phase):
        async with asyncio.timeout(1):
            while not any(r.get("event") == "provider_item_latency" and r.get("phase") == phase
                          and r.get("status") == "started" for r in records):
                await asyncio.sleep(0)

    async def test_delete_wait_ends_only_after_exact_owned_ack(self):
        route, sent = self.make_route()
        records = []
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=lambda _, data: records.append(json.loads(data))):
            task = asyncio.create_task(route._delete_owned_item("owned-item", current=lambda: True))
            try:
                await self.wait_started(records, "delete_ack_wait")
                self.assertEqual(len(sent), 1)
                route.observe_item_event(route.identity, {"type": "conversation.item.deleted", "item_id": "foreign"})
                self.assertFalse(task.done())
                route.observe_item_event(route.identity, {"type": "conversation.item.deleted", "item_id": "owned-item"})
                await task
                self.assertNotIn("owned-item", route._items)
            finally:
                task.cancel()
                await asyncio.gather(task, *tuple(route._tasks), return_exceptions=True)
        waits = [r for r in records if r.get("event") == "provider_item_latency"]
        self.assertEqual([r["status"] for r in waits], ["started", "completed"])
        self.assertEqual(waits[-1]["reason"], "deleted")
        self.assertEqual(waits[-1]["event_id"], sent[0]["event_id"])
        self.assertEqual(waits[-1]["item_id"], "owned-item")
        self.assertGreaterEqual(waits[-1]["duration_ms"], 0)

    async def test_cancelled_owned_ack_wait_keeps_unconfirmed_state(self):
        route, sent = self.make_route()
        records = []
        with patch("app.realtime.telemetry.LOGGER.info", side_effect=lambda _, data: records.append(json.loads(data))):
            task = asyncio.create_task(route._delete_owned_item("owned-item", current=lambda: True))
            try:
                await self.wait_started(records, "delete_ack_wait")
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(route._items["owned-item"].state, "delete_pending")
                self.assertEqual(len(sent), 1)
            finally:
                task.cancel()
                await asyncio.gather(task, *tuple(route._tasks), return_exceptions=True)
        terminal = [r for r in records if r.get("event") == "provider_item_latency"][-1]
        self.assertEqual((terminal["status"], terminal["reason"]), ("cancelled", "delete_pending"))
        self.assertNotIn("error_type", terminal)


if __name__ == "__main__":
    unittest.main()
