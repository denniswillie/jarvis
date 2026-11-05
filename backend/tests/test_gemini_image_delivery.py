from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4
import asyncio

from app.memory.tools import MemoryToolModule
from app.perception.current_image import CurrentImageService
from app.perception.frame_window import RecentVisualFrameWindow, SensoryVisualFrame
from app.perception.tools import CurrentImageToolModule
from app.realtime.gemini.session import GeminiAudioSession
from app.tools.module import FunctionToolModule, ToolBuildContext
from test_current_image_service import _jpeg
from test_gemini_audio_session import ENCODED, Phone, Wire, until
from test_gemini_domain_integration import ExecutionRepository, Experience, MessageRepository, context, settings
from test_historical_image_service import _evidence, _service


class GeminiImageDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context, self.settings = context(), settings()
        self.phone, self.wires = Phone(), []
        self.messages, self.executions = MessageRepository(), ExecutionRepository()
        self.window = RecentVisualFrameWindow(max_frames_per_session=4, ttl_seconds=10)
        self.lease = self.window.begin_stream(self.context.owner_id, self.context.jarvis_session_id, uuid4())
        now = datetime.now(UTC)
        self.frame = SensoryVisualFrame(
            frame_id=uuid4(), owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id,
            audio_session_id=self.context.audio_session_id, visual_session_id=self.lease.visual_session_id,
            frame_sequence=1, source="glasses", captured_at=now - timedelta(seconds=.25),
            server_received_at=now, content_type="image/jpeg", width=48, height=32,
            storage_bucket="fixture", storage_path="fixture/current.jpg",
            raw_image_expires_at=now + timedelta(minutes=1), image_bytes=_jpeg(),
        )
        self.window.put(self.frame, stream_lease=self.lease)
        current = CurrentImageService(frame_window=self.window, max_age_seconds=10,
                                      max_image_bytes=512000, max_width=1024, max_height=1024)
        self.historical = _evidence(owner_id=self.context.owner_id)
        self.evidence_repository = SimpleNamespace(get_image_evidence=AsyncMock(return_value=self.historical))
        self.loader = SimpleNamespace(download_object=AsyncMock(return_value=_jpeg()))
        historical_module = MemoryToolModule(
            repository=SimpleNamespace(), entity_repository=SimpleNamespace(),
            image_service=_service(self.evidence_repository, self.loader),
        )
        historical_tool = next(tool for tool in historical_module.build_tools(ToolBuildContext(
            self.settings, self.context.owner_id, self.context.jarvis_session_id,
        )) if tool.name == "get_memory_image")

        def factory(**kwargs):
            wire = Wire(**kwargs)
            self.wires.append(wire)
            return wire

        self.session = GeminiAudioSession(
            websocket=self.phone, context=self.context, settings=self.settings,
            repository=self.messages, tool_execution_repository=self.executions,
            experience_event_service=Experience(self.messages), connection_factory=factory,
            tool_modules=(CurrentImageToolModule(current), FunctionToolModule("historical", lambda _: [historical_tool])),
        )
        self.run = asyncio.create_task(self.session.run())
        await until(lambda: any(event["type"] == "session.created" for event in self.phone.events))

    async def asyncTearDown(self):
        await self.session.close("image_test_done")
        await self.run

    async def turn(self, segment="a"):
        await self.phone.send("audio.input.append", segment_id=segment, audio_base64=ENCODED)
        await until(lambda: segment in self.session._turns)
        await self.phone.send("audio.input.commit", segment_id=segment)
        await until(lambda: self.session._turns[segment].released)

    def images(self):
        return [part["inlineData"] for wire in self.wires for event in wire.sent
                for turn in event.get("clientContent", {}).get("turns", [])
                for part in turn.get("parts", []) if "inlineData" in part]

    def payloads(self):
        return [json.loads(event["realtimeInput"]["text"]) for wire in self.wires for event in wire.sent
                if "text" in event.get("realtimeInput", {})]

    def responses(self):
        return [response for wire in self.wires for event in wire.sent
                for response in event.get("toolResponse", {}).get("functionResponses", [])]

    async def request(self, name, args):
        await self.wires[-1].input.put({"toolCall": {"functionCalls": [{"id": "image-call", "name": name, "args": args}]}})
        await until(lambda: bool(self.responses()))
        return self.responses()[-1]

    async def test_current_pixels_and_timing_precede_end_marker(self):
        await self.turn()
        events = [next(iter(event["realtimeInput"])) if "realtimeInput" in event else "image"
                  for event in self.wires[0].sent]
        self.assertLess(events.index("image"), events.index("text"))
        self.assertLess(events.index("text"), events.index("activityEnd"))
        image_event = next(event["clientContent"] for event in self.wires[0].sent if "clientContent" in event)
        self.assertFalse(image_event["turnComplete"])
        self.assertEqual(base64.b64decode(self.images()[0]["data"]), self.frame.image_bytes)
        self.assertEqual(self.payloads()[0]["live_view"]["image"]["captured_at"], self.frame.captured_at.isoformat())
        self.assertEqual(self.session._turns["a"].evidence_status, "available")

    async def test_reuses_same_fresh_frame_without_a_second_image_write(self):
        await self.turn()
        await self.wires[0].publish({"generationComplete": True, "turnComplete": True})
        await until(lambda: self.session._turns["a"].output.completed)
        await self.turn("b")
        self.assertEqual(len(self.images()), 1)
        self.assertEqual(self.payloads()[-1]["live_view"]["status"], "reused")

    async def test_stream_replacement_after_preparation_rejects_pixels_at_writer(self):
        original = self.session._write_prepared_context

        async def replace_stream(turn, prepared):
            self.window.begin_stream(self.context.owner_id, self.context.jarvis_session_id, self.lease.visual_session_id)
            return await original(turn, prepared)

        self.session._write_prepared_context = replace_stream
        await self.turn()
        self.assertEqual(self.images(), [])
        self.assertEqual(self.payloads()[-1]["current_visual_evidence"], "unavailable")
        self.assertEqual(self.session._turns["a"].evidence_status, "unavailable")

    async def test_context_io_cannot_replace_the_cutoff_image_with_a_later_frame(self):
        async def slower_context():
            now = datetime.now(UTC)
            newer = replace(self.frame, frame_id=uuid4(), frame_sequence=2,
                            captured_at=now, server_received_at=now)
            self.window.put(newer, stream_lease=self.lease)
            return None

        self.session._context_service.shared_payload = slower_context
        await self.turn()
        self.assertEqual(self.payloads()[-1]["live_view"]["image"]["frame_id"], str(self.frame.frame_id))

    async def test_frame_received_after_the_cutoff_is_not_eligible(self):
        image, state = await self.session._context_service.select_current(
            namespace=1, not_after=self.frame.server_received_at - timedelta(milliseconds=1),
        )
        self.assertIsNone(image)
        self.assertEqual(state, "image_after_cutoff")

    async def test_current_image_tool_works_when_automatic_delivery_is_disabled(self):
        self.settings.VISUAL_REALTIME_ENABLED = False
        await self.turn()
        response = await self.request("get_current_camera_view", {})
        self.assertEqual(self.images(), [])
        self.assertEqual(base64.b64decode(response["parts"][0]["inlineData"]["data"]), self.frame.image_bytes)
        self.assertEqual(response["response"]["image_status"], "included_in_response")
        durable = next(iter(self.executions.rows.values())).result
        self.assertEqual(durable["image_status"], "selected")
        self.assertNotIn("parts", durable)
        self.assertNotIn("image_bytes", json.dumps(durable))

    async def test_new_activity_during_image_write_cannot_release_the_old_turn(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.wires[0].send_event

        async def slow_image(event):
            if "clientContent" in event:
                entered.set()
                await release.wait()
            await original(event)

        self.wires[0].send_event = slow_image
        await self.phone.send("audio.input.append", segment_id="a", audio_base64=ENCODED)
        await self.phone.send("audio.input.commit", segment_id="a")
        await until(entered.is_set)
        epoch = self.session._ingress_epoch
        await self.phone.send("audio.input.append", segment_id="b", audio_base64=ENCODED)
        await until(lambda: self.session._ingress_epoch > epoch)
        release.set()
        await until(lambda: "b" in self.session._turns)
        self.assertFalse(self.session._turns["a"].released)
        self.assertFalse(any("activityEnd" in event.get("realtimeInput", {})
                             for wire in self.wires for event in wire.sent))
        await self.phone.send("audio.input.commit", segment_id="b")
        await until(lambda: self.session._turns["b"].released)
        self.assertEqual(sum("activityEnd" in event.get("realtimeInput", {})
                             for wire in self.wires for event in wire.sent), 1)

    async def test_historical_resolver_keeps_original_session_and_never_becomes_current(self):
        self.settings.VISUAL_REALTIME_ENABLED = False
        await self.turn()
        response = await self.request("get_memory_image", {
            "record_id": str(self.historical.reference.record_id),
            "record_type": self.historical.reference.record_type,
        })
        self.assertEqual(response["response"]["image"]["source"], "historical")
        self.assertEqual(response["response"]["image"]["session_id"], str(self.historical.reference.session_id))
        self.assertEqual(base64.b64decode(response["parts"][0]["inlineData"]["data"]), _jpeg())
        self.assertIsNone(self.session._context_service.delivered_image)
        self.assertGreaterEqual(self.evidence_repository.get_image_evidence.await_count, 4)

    async def test_forgetting_historical_evidence_before_write_drops_attachment(self):
        self.settings.VISUAL_REALTIME_ENABLED = False
        original = self.session._deliver_tool_result

        async def forget(call, turn, result, image):
            self.evidence_repository.get_image_evidence.return_value = None
            await original(call, turn, result, image)

        self.session._deliver_tool_result = forget
        await self.turn()
        response = await self.request("get_memory_image", {
            "record_id": str(self.historical.reference.record_id),
            "record_type": self.historical.reference.record_type,
        })
        self.assertNotIn("parts", response)
        self.assertEqual(response["response"]["image_status"], "unavailable")
        self.loader.download_object.assert_awaited_once()
