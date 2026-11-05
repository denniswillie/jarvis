from __future__ import annotations

import asyncio
import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.tools.registry import JarvisToolRegistry, ToolDefinition
from evals.gemini_live.client import LiveProbe, ToolBridge, Turn, declarations
from evals.gemini_live.runner import MemoryFixture, memory_registry


class GeminiLiveAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    async def test_inline_tool_image_is_media_not_base64_in_json_result(self):
        registry = JarvisToolRegistry()
        registry.register(ToolDefinition(name="camera", description="Fixture camera",
            input_schema={"type": "object", "properties": {}},
            handler=AsyncMock(return_value={"status": "success"})))
        bridge = ToolBridge(registry, response_images={"camera": b"jpeg-bytes"})
        send = AsyncMock()
        await bridge.dispatch({"id": "image-call", "name": "camera", "args": {}}, send)
        await bridge.close()
        response = send.await_args.args[0]["toolResponse"]["functionResponses"][0]
        self.assertEqual(response["response"], {"status": "success"})
        self.assertEqual(base64.b64decode(response["parts"][0]["inlineData"]["data"]), b"jpeg-bytes")

    async def test_live_frame_and_audio_share_one_manual_activity(self):
        client = LiveProbe(key="key", manual_vad=True)
        client.socket = AsyncMock()
        with patch("evals.gemini_live.client.asyncio.sleep", AsyncMock()):
            await client.speech(bytes(3200), Turn(), image=b"jpeg")
        events = [json.loads(call.args[0])["realtimeInput"] for call in client.socket.send.await_args_list]
        self.assertEqual(events[0], {"activityStart": {}})
        self.assertIn("video", events[1])
        self.assertIn("audio", events[2])
        self.assertEqual(events[3], {"activityEnd": {}})

    async def test_tuning_is_explicit_and_preserved_in_setup(self):
        socket = AsyncMock()
        socket.recv.return_value = json.dumps({"setupComplete": {}})
        client = LiveProbe(key="key", thinking="medium", media_resolution="MEDIA_RESOLUTION_HIGH")
        with patch("evals.gemini_live.client.websockets.connect", AsyncMock(return_value=socket)):
            await client.connect()
        config = json.loads(socket.send.await_args_list[0].args[0])["setup"]["generationConfig"]
        self.assertEqual(config["thinkingConfig"]["thinkingLevel"], "medium")
        self.assertEqual(config["mediaResolution"], "MEDIA_RESOLUTION_HIGH")
        await client.close()

    async def test_image_history_is_loaded_before_realtime_question(self):
        socket = AsyncMock()
        socket.recv.return_value = json.dumps({"setupComplete": {}})
        client = LiveProbe(key="private-test-key", manual_vad=False)
        with patch("evals.gemini_live.client.websockets.connect", AsyncMock(return_value=socket)) as connect:
            await client.connect(image=b"original-image")
            await client.text("What is visible?", Turn())
        events = [json.loads(call.args[0]) for call in socket.send.await_args_list]
        self.assertTrue(events[0]["setup"]["historyConfig"]["initialHistoryInClientContent"])
        image = events[1]["clientContent"]["turns"][0]["parts"][0]["inlineData"]
        self.assertEqual(base64.b64decode(image["data"]), b"original-image")
        self.assertEqual(events[2]["realtimeInput"]["text"], "What is visible?")
        self.assertEqual(len(events), 3)
        self.assertNotIn("private-test-key", connect.call_args.args[0])
        await client.close()

    async def test_reconnect_requires_real_handle_and_does_not_reseed_history(self):
        socket = AsyncMock()
        socket.recv.side_effect = [json.dumps({"setupComplete": {}})]
        client = LiveProbe(key="key")
        with self.assertRaisesRegex(RuntimeError, "No resumable"):
            await client.connect(resume=True)
        client.handle = "server-issued-handle"
        with patch("evals.gemini_live.client.websockets.connect", AsyncMock(return_value=socket)):
            await client.connect(resume=True)
        setup = json.loads(socket.send.await_args.args[0])["setup"]
        self.assertEqual(setup["sessionResumption"]["handle"], "server-issued-handle")
        self.assertNotIn("historyConfig", setup)
        self.assertEqual(socket.send.await_count, 1)
        await client.close()

    async def test_receives_all_audio_parts_even_with_transcript_and_completion(self):
        turn = Turn()
        turn.add({"serverContent": {
            "modelTurn": {"parts": [
                {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AQI="}},
                {"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AwQ="}},
            ]}, "outputTranscription": {"text": "Hello"}, "turnComplete": True,
        }, "usageMetadata": {"promptTokensDetails": [{"modality": "IMAGE", "tokenCount": 1100}]}})
        self.assertEqual(turn.audio, bytes([1, 2, 3, 4]))
        self.assertEqual(turn.transcript, "Hello")
        self.assertTrue(turn.completed)
        self.assertEqual(turn.summary()["input_modalities"]["IMAGE"], 1100)

    async def test_transcript_is_never_counted_as_native_audio(self):
        turn = Turn()
        turn.add({"serverContent": {"outputTranscription": {"text": "Text only"}, "turnComplete": True}})
        self.assertEqual(turn.summary()["audio_bytes"], 0)
        self.assertIsNone(turn.summary()["first_audio_after_input_end_seconds"])
        self.assertFalse(turn.first_audio_event.is_set())

    async def test_repeated_provider_call_executes_only_once(self):
        handler = AsyncMock(return_value={"status": "success"})
        registry = JarvisToolRegistry()
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        registry.register(ToolDefinition(name="write", description="Test", input_schema=schema,
                                         handler=handler, mutates_state=True))
        bridge, send = ToolBridge(registry), AsyncMock()
        call = {"id": "same-id", "name": "write", "args": {}}
        await bridge.dispatch(call, send)
        await bridge.dispatch(call, send)
        await bridge.close()
        handler.assert_awaited_once()
        send.assert_awaited_once()
        self.assertEqual(declarations(registry)[0]["parametersJsonSchema"], schema)

    async def test_conflicting_duplicate_id_is_rejected(self):
        bridge = ToolBridge(JarvisToolRegistry())
        await bridge.dispatch({"id": "id", "name": "unknown", "args": {}}, AsyncMock())
        with self.assertRaisesRegex(ValueError, "different arguments"):
            await bridge.dispatch({"id": "id", "name": "unknown", "args": {"changed": True}}, AsyncMock())
        await bridge.close()

    async def test_cancelled_tool_response_does_not_repeat_completed_write(self):
        fixture = MemoryFixture()
        bridge = ToolBridge(memory_registry(fixture, "test-owner"), response_delay=.01)
        send = AsyncMock()
        call = {"id": "memory-call", "name": "remember", "args": {
            "memory_type": "fact", "content": "I prefer numbered steps.",
        }}
        await bridge.dispatch(call, send)
        await asyncio.wait_for(bridge.called.wait(), 1)
        bridge.cancel(["memory-call"])
        await bridge.dispatch(call, send)
        await bridge.close()
        self.assertEqual(len(fixture.entries), 1)
        self.assertEqual(fixture.owners, ["test-owner"])
        self.assertIn("memory-call", fixture.operations)
        send.assert_not_awaited()

    async def test_real_jarvis_memory_tools_read_back_fixture_after_new_bridge(self):
        fixture = MemoryFixture()
        bridge = ToolBridge(memory_registry(fixture, "owner"))
        await bridge.dispatch({"id": "remember-1", "name": "remember", "args": {
            "memory_type": "fact", "content": "I prefer short numbered instructions.",
        }}, AsyncMock())
        await bridge.close()
        second, send = ToolBridge(memory_registry(fixture, "owner")), AsyncMock()
        await second.dispatch({"id": "profile-1", "name": "get_memory_profile", "args": {}}, send)
        await second.close()
        response = send.await_args.args[0]["toolResponse"]["functionResponses"][0]["response"]
        self.assertEqual(response["status"], "success")
        self.assertIn("numbered", json.dumps(response))


if __name__ == "__main__":
    unittest.main()
