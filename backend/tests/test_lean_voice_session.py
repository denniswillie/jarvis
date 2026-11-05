import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.errors import JarvisHTTPError
from app.realtime.lean_selection import require_lean_offer, select_lean_voice
from app.realtime.lean_session import LEAN_INSTRUCTIONS, LeanRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.realtime.webrtc import OpenAIWebRTCConnection
from app.settings import JarvisSettings


class FakeProvider:
    transport = "webrtc"
    answer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"

    def __init__(self):
        self.sent = []

    async def connect(self):
        pass

    async def send_event(self, payload, **kwargs):
        payload = payload() if callable(payload) else payload
        if asyncio.iscoroutine(payload):
            payload = await payload
        if payload is None:
            return False
        self.sent.append(payload)
        return True

    async def receive_event(self):
        return {"type": "session.updated"}

    async def close(self):
        pass


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def close(self, **kwargs):
        pass


class LeanSelectionTests(unittest.TestCase):
    def test_legacy_default_and_explicit_rollback(self):
        for values in ([], ["legacy"]):
            self.assertFalse(select_lean_voice(settings=SimpleNamespace(LEAN_VOICE_ENABLED=True),
                requested_values=values, provider="openai", transport="webrtc"))

    def test_selected_lean_requires_enablement_and_exclusive_direct_transport(self):
        base = dict(settings=SimpleNamespace(LEAN_VOICE_ENABLED=True),
                    requested_values=["lean_v1"], provider="openai", transport="webrtc")
        self.assertTrue(select_lean_voice(**base))
        for replacement in ({"settings": SimpleNamespace(LEAN_VOICE_ENABLED=False)},
                            {"transport": "websocket"}, {"provider": "gemini"},
                            {"live_selected": True}, {"requested_values": ["lean_v1", "lean_v1"]}):
            with self.subTest(replacement=replacement), self.assertRaises(JarvisHTTPError):
                select_lean_voice(**(base | replacement))

    def test_offer_must_match_mode_and_not_enable_two_owners(self):
        require_lean_offer({"conversation_mode": "lean_v1"}, selected=True, capture_mode="auto")
        for payload, mode in (({}, "auto"), ({"conversation_mode": "lean_v1"}, "manual"),
                              ({"conversation_mode": "lean_v1", "voice_route": {}}, "auto")):
            with self.assertRaises(JarvisHTTPError):
                require_lean_offer(payload, selected=True, capture_mode=mode)

    def test_native_flags_are_opt_in_and_do_not_mutate_input_configuration(self):
        settings = JarvisSettings(_env_file=None)
        original = {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}}
        for native in (False, True):
            connection = OpenAIWebRTCConnection(settings=settings, owner_id="test", offer="v=0", native_turns=native)
            vad = connection.session_config(original)["audio"]["input"]["turn_detection"]
            self.assertIs(vad["create_response"], native)
            self.assertIs(vad["interrupt_response"], native)
        self.assertIn("format", original["audio"]["input"])


class LeanSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider = FakeProvider()
        self.socket = FakeSocket()
        self.repository = SimpleNamespace(
            get_recent_messages=AsyncMock(side_effect=AssertionError("Legacy context read")),
            create_message=AsyncMock())
        self.session = LeanRealtimeAudioSession(websocket=self.socket,
            context=RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id="lean-test",
                jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None), repository=self.repository,
            offer="v=0", provider_connection=self.provider, tool_modules=(), context_providers=())

    async def asyncTearDown(self):
        await self.session.close("test_done")

    async def created(self, response_id="resp_test"):
        await self.session._handle_openai_event({"type": "response.created", "response": {"id": response_id}})

    async def drain(self):
        tasks = tuple(t for t in self.session._lean_work if t is not self.session._lean_app_writer)
        if tasks:
            await asyncio.gather(*tasks)

    async def tool(self, call_id="call_test"):
        await self.session._handle_openai_event({"type": "response.function_call_arguments.done",
            "response_id": "resp_test", "call_id": call_id, "name": "live_briefing", "arguments": "{}"})

    async def done(self):
        await self.session._handle_openai_event({"type": "response.done",
            "response": {"id": "resp_test", "status": "completed", "output": []}})

    async def test_short_prompt_and_only_backed_tool_are_advertised(self):
        await self.session._send_session_update()
        config = self.provider.sent[-1]["session"]
        self.assertEqual(config["instructions"], LEAN_INSTRUCTIONS)
        self.assertLess(len(LEAN_INSTRUCTIONS), 2000)
        self.assertIn("Pending work alone is not a reason", config["instructions"])
        self.assertIn("arrive through normal background delivery", config["instructions"])
        self.assertRegex(config["instructions"], r"Acceptance is not\s+proof of a save")
        self.assertEqual([tool["name"] for tool in config["tools"]], ["live_briefing"])
        self.assertNotIn("jarvis_request_source", json.dumps(config))
        self.repository.get_recent_messages.assert_not_awaited()

    async def test_native_response_does_not_require_backend_request_id(self):
        await self.created()
        self.assertFalse(self.session._active_responses["resp_test"].interrupted)
        self.assertFalse(self.provider.sent)

    async def test_input_commit_does_not_create_an_ordinary_response(self):
        for kind in ("speech_started", "speech_stopped", "committed"):
            await self.session._handle_openai_event({"type": "input_audio_buffer." + kind, "item_id": "item_user"})
        self.assertFalse(self.provider.sent)
        self.repository.get_recent_messages.assert_not_awaited()

    async def test_tool_output_then_one_protocol_continuation(self):
        await self.created()
        await self.tool()
        await self.tool()
        await self.done()
        await self.drain()
        self.assertEqual([e["type"] for e in self.provider.sent], ["conversation.item.create", "response.create"])
        self.assertEqual(self.provider.sent[0]["item"]["call_id"], "call_test")

    async def test_barge_in_suppresses_obsolete_tool_continuation(self):
        await self.created()
        await self.tool()
        await self.session._handle_openai_event({"type": "input_audio_buffer.speech_started", "item_id": "new_user"})
        await self.done()
        await self.drain()
        self.assertNotIn("response.create", [e["type"] for e in self.provider.sent])
        self.assertNotIn("response.cancel", [e["type"] for e in self.provider.sent])

    async def test_conflicting_call_id_closes_without_second_dispatch(self):
        await self.created()
        await self.tool()
        await self.session._handle_openai_event({"type": "response.function_call_arguments.done",
            "response_id": "resp_test", "call_id": "call_test", "name": "live_task", "arguments": "{}"})
        self.assertTrue(self.session._closed.is_set())
        self.assertEqual(len(self.session._lean_calls), 1)

    async def test_startup_does_not_await_optional_area_lookup(self):
        blocker = asyncio.Event()
        async def area_read(*args):
            await blocker.wait()
        self.session._lean_services = SimpleNamespace(
            knowledge_runtime=SimpleNamespace(area_selections=SimpleNamespace(read=area_read)))
        await asyncio.wait_for(self.session._initialize_openai_session(), timeout=0.5)
        self.assertTrue(self.session._realtime_context_ready.is_set())
        self.assertIsNone(self.session._lean_live)
        self.repository.get_recent_messages.assert_not_awaited()

    async def test_no_old_context_injection_on_memory_updates(self):
        await self.session.notify_working_memory_updated(3)
        await self.session.notify_owner_working_memory_updated(4)
        await self.session.notify_memory_item_changed(object())
        self.assertFalse(self.provider.sent)

    async def test_native_speech_cancel_is_not_a_task_command(self):
        await self.created()
        await self.session._handle_client_event({"type": "response.cancel"})
        await self.drain()
        self.assertEqual([e["type"] for e in self.provider.sent], ["response.cancel", "output_audio_buffer.clear"])
        self.assertFalse(self.session._lean_calls)


if __name__ == "__main__":
    unittest.main()
