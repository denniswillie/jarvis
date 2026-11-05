"""Direct-media protocol checks using the production session and control handlers."""
import asyncio
import json
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.events import ClientEventError
from app.realtime.models import RealtimeSessionContext
from app.realtime.webrtc import OpenAIWebRTCConnection, validate_offer
from app.settings import JarvisSettings


OFFER = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n"


class OfferTests(TestCase):
    def test_offer_is_bounded_and_audio_only_control_contract_is_explicit(self):
        self.assertEqual(validate_offer({"type": "audio.webrtc.offer", "sdp": OFFER}), (OFFER, "auto"))
        for payload in (None, {}, {"type": "audio.input.append"},
                        {"type": "audio.webrtc.offer", "sdp": "secret"},
                        {"type": "audio.webrtc.offer", "sdp": OFFER, "capture_mode": "unknown"},
                        {"type": "audio.webrtc.offer", "sdp": OFFER + "x" * 100000}):
            with self.subTest(payload_type=type(payload).__name__), self.assertRaises(ValueError):
                validate_offer(payload)


class ConnectionTests(IsolatedAsyncioTestCase):
    def connection(self, handler):
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http.aclose)
        return OpenAIWebRTCConnection(settings=JarvisSettings(_env_file=None,
            OPENAI_API_KEY="fixture-only", OPENAI_REALTIME_MODEL="fixture-model"),
            owner_id="owner", offer=OFFER, http=http)

    async def test_setup_attaches_only_to_the_server_created_call_and_cleans_up(self):
        requests = []
        def handler(request):
            requests.append(request)
            if request.url.path.endswith("/hangup"):
                return httpx.Response(200)
            return httpx.Response(201, text=OFFER, headers={"location": "/v1/realtime/calls/rtc_fixture"})
        connection = self.connection(handler)
        socket = SimpleNamespace(close=AsyncMock(), send=AsyncMock())
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock(return_value=socket)) as connect:
            await connection.connect()
            self.assertEqual(connect.call_args.args[0], "wss://api.openai.com/v1/realtime?call_id=rtc_fixture")
            self.assertEqual(connection.answer, OFFER)
            self.assertEqual(requests[0].headers["authorization"], "Bearer fixture-only")
            await connection.send_event({"type": "session.update", "session": {
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}, "output": {}}
            }})
            config = json.loads(socket.send.call_args.args[0])["session"]
            self.assertNotIn("format", config["audio"]["input"])
            self.assertEqual(config["audio"]["input"]["turn_detection"]["type"], "semantic_vad")
            self.assertFalse(config["audio"]["input"]["turn_detection"]["create_response"])
            await connection.close()
        self.assertTrue(str(requests[-1].url).endswith("/rtc_fixture/hangup"))
        socket.close.assert_awaited_once()

    async def test_untrusted_call_location_is_never_used_for_credentials(self):
        connection = self.connection(lambda request: httpx.Response(201, text=OFFER,
            headers={"location": "https://other.invalid/v1/realtime/calls/rtc_fixture"}))
        with patch("app.realtime.webrtc.websockets.connect", AsyncMock()) as connect:
            with self.assertRaises(RuntimeError):
                await connection.connect()
            connect.assert_not_awaited()
        await connection.close()

    async def test_manual_mode_and_original_configuration_are_preserved(self):
        connection = self.connection(lambda request: httpx.Response(500))
        original = {"audio": {"input": {"turn_detection": None, "format": {"type": "audio/pcm"}}}}
        automatic = connection.session_config(original)
        self.assertIsNone(original["audio"]["input"]["turn_detection"])
        self.assertEqual(automatic["audio"]["input"]["turn_detection"]["eagerness"], "auto")
        connection.capture_mode = "manual"
        self.assertIsNone(connection.session_config(original)["audio"]["input"]["turn_detection"])
        await connection.close()


class DirectSessionTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = DirectRealtimeAudioSession(offer=OFFER,
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(audio_session_id="fixture", owner_id="owner",
                jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=())
        self.provider = SimpleNamespace(capture_mode="auto", send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.session.openai = self.provider
        self.session._send_app_event = AsyncMock(return_value=True)
        self.session._request_normal_response = AsyncMock(return_value=True)
        self.addAsyncCleanup(self.session.close)

    async def event(self, kind, **fields):
        await self.session._handle_openai_event({"type": kind, **fields})

    async def test_semantic_commit_requests_one_response_without_client_end_gate(self):
        await self.event("input_audio_buffer.speech_started", item_id="input-1")
        self.assertTrue(self.session._user_audio_pending)
        await self.event("input_audio_buffer.speech_stopped", item_id="input-1")
        await self.event("input_audio_buffer.committed", item_id="input-1")
        await asyncio.sleep(0)
        self.session._request_normal_response.assert_awaited_once_with(turn_generation=1)
        await self.event("input_audio_buffer.committed", item_id="input-1")
        await asyncio.sleep(0)
        self.session._request_normal_response.assert_awaited_once()
        self.assertFalse(any(call.args[0].get("type") == "input_audio_buffer.commit"
                             for call in self.provider.send_event.await_args_list))

    async def test_old_commit_cannot_answer_a_newer_utterance(self):
        await self.event("input_audio_buffer.speech_started", item_id="old")
        await self.event("input_audio_buffer.speech_started", item_id="new")
        await self.event("input_audio_buffer.committed", item_id="old")
        self.session._request_normal_response.assert_not_awaited()
        self.assertTrue(self.session._user_audio_pending)
        await self.event("input_audio_buffer.committed", item_id="new")
        await asyncio.sleep(0)
        self.session._request_normal_response.assert_awaited_once_with(turn_generation=2)

    async def test_media_is_not_relayed_or_counted_as_backend_forwarded_pcm(self):
        await self.session._forward_output_media({"type": "response.output_audio.delta", "delta": "AA=="},
                                                event_type="response.output_audio.delta")
        self.session._send_app_event.assert_not_awaited()
        self.assertFalse(self.session._native_playback)
        with self.assertRaises(ClientEventError):
            await self.session._handle_client_event({"type": "audio.input.append", "audio_base64": "AA=="})
        with self.assertRaises(ClientEventError):
            await self.session._handle_client_event({"type": "audio.input.commit"})

    async def test_interrupt_clears_direct_playback_without_cancelling_durable_work(self):
        self.session._direct_output_pending.add("old-output")
        self.session._direct_audio_responses.add("old-output")
        await self.event("input_audio_buffer.speech_started", item_id="new-input")
        self.provider.send_event.assert_any_await({"type": "output_audio_buffer.clear"})
        self.assertFalse(self.session._direct_output_pending)

    async def test_generated_done_does_not_release_the_audio_lane_before_playback_stops(self):
        self.session._direct_output_pending.add("speaking")
        await self.event("output_audio_buffer.started", response_id="speaking")
        await self.event("response.done", response={"id": "speaking", "status": "completed"})
        self.assertIn("speaking", self.session._direct_output_pending)
        await self.event("output_audio_buffer.stopped", response_id="speaking")
        self.assertFalse(self.session._direct_output_pending)
