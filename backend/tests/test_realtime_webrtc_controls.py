from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.events import ClientEventError
from app.realtime.models import RealtimeSessionContext
from app.realtime.session import RealtimeAudioSession
from app.settings import JarvisSettings


class DirectControlTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = DirectRealtimeAudioSession(
            offer="v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            websocket=SimpleNamespace(close=AsyncMock(), send_text=AsyncMock()),
            context=RealtimeSessionContext(
                audio_session_id="fixture", owner_id="owner", jarvis_session_id=uuid4(),
                client_kind="rayban_audio", session=SimpleNamespace(),
            ),
            settings=JarvisSettings(_env_file=None),
            repository=SimpleNamespace(create_message=AsyncMock(), get_recent_messages=AsyncMock(return_value=[])),
            tool_modules=(), context_providers=(),
        )
        self.provider = SimpleNamespace(capture_mode="auto", send_event=AsyncMock(return_value=True), close=AsyncMock())
        self.session.openai = self.provider
        self.session._send_app_event = AsyncMock(return_value=True)
        self.session._send_session_update = AsyncMock()
        self.addAsyncCleanup(self.session.close)

    async def test_capture_mode_acknowledges_installed_mode(self):
        await self.session._handle_client_event({"type": "audio.capture_mode", "capture_mode": "manual"})
        self.assertEqual(self.provider.capture_mode, "manual")
        self.session._send_session_update.assert_awaited_once()
        self.session._send_app_event.assert_any_await("audio.capture_mode.ready", capture_mode="manual")

    async def test_invalid_mode_never_acknowledges(self):
        with self.assertRaises(ClientEventError):
            await self.session._handle_client_event({"type": "audio.capture_mode", "capture_mode": "invalid"})
        self.session._send_app_event.assert_not_awaited()

    async def test_manual_capture_ready_follows_buffer_clear(self):
        self.provider.capture_mode = "manual"
        events = []
        async def send(payload):
            events.append(payload["type"])
            return True
        async def notify(kind, **fields):
            events.append(kind)
            return True
        self.provider.send_event.side_effect = send
        self.session._send_app_event.side_effect = notify
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": "manual-1"})
        self.assertTrue(self.session._user_audio_pending)
        self.assertEqual(events[-2:], ["input_audio_buffer.clear", "audio.input.ready"])

    async def test_release_before_ready_settles_empty_manual_turn(self):
        self.provider.capture_mode = "manual"
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": "manual-1"})
        await self.session._handle_client_event({"type": "audio.input.abort", "segment_id": "manual-1"})
        self.assertFalse(self.session._user_audio_pending)
        self.assertIsNone(self.session._direct_manual_segment_id)
        self.assertFalse(any(call.args[0].get("type") == "response.create"
                             for call in self.provider.send_event.await_args_list))

    async def test_old_manual_abort_cannot_clear_new_capture(self):
        self.provider.capture_mode = "manual"
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": "new"})
        count = self.provider.send_event.await_count
        await self.session._handle_client_event({"type": "audio.input.abort", "segment_id": "old"})
        self.assertTrue(self.session._user_audio_pending)
        self.assertEqual(self.provider.send_event.await_count, count)

    async def test_manual_identifiers_are_required_and_old_commits_rejected(self):
        self.provider.capture_mode = "manual"
        for identifier in (None, "", 1, "x" * 257):
            with self.subTest(identifier=identifier), self.assertRaises(ClientEventError):
                await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": identifier})
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": "new"})
        with self.assertRaises(ClientEventError):
            await self.session._handle_client_event({"type": "audio.input.commit", "segment_id": "old"})

    async def test_only_accepted_response_is_permitted_on_phone(self):
        with patch.object(RealtimeAudioSession, "_handle_openai_event", new=AsyncMock()):
            for response_id, interrupted in (("accepted", False), ("late", True)):
                self.session._active_responses[response_id] = SimpleNamespace(interrupted=interrupted)
                await self.session._handle_openai_event({"type": "response.created", "response": {"id": response_id}})
        self.session._send_app_event.assert_awaited_once_with(
            "audio.webrtc.response", response_id="accepted", turn_generation=0,
        )
        self.assertEqual(self.session._direct_output_pending, {"accepted"})
        self.session._active_responses.clear()

    async def test_audio_done_before_buffer_started_keeps_speech_lane(self):
        self.session._direct_output_pending.add("response-1")
        with patch.object(RealtimeAudioSession, "_handle_openai_event", new=AsyncMock()):
            await self.session._handle_openai_event({"type": "response.done", "response": {
                "id": "response-1", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "audio", "transcript": "Hello"}]}],
            }})
        self.assertIn("response-1", self.session._direct_output_pending)
        await self.session._handle_openai_event({"type": "output_audio_buffer.started", "response_id": "response-1"})
        await self.session._handle_openai_event({"type": "output_audio_buffer.stopped", "response_id": "response-1"})
        self.assertFalse(self.session._direct_output_pending)

    async def test_tool_only_completion_does_not_hold_speech_lane(self):
        self.session._direct_output_pending.add("tool-response")
        with patch.object(RealtimeAudioSession, "_handle_openai_event", new=AsyncMock()):
            await self.session._handle_openai_event({"type": "response.done", "response": {
                "id": "tool-response", "status": "completed", "output": [{"type": "function_call"}],
            }})
        self.assertFalse(self.session._direct_output_pending)

    async def test_unknown_buffer_cannot_acquire_playback_ownership(self):
        await self.session._handle_openai_event({"type": "output_audio_buffer.started", "response_id": "unknown"})
        self.assertFalse(self.session._direct_audio_responses)
        self.session._send_app_event.assert_not_awaited()

    async def test_pending_tool_cancel_does_not_clear_nonexistent_audio(self):
        self.session._direct_output_pending.add("tool-response")
        with patch.object(RealtimeAudioSession, "_cancel_response_locked", new=AsyncMock()):
            await self.session._cancel_response_locked()
        self.provider.send_event.assert_not_awaited()

    async def test_output_item_aliases_share_the_authoritative_control_path(self):
        self.session._active_responses["accepted"] = SimpleNamespace(interrupted=False, completed_response=None)
        self.session._direct_output_pending.add("accepted")
        with patch.object(RealtimeAudioSession, "_handle_openai_event", new=AsyncMock()):
            for suffix in ("added", "created", "done", "completed"):
                self.session._send_app_event.reset_mock()
                await self.session._handle_openai_event({
                    "type": "response.output_item." + suffix, "response_id": "accepted",
                    "item": {"id": "item-1", "type": "message", "phase": "final_answer", "content": [{"text": "private"}]},
                })
                self.session._send_app_event.assert_awaited_once_with(
                    "audio.webrtc.output_item", response_id="accepted",
                    item={"id": "item-1", "type": "message", "phase": "final_answer"},
                )
        self.session._active_responses.clear()

    async def test_unadmitted_or_interrupted_items_never_open_phone_output(self):
        with patch.object(RealtimeAudioSession, "_handle_openai_event", new=AsyncMock()):
            for response_id, interrupted, pending in (("unknown", False, False), ("late", True, True)):
                self.session._active_responses[response_id] = SimpleNamespace(interrupted=interrupted, completed_response=None)
                if pending:
                    self.session._direct_output_pending.add(response_id)
                await self.session._handle_openai_event({
                    "type": "response.output_item.created", "response_id": response_id,
                    "item": {"id": "item-1", "type": "message"},
                })
        self.session._send_app_event.assert_not_awaited()
        self.session._active_responses.clear()

    async def test_context_prefetch_overlaps_connect_and_answer_precedes_context(self):
        context_started = asyncio.Event()
        release_context = asyncio.Event()
        order = []

        async def build(_session):
            context_started.set()
            await release_context.wait()
            return "prepared-context"

        async def connect():
            await context_started.wait()
            order.append("connected")

        async def notify(kind, **fields):
            order.append(kind)
            if kind == "audio.webrtc.answer":
                self.assertFalse(release_context.is_set())
                release_context.set()
            return True

        async def update():
            self.assertEqual(await self.session._build_assistant_context(), "prepared-context")
            order.append("context_installed")
            return 0

        self.provider.connect = connect
        self.provider.answer = "v=0\r\n"
        self.session._send_app_event.side_effect = notify
        self.session._send_session_update.side_effect = update
        self.session._wait_for_openai_session_ready = AsyncMock()
        with patch.object(RealtimeAudioSession, "_build_assistant_context", new=build):
            async with asyncio.timeout(1):
                await self.session._initialize_openai_session()
        self.assertEqual(order, ["connected", "audio.webrtc.answer", "context_installed"])
        self.assertTrue(self.session._realtime_context_ready.is_set())

    async def test_failed_connect_joins_context_prefetch(self):
        started = asyncio.Event()
        settled = asyncio.Event()

        async def build(_session):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                settled.set()

        async def connect():
            await started.wait()
            raise RuntimeError("fixture connect failure")

        self.provider.connect = connect
        with patch.object(RealtimeAudioSession, "_build_assistant_context", new=build):
            async with asyncio.timeout(1):
                with self.assertRaisesRegex(RuntimeError, "fixture connect failure"):
                    await self.session._initialize_openai_session()
        self.assertTrue(settled.is_set())
        self.assertIsNone(self.session._direct_initial_context)
