"""Actual lean dispatch/control wiring with substituted domain/provider I/O."""
import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.files.transfer import CAPTURE_MAX_BYTES, CAPTURE_MAX_DURATION_MS, FileTransferPolicy
from app.realtime.lean_session import LeanRealtimeAudioSession
from app.realtime.models import RealtimeSessionContext
from app.settings import JarvisSettings
from test_lean_voice_session import FakeProvider, FakeSocket


class RecordingVoiceSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.provider, self.socket = FakeProvider(), FakeSocket()
        self.provider.created_call_id = "fixture-recording-provider"
        self.generation = uuid4()

        async def admit(command, area_id, policy):
            return SimpleNamespace(outcome="admitted", capture_allowed=True,
                scope=SimpleNamespace(recording_id=command.operation_id, max_bytes=policy.max_bytes,
                    max_duration_ms=policy.max_duration_ms, include_jarvis_audio=False))

        self.admissions = SimpleNamespace(selections=SimpleNamespace(read=AsyncMock(return_value={
            "area_id": str(uuid4()), "area_status": "active"})), admit=AsyncMock(side_effect=admit), stop=AsyncMock())
        services = SimpleNamespace(recording_admission=self.admissions,
            recording_policy=FileTransferPolicy("mp4_capture_v1", 40000000))
        self.session = LeanRealtimeAudioSession(websocket=self.socket,
            context=RealtimeSessionContext(audio_session_id=str(uuid4()), owner_id="lean-recording-test",
                jarvis_session_id=uuid4(), client_kind="phone_audio", session=SimpleNamespace()),
            settings=JarvisSettings(_env_file=None), repository=SimpleNamespace(create_message=AsyncMock()),
            offer="v=0", provider_connection=self.provider, runtime_services=services,
            tool_modules=(), context_providers=())
        await self.session._handle_client_event({"type": "recording.capabilities", "version": 1,
            "generation_id": str(self.generation), "available": True, "include_jarvis_audio": True,
            "max_bytes": 40000000, "max_duration_ms": CAPTURE_MAX_DURATION_MS})

    async def asyncTearDown(self):
        await self.session.close("test_done")

    async def drain(self):
        jobs = tuple(t for t in self.session._lean_work if t is not self.session._lean_app_writer)
        if jobs:
            await asyncio.gather(*jobs)

    async def request(self):
        await self.session._handle_openai_event({"type": "response.created", "response": {"id": "recording-response"}})
        await self.session._handle_openai_event({"type": "response.function_call_arguments.done",
            "response_id": "recording-response", "call_id": "recording-call", "name": "live_recording",
            "arguments": json.dumps({"action": "start", "include_jarvis_audio": False})})

    def controls(self):
        result = []
        while not self.session._lean_app_events.empty():
            kind, body = self.session._lean_app_events.get_nowait()
            self.session._lean_app_events.task_done()
            if kind == "recording.command":
                result.append(body)
        return result

    async def test_actual_tool_dispatch_and_quiet_device_report(self):
        await self.session._send_session_update()
        self.assertIn("live_recording", [t["name"] for t in self.provider.sent[-1]["session"]["tools"]])
        await self.request()
        await self.drain()
        controls = self.controls()
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["audio_session_id"], self.session.context.audio_session_id)
        self.provider.sent.clear()
        report = {"type": "recording.report", "version": 1, "generation_id": str(self.generation),
            "recording_id": controls[0]["recording_id"], "sequence": 1, "state": "started"}
        await self.session._handle_client_event(report)
        await self.session._handle_client_event(report)
        await self.drain()
        self.assertEqual(len(self.provider.sent), 1)
        self.assertEqual(self.provider.sent[0]["type"], "conversation.item.create")
        text = self.provider.sent[0]["item"]["content"][0]["text"]
        self.assertIn('"basis":"client_report"', text)
        self.assertIn('"saved":false', text)

    async def test_provider_reader_and_speech_interruption_do_not_cancel_pending_admission(self):
        entered, release = asyncio.Event(), asyncio.Event()
        normal = self.admissions.admit.side_effect

        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await normal(*args, **kwargs)

        self.admissions.admit.side_effect = delayed
        await self.request()
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(self.session._handle_openai_event({
            "type": "input_audio_buffer.speech_started", "item_id": "new-question"}), 0.25)
        self.admissions.stop.assert_not_awaited()
        release.set()
        await self.drain()
        self.assertEqual([e["action"] for e in self.controls()], ["start"])
