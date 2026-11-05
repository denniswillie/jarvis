"""Real native/session/bridge writers with controlled external transport and storage.

No provider, SQL, RTP or native app acceptance is claimed. Existing fixture
methods are borrowed explicitly so inherited test collection is not inflated.
"""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

from app.realtime.events import ClientEventError
import test_voice_context_response_pipeline as pipeline


class NativeManualInputReadinessTests(unittest.IsolatedAsyncioTestCase):
    cleanup = pipeline.VoiceContextResponsePipelineTests.cleanup
    until = pipeline.VoiceContextResponsePipelineTests.until
    rows = pipeline.VoiceContextResponsePipelineTests.rows
    observation = pipeline.VoiceContextResponsePipelineTests.observation
    expire_inside_adapter_guard = pipeline.VoiceContextResponsePipelineTests.expire_inside_adapter_guard
    mutations = pipeline.VoiceContextResponsePipelineTests.mutations
    creates = pipeline.VoiceContextResponsePipelineTests.creates

    async def asyncSetUp(self):
        await pipeline.VoiceContextResponsePipelineTests.asyncSetUp(self)
        await self.session._handle_client_event({"type": "audio.capture_mode", "capture_mode": "manual"})
        self.assertEqual(self.session.openai.capture_mode, "manual")
        self.assertTrue(any(e["type"] == "audio.capture_mode.ready" for e in self.app.events))

    def ready_events(self):
        return [e for e in self.app.events if e["type"] == "audio.input.ready"]

    async def begin(self, segment):
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": segment})

    async def test_actual_begin_behind_context_write_readies_before_context_ack(self):
        entered, release = asyncio.Event(), asyncio.Event()
        created = []
        original_send = self.provider.send

        async def controlled_send(raw):
            payload = json.loads(raw)
            if payload["type"] == "conversation.item.create" and not created:
                self.provider.writes.append(payload)
                created.append(payload)
                entered.set()
                await release.wait()
                return  # Exact create ACK remains intentionally pending.
            await original_send(raw)

        self.provider.send = controlled_send
        segment = str(uuid4())
        task = None
        try:
            await self.observation("new as-of evidence during manual begin")
            async with asyncio.timeout(2):
                await entered.wait()
            self.assertFalse(self.runtime.ready)
            task = asyncio.create_task(self.begin(segment))
            await self.until(lambda: self.session._direct_manual_segment_id == segment)
            self.assertEqual(self.ready_events(), [])
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertFalse(self.runtime.ready)
            ready = self.ready_events()
            self.assertEqual(len(ready), 1)
            self.assertEqual(ready[0]["segment_id"], segment)
            self.assertEqual(ready[0]["route"], self.runtime.request.route.model_dump(mode="json"))
            self.assertEqual(ready[0]["audio_session_id"], self.session.context.audio_session_id)
            self.assertFalse(await self.session._send_app_event("audio.webrtc.started", response_id="resp_unadmitted"))
            self.assertEqual(self.creates(), [])
            self.assertTrue(any(e["type"] == "input_audio_buffer.clear" for e in self.provider.writes))
            await self.provider.inject("conversation.item.done", item=created[0]["item"])
            await self.until(lambda: self.runtime.ready)
        finally:
            release.set()
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            self.provider.send = original_send

    async def test_duplicate_begin_does_not_clear_or_advance_turn_foreign_segment_rejected(self):
        segment = str(uuid4())
        await self.begin(segment)
        generation = self.session._turn_generation
        clears = sum(e["type"] == "input_audio_buffer.clear" for e in self.provider.writes)
        await self.begin(segment)
        self.assertEqual(self.session._turn_generation, generation)
        self.assertEqual(sum(e["type"] == "input_audio_buffer.clear" for e in self.provider.writes), clears)
        with self.assertRaisesRegex(ClientEventError, "conflicts with the active segment"):
            await self.begin(str(uuid4()))
        with self.assertRaisesRegex(ClientEventError, "does not match"):
            await self.session._handle_client_event({"type": "audio.input.commit", "segment_id": str(uuid4())})
        self.assertEqual(self.session._direct_manual_segment_id, segment)
        self.assertTrue(all(e["segment_id"] == segment for e in self.ready_events()))

    async def test_duplicate_pending_begin_has_no_early_receipt_or_second_clear(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.provider.send
        clears = []

        async def held(raw):
            if json.loads(raw)["type"] == "input_audio_buffer.clear":
                clears.append(raw)
                entered.set()
                await release.wait()
            await original(raw)

        segment = str(uuid4())
        self.provider.send = held
        task = asyncio.create_task(self.begin(segment))
        try:
            async with asyncio.timeout(1):
                await entered.wait()
            await self.begin(segment)
            self.assertEqual(self.ready_events(), [])
            self.assertEqual(len(clears), 1)
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertEqual(len(self.ready_events()), 1)
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.provider.send = original

    async def test_provider_clear_not_written_never_readies_input(self):
        original = self.session.openai.send_event

        async def rejected(event, **kwargs):
            if isinstance(event, dict) and event.get("type") == "input_audio_buffer.clear":
                return False
            return await original(event, **kwargs)

        self.session.openai.send_event = rejected
        try:
            await self.begin(str(uuid4()))
            self.assertEqual(self.ready_events(), [])
            self.assertIsNone(self.session._direct_manual_clear_generation)
        finally:
            self.session.openai.send_event = original

    async def test_pending_client_write_rechecks_all_owned_input_boundaries(self):
        segment = str(uuid4())
        await self.begin(segment)
        original_send = self.runtime._owner_send_text
        session, runtime = self.session, self.runtime
        cases = {
            "runtime_closing": (runtime, "_closing", True),
            "context_revoked": (session, "_context_revoked", True),
            "channel_closing": (runtime.channel, "_closing", True),
            "turn_replaced": (session, "_turn_generation", session._turn_generation + 1),
            "segment_replaced": (session, "_direct_manual_segment_id", str(uuid4())),
            "clear_not_confirmed": (session, "_direct_manual_clear_generation", None),
            "provider_replaced": (session, "openai", SimpleNamespace(capture_mode="manual")),
            "runtime_replaced": (session, "_voice_route_runtime", object()),
            "generation_replaced": (runtime, "request", runtime.request.model_copy(update={
                "route": runtime.request.route.model_copy(update={"route_generation": 2})})),
            "owner_replaced": (session, "context", SimpleNamespace(owner_id="other",
                jarvis_session_id=session.context.jarvis_session_id, audio_session_id=session.context.audio_session_id)),
            "session_replaced": (session, "context", SimpleNamespace(owner_id=session.context.owner_id,
                jarvis_session_id=uuid4(), audio_session_id=session.context.audio_session_id)),
            "connection_replaced": (session, "context", SimpleNamespace(owner_id=session.context.owner_id,
                jarvis_session_id=session.context.jarvis_session_id, audio_session_id="other-audio")),
        }
        for name, (owner, attribute, changed) in cases.items():
            with self.subTest(boundary=name):
                previous = getattr(owner, attribute)
                before = len(self.ready_events())

                async def queued(text, **kwargs):
                    setattr(owner, attribute, changed)
                    return await original_send(text, **kwargs)

                runtime._owner_send_text = queued
                try:
                    self.assertFalse(await session._send_app_event("audio.input.ready", segment_id=segment))
                    self.assertEqual(len(self.ready_events()), before)
                finally:
                    setattr(owner, attribute, previous)
                    runtime._owner_send_text = original_send

    async def test_generation_change_during_provider_clear_cannot_ready_old_segment(self):
        original = self.provider.send

        async def changed(raw):
            await original(raw)
            if json.loads(raw)["type"] == "input_audio_buffer.clear":
                self.session._turn_generation += 1

        self.provider.send = changed
        try:
            await self.begin(str(uuid4()))
            self.assertEqual(self.ready_events(), [])
            self.assertIsNone(self.session._direct_manual_clear_generation)
        finally:
            self.provider.send = original

    async def test_ready_does_not_bypass_original_write_guard_or_output_context(self):
        segment = str(uuid4())
        await self.begin(segment)
        original_count = len(self.ready_events())
        self.runtime._context_replacement_active = True
        entries = []

        @asynccontextmanager
        async def denied():
            entries.append(True)
            raise RuntimeError("controlled original write guard rejection")
            yield

        try:
            with self.assertRaisesRegex(RuntimeError, "original write guard"):
                await self.runtime._app_send_text(json.dumps({"type": "audio.input.ready", "segment_id": segment}),
                    write_guard=denied)
            self.assertEqual(entries, [True])
            self.assertEqual(len(self.ready_events()), original_count)
            self.assertFalse(await self.session._send_app_event("audio.webrtc.response", response_id="resp_unadmitted"))
            self.assertFalse(await self.runtime._app_send_text(json.dumps({
                "type": "audio.input.ready", "segment_id": segment, "response_id": "resp_injected"})))
            self.assertEqual(self.creates(), [])
        finally:
            self.runtime._context_replacement_active = False
            self.runtime._context_replacement_settled.set()

    async def test_input_receipt_diagnostics_are_bounded_private_and_nonthrowing(self):
        with patch("app.realtime.direct_session._NATIVE_CONTROL_LOGGER.warning") as logger:
            self.session._native_control_log_count = 0
            self.session._native_control_log_truncated = False
            for _ in range(1000):
                self.session._native_control_diagnostic("runtime_send_result", "audio.input.ready",
                    reason="private-context-secret", segment_id="private-transcript-secret",
                    send_result=False, turn_generation=1)
            records = [json.loads(call.args[1]) for call in logger.call_args_list]
            rendered = json.dumps(records)
            self.assertNotIn("private-context-secret", rendered)
            self.assertNotIn("private-transcript-secret", rendered)
            self.assertEqual(sum(r["stage"] == "truncated" for r in records), 1)
            self.assertLess(len(records), 1000)
            self.assertTrue(records[0]["segment_id_redacted"])
            self.assertIsNone(records[0]["segment_id"])
        self.session._native_control_log_count = 0
        self.session._native_control_log_truncated = False
        with patch("app.realtime.direct_session._NATIVE_CONTROL_LOGGER.warning", side_effect=RuntimeError("private")) as broken:
            await self.begin(str(uuid4()))
            broken.assert_called()
        self.assertEqual(len(self.ready_events()), 1)
