from __future__ import annotations

import asyncio
import base64
from dataclasses import FrozenInstanceError, replace
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from app.errors import JarvisHTTPError
from app.realtime.gemini.client import (
    GeminiLiveConnection, GeminiTransportError, LIVE_ENDPOINT, build_setup_event,
)
from app.realtime.gemini.config import GeminiLiveConfig
from app.realtime.gemini.events import GeminiOutputTurn, decode_pcm16
from app.settings import JarvisSettings


class FakeSocket:
    def __init__(self) -> None:
        self.events: asyncio.Queue = asyncio.Queue()
        self.events.put_nowait(json.dumps({"setupComplete": {}}))
        self.sent: list[dict] = []
        self.send = AsyncMock(side_effect=self._send)
        self.close = AsyncMock()

    async def _send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        value = await self.events.get()
        if isinstance(value, BaseException):
            raise value
        return value


class GeminiConfigTests(unittest.TestCase):
    def test_configuration_is_immutable_and_redacts_secret(self):
        config = GeminiLiveConfig(api_key="fixture-secret")
        self.assertNotIn("fixture-secret", repr(config))
        with self.assertRaises(FrozenInstanceError):
            config.model = "changed"

    def test_invalid_unused_budget_does_not_break_global_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = JarvisSettings(
                _env_file=None, GEMINI_API_KEY="fixture-key",
                GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS="invalid",
            )
        with self.assertRaises(JarvisHTTPError) as error:
            GeminiLiveConfig.from_settings(settings)
        self.assertEqual(error.exception.code, "gemini_live_not_configured")

    def test_existing_capture_rate_and_configured_budget_are_preserved(self):
        with patch.dict(os.environ, {}, clear=True):
            config = GeminiLiveConfig.from_settings(JarvisSettings(
                _env_file=None, GEMINI_API_KEY="fixture-key",
                GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS="0.4",
            ))
        self.assertEqual(config.input_sample_rate, 24000)
        self.assertEqual(config.preparation_timeout_seconds, 0.4)

    def test_invalid_limits_rejected(self):
        config = GeminiLiveConfig(api_key="fixture-key")
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(config, preparation_timeout_seconds=value)

    def test_native_setup_has_manual_activity_and_nonresponding_initial_history(self):
        config = GeminiLiveConfig(api_key="fixture-key")
        setup = build_setup_event(config, instructions="fixture") ["setup"]
        self.assertEqual(setup["model"], "models/gemini-3.1-flash-live-preview")
        self.assertEqual(setup["realtimeInputConfig"]["automaticActivityDetection"], {"disabled": True})
        self.assertEqual(setup["historyConfig"], {"initialHistoryInClientContent": True})
        self.assertNotIn("tools", setup)
        self.assertNotIn("fixture-key", json.dumps(setup))
        resumed = build_setup_event(config, instructions="fixture", resume_handle="handle")["setup"]
        self.assertEqual(resumed["sessionResumption"], {"handle": "handle"})
        self.assertNotIn("historyConfig", resumed)


class GeminiTransportTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.socket = FakeSocket()
        self.connector = AsyncMock(return_value=self.socket)
        self.patch = patch("app.realtime.gemini.client.websockets.connect", self.connector)
        self.patch.start()
        self.connection = GeminiLiveConnection(config=GeminiLiveConfig(
            api_key="fixture-secret", setup_timeout_seconds=.2,
            write_timeout_seconds=.05, close_timeout_seconds=.05,
        ))

    async def asyncTearDown(self):
        await self.connection.close()
        self.patch.stop()

    async def test_connect_handshake_history_then_audio(self):
        await self.connection.connect(instructions="fixture")
        await self.connection.send_event({"realtimeInput": {"activityStart": {}}})
        self.assertEqual([next(iter(e)) for e in self.socket.sent], ["setup", "clientContent", "realtimeInput"])
        self.assertTrue(self.connection.ready)
        self.assertEqual(self.connector.call_args.args, (LIVE_ENDPOINT,))
        self.assertEqual(self.connector.call_args.kwargs["additional_headers"], {"x-goog-api-key": "fixture-secret"})
        self.assertEqual(self.connector.call_args.kwargs["max_queue"], 16)

    async def test_media_cannot_be_sent_before_setup(self):
        with self.assertRaises(GeminiTransportError):
            await self.connection.send_event({"realtimeInput": {"activityStart": {}}})
        self.socket.send.assert_not_awaited()

    async def test_resume_does_not_replay_history(self):
        await self.connection.connect(instructions="fixture", resume_handle="fixture-handle")
        self.assertEqual(len(self.socket.sent), 1)
        self.assertNotIn("historyConfig", self.socket.sent[0]["setup"])

    async def test_resume_with_fresh_history_is_rejected_before_connect(self):
        with self.assertRaises(GeminiTransportError):
            await self.connection.connect(instructions="fixture", history=[{}], resume_handle="handle")
        self.connector.assert_not_awaited()

    async def test_setup_side_events_are_delivered_once(self):
        self.socket.events = asyncio.Queue()
        update = {"sessionResumptionUpdate": {"resumable": True, "newHandle": "fixture"}}
        for event in (update, {"setupComplete": {}}, {"serverContent": {"turnComplete": True}}):
            self.socket.events.put_nowait(json.dumps(event))
        await self.connection.connect(instructions="fixture")
        self.assertEqual(await self.connection.receive_event(), update)
        self.assertIn("serverContent", await self.connection.receive_event())

    async def test_setup_error_is_redacted_and_socket_closed(self):
        self.socket.events = asyncio.Queue()
        self.socket.events.put_nowait(json.dumps({"error": {"message": "fixture-secret"}}))
        with self.assertRaises(GeminiTransportError) as error:
            await self.connection.connect(instructions="fixture")
        self.assertEqual(str(error.exception), "gemini_provider_error")
        self.socket.close.assert_awaited_once()

    async def test_setup_timeout_is_bounded(self):
        self.socket.events = asyncio.Queue()
        with self.assertRaises(GeminiTransportError):
            await self.connection.connect(instructions="fixture")
        self.socket.close.assert_awaited_once()

    async def test_setup_cancellation_closes_connection(self):
        self.socket.events = asyncio.Queue()
        sent = asyncio.Event()
        self.socket.send.side_effect = lambda _: sent.set()
        task = asyncio.create_task(self.connection.connect(instructions="fixture"))
        await asyncio.wait_for(sent.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.socket.close.assert_awaited_once()
        self.assertFalse(self.connection.ready)

    async def test_close_while_dialing_closes_late_socket(self):
        dialing, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args, **kwargs):
            dialing.set()
            await release.wait()
            return self.socket

        self.connector.side_effect = delayed
        task = asyncio.create_task(self.connection.connect(instructions="fixture"))
        await dialing.wait()
        await self.connection.close()
        release.set()
        with self.assertRaises(GeminiTransportError):
            await task
        self.socket.close.assert_awaited_once()
        self.socket.send.assert_not_awaited()

    async def test_failed_write_is_terminal_without_retry(self):
        await self.connection.connect(instructions="fixture")
        self.socket.send.side_effect = OSError("fixture-secret")
        with self.assertRaises(GeminiTransportError) as error:
            await self.connection.send_event({"realtimeInput": {"activityStart": {}}})
        self.assertEqual(str(error.exception), "gemini_write_failed")
        self.assertFalse(self.connection.ready)
        self.socket.close.assert_awaited_once()
        self.assertEqual(self.connector.await_count, 1)

    async def test_cancelled_write_is_terminal(self):
        await self.connection.connect(instructions="fixture")
        entered, hold = asyncio.Event(), asyncio.Event()

        async def blocked(raw):
            entered.set()
            await hold.wait()

        self.socket.send.side_effect = blocked
        task = asyncio.create_task(self.connection.send_event({"realtimeInput": {"activityStart": {}}}))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.connection.ready)
        self.socket.close.assert_awaited_once()

    async def test_write_timeout_is_terminal(self):
        await self.connection.connect(instructions="fixture")

        async def blocked(raw):
            await asyncio.Event().wait()

        self.socket.send.side_effect = blocked
        with self.assertRaises(GeminiTransportError):
            await self.connection.send_event({"realtimeInput": {"activityStart": {}}})
        self.assertFalse(self.connection.ready)

    async def test_malformed_provider_json_closes_without_echo(self):
        await self.connection.connect(instructions="fixture")
        self.socket.events.put_nowait("fixture-secret not json")
        with self.assertRaises(GeminiTransportError) as error:
            await self.connection.receive_event()
        self.assertNotIn("fixture-secret", str(error.exception))
        self.socket.close.assert_awaited_once()

    async def test_close_is_idempotent_and_connection_cannot_be_reused(self):
        await self.connection.connect(instructions="fixture")
        await asyncio.gather(self.connection.close(), self.connection.close())
        self.socket.close.assert_awaited_once()
        with self.assertRaises(GeminiTransportError):
            await self.connection.connect(instructions="fixture")


def audio_part(pcm: bytes = b"\x01\x00\x02\x00", mime: str = "audio/pcm;rate=24000") -> dict:
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(pcm).decode()}}


class GeminiOutputTests(unittest.TestCase):
    def setUp(self):
        self.turn = GeminiOutputTurn("run", 1, 1)

    def test_all_parts_and_transcription_in_one_native_event_are_preserved(self):
        events = self.turn.consume({"serverContent": {
            "modelTurn": {"parts": [audio_part(), {"text": "not speech"}, audio_part()]},
            "outputTranscription": {"text": "Hello."},
            "generationComplete": True, "turnComplete": True,
        }, "toolCall": {"functionCalls": []}})
        self.assertEqual([e["type"] for e in events], [
            "transcript.delta", "audio.output.delta", "audio.output.delta",
            "audio.output.done", "transcript.done", "response.done",
        ])
        for event in events:
            self.turn.record_forwarded(event)
        evidence = self.turn.diagnostics()
        self.assertEqual(evidence["provider_audio_bytes"], 8)
        self.assertEqual(evidence["forwarded_audio_bytes"], 8)
        self.assertEqual(evidence["provider_audio_sha256"], evidence["forwarded_audio_sha256"])
        self.assertEqual({e["item_id"] for e in events}, {"run:g1:r1:audio"})

    def test_forwarded_counts_require_an_actual_send_acknowledgement(self):
        events = self.turn.consume({"serverContent": {"modelTurn": {"parts": [audio_part()]}}})
        self.assertEqual(self.turn.forwarded_audio_bytes, 0)
        self.turn.record_forwarded(events[0])
        self.assertEqual(self.turn.forwarded_audio_bytes, 4)
        with self.assertRaises(ValueError):
            self.turn.record_forwarded(events[0])

    def test_interruption_remains_sticky_until_separate_terminal_message(self):
        initial = self.turn.consume({"serverContent": {"interrupted": True}})
        late = self.turn.consume({"serverContent": {"modelTurn": {"parts": [audio_part()]}}})
        final = self.turn.consume({"serverContent": {"turnComplete": True}})
        self.assertEqual(initial[0]["type"], "audio.output.clear")
        self.assertEqual(late, [])
        self.assertEqual(final[-1]["status"], "cancelled")
        self.assertEqual(self.turn.provider_audio_bytes, 4)
        self.assertEqual(self.turn.forwarded_audio_bytes, 0)

    def test_generation_done_does_not_complete_response(self):
        first = self.turn.consume({"serverContent": {"generationComplete": True}})
        self.assertFalse(self.turn.completed)
        self.assertEqual([e["type"] for e in first], ["audio.output.done"])
        final = self.turn.consume({"serverContent": {"turnComplete": True}})
        self.assertEqual([e["type"] for e in final], ["response.done"])
        self.assertEqual(self.turn.consume({"serverContent": {"turnComplete": True}}), [])

    def test_old_terminal_does_not_mutate_a_new_output_instance(self):
        other = GeminiOutputTurn("run", 1, 2)
        self.turn.interrupt(source="client")
        self.turn.consume({"serverContent": {"turnComplete": True}})
        self.assertFalse(other.completed)
        self.assertNotEqual(other.item_id, self.turn.item_id)
        self.assertNotEqual(GeminiOutputTurn("run", 2, 1).item_id, self.turn.item_id)

    def test_pcm_validation_rejects_empty_partial_oversized_and_wrong_encoding(self):
        for data in ("", "!", "AA==", base64.b64encode(b"123456").decode()):
            with self.subTest(data=data), self.assertRaises(ValueError):
                decode_pcm16(data, max_bytes=4)

    def test_wrong_native_audio_format_fails_instead_of_misplaying(self):
        with self.assertRaises(GeminiTransportError):
            self.turn.consume({"serverContent": {"modelTurn": {"parts": [audio_part(mime="audio/pcm;rate=16000")]}}})

    def test_playback_or_heard_status_is_never_inferred_from_turn_completion(self):
        events = self.turn.consume({"serverContent": {
            "outputTranscription": {"text": "Some generated speech."}, "turnComplete": True,
        }})
        final = next(e for e in events if e["type"] == "transcript.done")
        self.assertEqual(final["delivery_status"], "generated")
        self.assertEqual(self.turn.forwarded_audio_bytes, 0)
