from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.realtime.gemini.client import GeminiTransportError
from app.realtime.gemini.session import GeminiAudioSession, PreparedInput
from app.realtime.models import RealtimeSessionContext
from app.settings import JarvisSettings


PCM = b"\x01\x00" * 240
ENCODED = base64.b64encode(PCM).decode()


class Phone:
    def __init__(self):
        self.input = asyncio.Queue()
        self.events = []
        self.close = AsyncMock()
        self.send_json = AsyncMock(side_effect=self.events.append)

    async def iter_text(self):
        while True:
            event = await self.input.get()
            if event is None:
                return
            yield event if isinstance(event, str) else json.dumps(event)

    async def send(self, kind, **fields):
        await self.input.put({"type": kind, "client_event_id": str(uuid4()), **fields})


class Wire:
    def __init__(self, *, config):
        self.config = config
        self.sent = []
        self.input = asyncio.Queue()
        self.connect = AsyncMock()
        self.close = AsyncMock()
        self.send_event = AsyncMock(side_effect=self.sent.append)

    async def receive_event(self):
        return await self.input.get()

    async def publish(self, content):
        await self.input.put({"serverContent": content})


async def until(predicate):
    async with asyncio.timeout(1.5):
        while not predicate():
            await asyncio.sleep(.001)


class GeminiAudioSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.phone = Phone()
        self.wires = []

        def factory(**kwargs):
            wire = Wire(**kwargs)
            self.wires.append(wire)
            return wire

        with patch.dict(os.environ, {}, clear=True):
            settings = JarvisSettings(_env_file=None, GEMINI_API_KEY="fixture-key",
                                      GEMINI_LIVE_PREPARATION_TIMEOUT_SECONDS=.1)
        self.session = GeminiAudioSession(
            websocket=self.phone,
            context=RealtimeSessionContext(
                audio_session_id="fixture-audio", owner_id="fixture-owner",
                jarvis_session_id=uuid4(), client_kind="phone_audio",
                session=SimpleNamespace(behavior_instructions="Fixture behavior."),
            ), settings=settings, repository=SimpleNamespace(), connection_factory=factory,
        )
        self.run_task = None

    async def asyncTearDown(self):
        await asyncio.wait_for(self.session.close("test_complete"), 1)
        if self.run_task is not None:
            await asyncio.wait_for(self.run_task, 1)

    def events(self, kind):
        return [event for event in self.phone.events if event["type"] == kind]

    async def start(self):
        self.run_task = asyncio.create_task(self.session.run())
        await until(lambda: bool(self.events("session.created")))

    async def append(self, segment="a"):
        await self.phone.send("audio.input.append", segment_id=segment, audio_base64=ENCODED)
        await until(lambda: segment in self.session._turns and self.session._turns[segment].input_bytes > 0)

    async def commit(self, segment="a"):
        await self.phone.send("audio.input.commit", segment_id=segment)
        await self.phone.send("response.create")
        await until(lambda: self.session._turns[segment].released)

    async def answer(self, *, text="Hello.", terminal=True):
        await self.wires[-1].publish({
            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": ENCODED}}]},
            "outputTranscription": {"text": text},
            **({"generationComplete": True, "turnComplete": True} if terminal else {}),
        })

    async def test_start_commit_and_create_map_to_exactly_one_native_release(self):
        await self.start()
        await self.append()
        await self.append()
        await self.commit()
        kinds = [next(iter(e["realtimeInput"])) for e in self.wires[0].sent]
        self.assertEqual(kinds, ["activityStart", "audio", "audio", "text", "activityEnd"])
        self.assertEqual(self.wires[0].sent[1]["realtimeInput"]["audio"]["mimeType"], "audio/pcm;rate=24000")
        self.assertEqual(len(self.events("session.created")), 1)
        timing = self.session.diagnostics()["turns"][0]
        self.assertLessEqual(timing["commit_received_at"], timing["preparation_started_at"])
        self.assertLessEqual(timing["preparation_started_at"], timing["released_at"])
        self.assertLess(timing["preparation_duration_ms"], 100)

    async def test_output_all_parts_and_terminal_events_use_stable_ids(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer()
        await until(lambda: bool(self.events("response.done")))
        self.assertEqual(self.events("audio.output.delta")[0]["item_id"], "fixture-audio:g1:r1:audio")
        turn = self.session._turns["a"]
        self.assertEqual(turn.output.forwarded_audio_bytes, len(PCM))
        self.assertEqual(turn.output.provider_audio_bytes, len(PCM))

    async def test_late_input_transcript_keeps_original_input_identity(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer()
        await until(lambda: bool(self.events("response.done")))
        await self.append("b")
        await self.wires[-1].publish({"inputTranscription": {"text": "First spoken question."}})
        await self.wires[-1].publish({"inputTranscription": {"text": "Second spoken question."}})
        await until(lambda: len(self.events("user.transcript.done")) == 2)
        self.assertEqual([e["segment_id"] for e in self.events("user.transcript.done")], ["a", "b"])

    async def test_standalone_create_empty_commit_and_malformed_pcm_do_not_open_activity(self):
        await self.start()
        await self.phone.send("response.create")
        await self.phone.send("audio.input.commit", segment_id="a")
        await self.phone.send("audio.input.append", segment_id="a", audio_base64="AA==")
        await until(lambda: len(self.events("error")) == 3)
        self.assertEqual(self.wires[0].sent, [])
        self.assertTrue(all(e["recoverable"] for e in self.events("error")))

    async def test_duplicate_commit_and_control_replay_do_not_generate_extra_response(self):
        await self.start()
        await self.append()
        await self.commit()
        event = {"type": "audio.input.commit", "segment_id": "a", "client_event_id": "duplicate"}
        await self.phone.input.put(event)
        await self.phone.input.put(event)
        await self.phone.send("response.create")
        await until(lambda: bool(self.events("error")))
        self.assertEqual(sum("activityEnd" in e["realtimeInput"] for e in self.wires[0].sent), 1)

    async def test_conflicting_mid_segment_is_rejected(self):
        await self.start()
        await self.append()
        await self.phone.send("audio.input.append", segment_id="b", audio_base64=ENCODED)
        await until(lambda: bool(self.events("error")))
        self.assertEqual(self.events("error")[-1]["code"], "conflicting_segment")
        self.assertNotIn("b", self.session._turns)
        await self.commit("a")
        self.assertTrue(self.session._turns["a"].released)

    async def test_optional_preparation_timeout_uses_truthful_fallback_within_total_budget(self):
        async def slow(turn):
            await asyncio.Event().wait()

        self.session._prepare_turn = slow
        await self.start()
        await self.append()
        await self.commit()
        turn = self.session._turns["a"]
        self.assertEqual(turn.evidence_status, "unavailable")
        context = next(e["realtimeInput"]["text"] for e in self.wires[0].sent if "text" in e["realtimeInput"])
        self.assertEqual(json.loads(context)["current_visual_evidence"], "unavailable")

    async def test_required_preparation_failure_does_not_release_or_retry(self):
        self.session._prepare_turn = AsyncMock(side_effect=GeminiTransportError("unsafe_operation_state"))
        await self.start()
        await self.append()
        await self.phone.send("audio.input.commit", segment_id="a")
        await until(lambda: self.session._closed.is_set())
        self.assertFalse(any("activityEnd" in e["realtimeInput"] for e in self.wires[0].sent))
        self.assertEqual(len(self.wires), 1)

    async def test_cancel_before_first_append_has_no_native_effect(self):
        await self.start()
        await self.phone.send("response.cancel")
        await self.append()
        self.assertEqual([next(iter(e["realtimeInput"])) for e in self.wires[0].sent], ["activityStart", "audio"])

    async def test_barge_in_drops_old_audio_and_old_terminal_does_not_finish_new_turn(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer(terminal=False)
        await until(lambda: bool(self.events("audio.output.delta")))
        await self.phone.send("response.cancel")
        await self.append("b")
        await self.commit("b")
        await self.wires[0].publish({"interrupted": True})
        await self.wires[0].publish({"turnComplete": True})
        await until(lambda: bool(self.events("response.done")))
        self.assertTrue(self.events("response.done")[0]["interrupted"])
        self.assertFalse(self.session._turns["b"].output.completed)
        await self.wires[0].publish({"turnComplete": True})
        await self.answer(text="New answer.")
        await until(lambda: len(self.events("response.done")) == 2)
        self.assertEqual(self.events("transcript.done")[-1]["text"], "New answer.")
        self.assertEqual(len(self.wires), 1)

    async def test_new_input_during_preparation_resets_same_provider_without_old_end_marker(self):
        entered = asyncio.Event()

        async def prepare(turn):
            if turn.segment_id == "a":
                entered.set()
                await asyncio.Event().wait()
            return PreparedInput("fresh context")

        self.session._prepare_turn = prepare
        await self.start()
        await self.append()
        await self.phone.send("audio.input.commit", segment_id="a")
        await entered.wait()
        await self.append("b")
        await self.commit("b")
        self.assertEqual(len(self.wires), 2)
        self.assertIs(self.wires[0].config, self.wires[1].config)
        self.assertFalse(any("activityEnd" in e["realtimeInput"] for e in self.wires[0].sent))
        self.assertEqual(sum("activityEnd" in e["realtimeInput"] for e in self.wires[1].sent), 1)
        self.assertEqual(len(self.events("session.created")), 1)

    async def test_new_input_invalidates_inflight_context_write_before_old_end(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(event):
            self.wires[0].sent.append(event)
            if "text" in event["realtimeInput"]:
                entered.set()
                await release.wait()

        self.wires[0].send_event.side_effect = send
        await self.start()
        await self.append()
        await self.phone.send("audio.input.commit", segment_id="a")
        await entered.wait()
        await self.phone.send("audio.input.append", segment_id="b", audio_base64=ENCODED)
        await until(lambda: self.session._ingress_segment == "b")
        release.set()
        await until(lambda: "b" in self.session._turns)
        self.assertFalse(any("activityEnd" in e["realtimeInput"] for e in self.wires[0].sent))

    async def test_events_from_retired_provider_generation_are_ignored(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.append("b")
        await self.commit("b")
        await self.session._queue.put(("provider", (1, {"serverContent": {"turnComplete": True}})))
        await self.session._queue.put(("provider_failure", (1, "old_connection_failure")))
        await self.answer()
        await until(lambda: self.session._turns["b"].output.completed)
        self.assertEqual(self.events("response.done")[-1]["response_id"], "fixture-audio:g2:r2")

    async def test_playback_truncation_is_local_evidence_not_an_openai_command(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer()
        await until(lambda: bool(self.events("response.done")))
        sent = len(self.wires[0].sent)
        await self.phone.send("playback.truncated", item_id="fixture-audio:g1:r1:audio", audio_end_ms=5)
        await until(lambda: self.session._turns["a"].playback_end_ms == 5)
        self.assertEqual(len(self.wires[0].sent), sent)
        await self.phone.send("playback.truncated", item_id="fixture-audio:g1:r1:audio", audio_end_ms=999999)
        await until(lambda: bool(self.events("error")))
        self.assertEqual(self.session._turns["a"].playback_end_ms, 5)

    async def test_close_during_setup_cancels_owned_connection_task(self):
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.wires[0].connect.side_effect = blocked
        self.run_task = asyncio.create_task(self.session.run())
        await entered.wait()
        await self.phone.send("session.close")
        await asyncio.wait_for(self.run_task, 1)
        self.assertEqual(self.events("session.created"), [])
        self.assertEqual(self.session.diagnostics()["owned_tasks"], 0)

    async def test_disconnect_mid_segment_closes_without_answer_release(self):
        await self.start()
        await self.append()
        await self.phone.input.put(None)
        await asyncio.wait_for(self.run_task, 1)
        self.assertFalse(any("activityEnd" in e["realtimeInput"] for e in self.wires[0].sent))
        self.phone.close.assert_awaited_once()
        self.assertEqual(self.session.diagnostics()["owned_tasks"], 0)

    async def test_failed_context_transport_is_terminal_and_not_reported_delivered(self):
        async def send(event):
            if "text" in event["realtimeInput"]:
                raise GeminiTransportError("gemini_write_failed")
            self.wires[0].sent.append(event)

        self.wires[0].send_event.side_effect = send
        await self.start()
        await self.append()
        await self.phone.send("audio.input.commit", segment_id="a")
        await until(lambda: self.session._closed.is_set())
        self.assertIsNone(self.session._turns["a"].evidence_status)
        self.assertFalse(self.session._turns["a"].released)

    async def test_response_timeout_is_terminal_and_bounded(self):
        self.session.config = replace(self.session.config, response_timeout_seconds=.02)
        await self.start()
        await self.append()
        await self.commit()
        await until(lambda: self.session._closed.is_set())
        self.assertEqual(self.events("error")[-1]["code"], "gemini_response_timeout")

    async def test_invalid_new_segment_does_not_cancel_valid_preparation(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def prepare(turn):
            entered.set()
            await release.wait()
            return PreparedInput("valid context")

        self.session._prepare_turn = prepare
        await self.start()
        await self.append()
        await self.phone.send("audio.input.commit", segment_id="a")
        await entered.wait()
        await self.phone.send("audio.input.append", segment_id="b", audio_base64="AA==")
        await until(lambda: bool(self.events("error")))
        release.set()
        await until(lambda: self.session._turns["a"].released)
        self.assertEqual(len(self.wires), 1)

    async def test_late_cancel_after_new_activity_does_not_cancel_new_input(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer(terminal=False)
        await until(lambda: self.session._turns["a"].provider_started)
        await self.append("b")
        await self.phone.send("response.cancel")
        await self.commit("b")
        self.assertFalse(self.session._turns["b"].cancelled)

    async def test_history_retains_logical_order_when_input_arrives_after_output(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer()
        await until(lambda: bool(self.events("response.done")))
        await self.wires[0].publish({"inputTranscription": {"text": "Earlier question."}})
        await until(lambda: bool(self.events("user.transcript.done")))
        history = self.session._conversation_history()
        self.assertEqual([item["role"] for item in history], ["user", "model"])
        self.assertEqual(history[0]["parts"][0]["text"], "Earlier question.")

    async def test_assistant_persistence_hook_waits_for_interruption_or_turn_settlement(self):
        finalized = AsyncMock()
        self.session._assistant_transcript_finalized = finalized
        await self.start()
        await self.append()
        await self.commit()
        await self.answer(terminal=False)
        await self.wires[0].publish({"generationComplete": True})
        await until(lambda: bool(self.events("transcript.done")))
        finalized.assert_not_awaited()
        await self.wires[0].publish({"interrupted": True})
        await self.wires[0].publish({"turnComplete": True})
        await until(lambda: finalized.await_count == 1)
        self.assertTrue(finalized.call_args.args[0].output.interrupted)

    async def test_replayed_cancel_cannot_invalidate_a_later_turn(self):
        await self.start()
        await self.append()
        await self.commit()
        await self.answer(terminal=False)
        await until(lambda: self.session._turns["a"].provider_started)
        cancel = {"type": "response.cancel", "client_event_id": "original-cancel"}
        await self.phone.input.put(cancel)
        await self.append("b")
        await self.commit("b")
        await self.phone.input.put(cancel)
        await self.wires[0].publish({"interrupted": True, "turnComplete": True})
        await self.answer(text="Fresh answer.")
        await until(lambda: self.session._turns["b"].output.completed)
        self.assertFalse(self.session._turns["b"].output.interrupted)

    async def test_backpressure_is_bounded_and_external_stop_cancels_blocked_writer(self):
        entered = asyncio.Event()

        async def blocked(event):
            if "audio" in event["realtimeInput"]:
                entered.set()
                await asyncio.Event().wait()
            self.wires[0].sent.append(event)

        self.wires[0].send_event.side_effect = blocked
        await self.start()
        for _ in range(70):
            await self.phone.send("audio.input.append", segment_id="a", audio_base64=ENCODED)
        await entered.wait()
        await until(lambda: self.session._queue.full())
        self.assertEqual(self.session._queue.qsize(), 32)
        await asyncio.wait_for(self.session.close("external_stop"), 1)
        await asyncio.wait_for(self.run_task, 1)
        self.assertEqual(self.session.diagnostics()["owned_tasks"], 0)
