from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import inspect
import unittest
from unittest.mock import patch

from app.realtime.gemini.client import GeminiTransportError, build_setup_event
from app.realtime.gemini.recovery import GeminiRecovery
from app.realtime.gemini.session import GeminiAudioSession
from evals.gemini_live.session_probe import ProbeExperienceService, ProbeMessageRepository
from test_gemini_audio_session import ENCODED, Phone, Wire, until
from test_gemini_domain_integration import context, settings


class GeminiRecoveryStateTests(unittest.TestCase):
    def test_only_safe_checkpoints_are_retained_and_never_exposed(self):
        state = GeminiRecovery()
        state.observe({"resumable": True, "newHandle": "private-handle"}, checkpoint_safe=False)
        self.assertIsNone(state.handle)
        state.observe({"resumable": True, "newHandle": "private-handle"}, checkpoint_safe=True)
        self.assertNotIn("private-handle", json.dumps(state.diagnostics()))
        state.observe({"resumable": False}, checkpoint_safe=True)
        self.assertIsNone(state.handle)

    def test_goaway_deadline_is_bounded_and_cannot_be_extended(self):
        state = GeminiRecovery()
        self.assertEqual(state.request({"timeLeft": "10s"}, now=1, max_wait=5), 6)
        self.assertEqual(state.request({"timeLeft": "100s"}, now=2, max_wait=5), 6)
        for value in (None, "NaNs", "infs", "-1s", "0s", 5, "tomorrow"):
            with self.subTest(value=value), self.assertRaises(GeminiTransportError):
                state.request({"timeLeft": value}, now=1, max_wait=5)

    def test_deferred_input_is_bounded_and_release_preserves_order(self):
        state = GeminiRecovery()
        with patch("app.realtime.gemini.recovery.MAX_DEFERRED_EVENTS", 2):
            state.defer(({"type": "first"}, 1, 0.0))
            state.defer(({"type": "second"}, 1, 0.1))
            with self.assertRaises(GeminiTransportError):
                state.defer(({"type": "third"}, 1, 0.2))
        self.assertEqual([item[0]["type"] for item in state.release()], ["first", "second"])
        self.assertEqual(state.diagnostics()["deferred_bytes"], 0)


class GeminiRecoverySessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.phone, self.wires = Phone(), []
        self.factory_hook = None

        def factory(**kwargs):
            wire = Wire(**kwargs)
            self.wires.append(wire)
            if self.factory_hook:
                self.factory_hook(wire)
            return wire

        repository = ProbeMessageRepository()
        self.session = GeminiAudioSession(
            websocket=self.phone, context=context(), settings=settings(), repository=repository,
            experience_event_service=ProbeExperienceService(repository), connection_factory=factory,
        )
        self.running = asyncio.create_task(self.session.run())
        await until(lambda: any(e["type"] == "session.created" for e in self.phone.events))

    async def asyncTearDown(self):
        await self.session.close("recovery_test_done")
        await self.running

    async def checkpoint(self, handle="private-checkpoint"):
        await self.wires[-1].input.put({"sessionResumptionUpdate": {"resumable": True, "newHandle": handle}})
        await until(lambda: self.session.recovery.handle == handle)

    async def begin_turn(self, segment="a", commit=True):
        await self.phone.send("audio.input.append", segment_id=segment, audio_base64=ENCODED)
        await until(lambda: segment in self.session._turns)
        if commit:
            await self.phone.send("audio.input.commit", segment_id=segment)
            await self.phone.send("response.create")
            await until(lambda: self.session._turns[segment].released)

    async def finish_turn(self, segment="a"):
        await self.wires[-1].publish({"inputTranscription": {"text": "An ordinary question."}})
        await self.wires[-1].publish({
            "modelTurn": {"parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": ENCODED}}]},
            "outputTranscription": {"text": "An answer."}, "generationComplete": True, "turnComplete": True,
        })
        await until(lambda: self.session._turns[segment].output.completed)

    async def test_idle_goaway_resumes_same_setup_without_replaying_history_or_audio(self):
        await self.begin_turn()
        await self.finish_turn()
        old_setup = self.wires[0].connect.await_args.kwargs
        await self.checkpoint()
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await until(lambda: self.session.recovery.resumptions == 1)
        resumed = self.wires[1].connect.await_args.kwargs
        self.assertEqual(resumed["resume_handle"], "private-checkpoint")
        self.assertEqual(resumed["instructions"], old_setup["instructions"])
        self.assertEqual(resumed["tools"], old_setup["tools"])
        self.assertNotIn("history", resumed)
        self.assertEqual(self.wires[1].sent, [])
        self.assertEqual(self.session._native_namespace, 1)
        self.assertEqual(self.session._connection_generation, 2)
        self.assertEqual(sum(e["type"] == "session.created" for e in self.phone.events), 1)
        self.assertNotIn("private-checkpoint", json.dumps(self.session.diagnostics()))

    async def test_current_activity_can_finish_and_new_input_waits_for_resume(self):
        await self.begin_turn(commit=False)
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await until(lambda: self.session.recovery.pending)
        await self.phone.send("audio.input.commit", segment_id="a")
        await self.phone.send("response.create")
        await until(lambda: self.session._turns["a"].released)
        await self.wires[0].publish({"modelTurn": {"parts": [{"inlineData": {
            "mimeType": "audio/pcm;rate=24000", "data": ENCODED,
        }}]}})
        await until(lambda: self.session._turns["a"].provider_started)
        old_audio_count = sum("audio" in e.get("realtimeInput", {}) for e in self.wires[0].sent)
        await self.phone.send("audio.input.append", segment_id="b", audio_base64=ENCODED)
        await until(lambda: self.session.recovery.has_deferred)
        self.assertNotIn("b", self.session._turns)
        await self.finish_turn()
        # During a pending go-away, a safe checkpoint can be consumed by the
        # reconnection before checkpoint() observes its transient handle.
        # Synchronize on the durable transition and inspect the submitted handle.
        await self.wires[0].input.put({"sessionResumptionUpdate": {
            "resumable": True, "newHandle": "private-checkpoint",
        }})
        await until(lambda: self.session.recovery.resumptions == 1)
        await until(lambda: "b" in self.session._turns)
        self.assertEqual(self.wires[1].connect.await_args.kwargs["resume_handle"], "private-checkpoint")
        self.assertEqual(sum("audio" in e.get("realtimeInput", {}) for e in self.wires[0].sent), old_audio_count)
        self.assertEqual(sum("audio" in e.get("realtimeInput", {}) for e in self.wires[1].sent), 1)

    async def test_disconnect_during_unacknowledged_input_fails_without_replay(self):
        await self.checkpoint()
        await self.begin_turn(commit=False)
        self.assertIsNone(self.session.recovery.handle)
        await self.session._queue.put(("provider_failure", (1, "gemini_receive_failed")))
        await until(self.session._closed.is_set)
        self.assertEqual(len(self.wires), 1)
        self.assertEqual(self.session.diagnostics()["close_reason"], "gemini_recovery_not_safe")

    async def test_idle_disconnect_with_checkpoint_resumes(self):
        await self.checkpoint()
        await self.session._queue.put(("provider_failure", (1, "gemini_receive_failed")))
        await until(lambda: self.session.recovery.resumptions == 1)
        self.assertEqual(self.wires[1].connect.await_args.kwargs["resume_handle"], "private-checkpoint")

    async def test_provider_protocol_error_is_not_retried_as_a_network_disconnect(self):
        await self.checkpoint()
        await self.session._queue.put(("provider_failure", (1, "gemini_invalid_provider_event")))
        await until(self.session._closed.is_set)
        self.assertEqual(len(self.wires), 1)
        self.assertEqual(self.session.diagnostics()["close_reason"], "gemini_invalid_provider_event")

    async def test_cancelling_owned_task_before_start_closes_its_coroutine(self):
        operation = asyncio.sleep(100)
        task = self.session._spawn(operation)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertEqual(inspect.getcoroutinestate(operation), inspect.CORO_CLOSED)
        self.assertNotIn(task, self.session._tasks)

    async def test_resume_retry_is_bounded_and_never_becomes_a_fresh_session(self):
        await self.checkpoint()
        self.factory_hook = lambda wire: setattr(wire.connect, "side_effect", GeminiTransportError("gemini_setup_failed"))
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await until(self.session._closed.is_set)
        self.assertEqual(len(self.wires), 3)
        for wire in self.wires[1:]:
            self.assertEqual(wire.connect.await_args.kwargs["resume_handle"], "private-checkpoint")
            self.assertNotIn("history", wire.connect.await_args.kwargs)
            self.assertEqual(wire.sent, [])
        self.assertEqual(self.session.diagnostics()["close_reason"], "gemini_resume_failed")

    async def test_stale_provider_events_after_resume_cannot_complete_new_output(self):
        await self.checkpoint()
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await until(lambda: self.session.recovery.resumptions == 1)
        await self.begin_turn("new")
        await self.session._queue.put(("provider", (1, {"serverContent": {"interrupted": True, "turnComplete": True}})))
        await self.session._queue.put(("provider_failure", (1, "gemini_receive_failed")))
        await self.wires[-1].publish({"inputTranscription": {"text": "New question."}})
        await until(lambda: self.session._turns["new"].input_finalized)
        self.assertFalse(self.session._turns["new"].output.completed)
        self.assertFalse(self.session._closed.is_set())

    async def test_unknown_checkpoint_deadline_closes_without_fabricated_resume(self):
        await self.wires[0].input.put({"goAway": {"timeLeft": "0.02s"}})
        await until(self.session._closed.is_set)
        self.assertEqual(len(self.wires), 1)
        self.assertEqual(self.session.diagnostics()["close_reason"], "gemini_recovery_deadline")

    async def test_stop_during_resume_setup_joins_connection_task(self):
        entered = asyncio.Event()

        async def blocked(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        await self.checkpoint()
        self.factory_hook = lambda wire: setattr(wire.connect, "side_effect", blocked)
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await entered.wait()
        await self.session.close("stop_during_resume")
        await self.running
        self.assertEqual(self.session.diagnostics()["owned_tasks"], 0)
        self.assertEqual(self.session.recovery.diagnostics()["deferred_events"], 0)

    async def test_resumption_keeps_native_call_identity_but_fresh_reset_changes_it(self):
        args = {"id": "same-native-id", "name": "remember", "args": {"content": "fixture"}}
        original, created = self.session.operations.receive(args, namespace=self.session._native_namespace, turn_number=1)
        await self.checkpoint()
        await self.wires[0].input.put({"goAway": {"timeLeft": "2s"}})
        await until(lambda: self.session.recovery.resumptions == 1)
        resumed, created = self.session.operations.receive(args, namespace=self.session._native_namespace, turn_number=2)
        self.assertFalse(created)
        self.assertIs(resumed, original)
        await self.session._reset_before_new_input()
        fresh, created = self.session.operations.receive(args, namespace=self.session._native_namespace, turn_number=3)
        self.assertTrue(created)
        self.assertNotEqual(fresh.execution_id, original.execution_id)

    async def test_context_compression_is_enabled_without_replacing_native_defaults(self):
        setup = build_setup_event(self.session.config, instructions="fixture", resume_handle="private")
        self.assertEqual(setup["setup"]["contextWindowCompression"], {"slidingWindow": {}})
        self.assertNotIn("historyConfig", setup["setup"])
