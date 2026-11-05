"""Offline notification boundaries, not provider or audible acceptance."""
import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.live_state.models import SourceDependency
from app.realtime.lean_attention import LeanNativeAttention
from app.voice_routes.contracts import (
    DeliveryAttemptReference, DeliveryEvidenceState as Evidence,
    InterruptSpeechCommand, NotificationOffer,
)
import test_lean_voice_session as fixtures


class LeanNativeAttentionTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.LeanSessionTests.asyncSetUp
    asyncTearDown = fixtures.LeanSessionTests.asyncTearDown

    async def test_completed_ordinary_turn_releases_real_attention_admission(self):
        self.setup_offer()
        self.session._attention_delivery = None
        self.session._realtime_context_ready.set()
        repository = SimpleNamespace(claim_next=AsyncMock(return_value=None))
        await fixtures.LeanSessionTests.created(self)
        await fixtures.LeanSessionTests.done(self)
        await self.session._finish_native_response("resp_test")
        self.assertFalse(self.session._active_responses)
        await self.bridge._admit(repository, None, worker_id="fixture", receipt_seconds=1)
        repository.claim_next.assert_awaited_once()

    async def test_successor_buffer_drain_releases_missing_predecessor_stop(self):
        self.setup_offer()
        self.session._attention_delivery = None
        self.session._realtime_context_ready.set()
        repository = SimpleNamespace(claim_next=AsyncMock(return_value=None))
        await fixtures.LeanSessionTests.created(self)
        self.event("output_audio_buffer.started", response_id="resp_test")
        await fixtures.LeanSessionTests.done(self)
        await self.session._finish_native_response("resp_test")
        self.event("output_audio_buffer.started", response_id="continuation")
        self.event("response.done", response_id="continuation",
                   response={"id": "continuation", "status": "completed", "output": [{"type": "message"}]})
        await self.bridge._admit(repository, None, worker_id="fixture", receipt_seconds=1)
        repository.claim_next.assert_not_awaited()
        self.event("output_audio_buffer.stopped", response_id="continuation")
        self.assertFalse(self.bridge._playing)
        await self.bridge._admit(repository, None, worker_id="fixture", receipt_seconds=1)
        repository.claim_next.assert_awaited_once()

    async def test_older_or_unknown_stop_cannot_clear_newer_output(self):
        self.setup_offer()
        self.event("output_audio_buffer.started", response_id="older")
        self.event("output_audio_buffer.started", response_id="newer")
        self.event("output_audio_buffer.started", response_id="older")
        self.event("output_audio_buffer.stopped", response_id="older")
        self.event("output_audio_buffer.cleared", response_id="older")
        self.event("output_audio_buffer.stopped", response_id="unknown")
        self.assertEqual(set(self.bridge._playing), {"newer"})
        self.assertFalse(self.bridge._idle())
        self.event("output_audio_buffer.stopped", response_id="newer")
        self.assertTrue(self.bridge._idle())

    async def test_shared_buffer_drain_does_not_fabricate_owned_notification_receipt(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.event("output_audio_buffer.started")
        self.event("response.done", response={"id": "notification_response", "status": "completed",
                                              "output": [{"type": "message"}]})
        self.event("output_audio_buffer.started", response_id="ordinary_successor")
        self.event("output_audio_buffer.stopped", response_id="ordinary_successor")
        self.assertFalse(self.bridge._playing)
        self.assertFalse(self.bridge._offers[self.offer.offer_id].stopped.is_set())
        self.assertNotIn(Evidence.OUTPUT_FINISHED,
                        [call.args[0].state for call in self.bridge.sink.on_event.call_args_list])

    def setup_offer(self):
        self.session._lean_provider_ready = True
        self.bridge = LeanNativeAttention(self.session, SimpleNamespace(tasks=object()))
        self.session._lean_attention = self.bridge
        self.bridge.sink.on_event = Mock()
        self.bridge.sink.close = AsyncMock()
        self.offer = NotificationOffer(route=self.bridge.request.route, offer_id=uuid4(),
            attempt=DeliveryAttemptReference(notification_intent_id=uuid4(), attempt_id=uuid4(),
                attempt_generation=1, source_payload_id=uuid4(),
                source=SourceDependency(source_namespace="task_result", source_id=uuid4(), revision=1)),
            text="The requested note update committed as version 2.",
            offered_at=self.session._now(), expires_at=self.session._now() + timedelta(seconds=60))
        self.delivery = SimpleNamespace(offer=self.offer, owns_offer=lambda offer: offer == self.offer,
            sink=self.bridge.sink, request_stop=Mock(), stop=AsyncMock(), task=None)
        self.session._attention_delivery = self.delivery
        self.guard_count = 0
        @asynccontextmanager
        async def guard(offer):
            self.assertEqual(offer, self.offer)
            self.guard_count += 1
            yield
        self.bridge.sink.guard = guard

    def created(self, response_id="notification_response", metadata=None):
        state = self.bridge._offers[self.offer.offer_id]
        self.bridge.observe({"type": "response.created", "response": {
            "id": response_id, "metadata": metadata if metadata is not None else {
                "jarvis_offer_id": str(self.offer.offer_id),
                "jarvis_attempt_id": str(self.offer.attempt.attempt_id),
                "jarvis_request_id": state.request_id}}})

    def event(self, kind, response_id="notification_response", **extra):
        self.bridge.observe({"type": kind, "response_id": response_id, **extra})

    async def test_result_is_assistant_data_then_correlated_single_voice_response(self):
        self.setup_offer()
        result = await self.bridge.offer_notification(self.offer)
        self.assertEqual(result.state, Evidence.SUBMITTED)
        self.assertEqual(self.guard_count, 1)
        context, response = self.provider.sent
        self.assertEqual(context["item"]["role"], "assistant")
        self.assertEqual(context["item"]["type"], "message")
        self.assertIn("not yet spoken", context["item"]["content"][0]["text"])
        self.assertEqual(response["response"]["tools"], [])
        self.assertEqual(response["response"]["metadata"]["jarvis_offer_id"], str(self.offer.offer_id))
        self.assertTrue(self.session._source_context.has_sources)

    async def test_duplicate_offer_cannot_resend(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        from app.attention.models import AttentionError
        with self.assertRaises(AttentionError):
            await self.bridge.offer_notification(self.offer)
        self.assertEqual(len(self.provider.sent), 2)

    async def test_busy_user_leaves_no_result_or_response_write(self):
        self.setup_offer()
        self.session._user_audio_pending = True
        result = await self.bridge.offer_notification(self.offer)
        self.assertEqual(result.state, Evidence.UNKNOWN)
        self.assertEqual(self.provider.sent, [])

    async def test_speech_between_context_and_response_prevents_new_speech(self):
        self.setup_offer()
        original = self.provider.send_event
        async def send(event, **kwargs):
            written = await original(event, **kwargs)
            self.session._turn_generation += 1
            return written
        self.provider.send_event = send
        result = await self.bridge.offer_notification(self.offer)
        self.assertEqual(result.state, Evidence.UNKNOWN)
        self.assertEqual(len(self.provider.sent), 1)

    async def test_actual_writer_rechecks_speech_after_queue_wait(self):
        self.setup_offer()
        original = self.provider.send_event
        async def send(event, **kwargs):
            self.session._user_audio_pending = True
            return await original(event, **kwargs)
        self.provider.send_event = send
        await self.bridge.offer_notification(self.offer)
        self.assertEqual(self.provider.sent, [])

    async def test_wrong_metadata_cannot_claim_an_ordinary_response(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created(metadata={"jarvis_offer_id": str(self.offer.offer_id)})
        self.assertFalse(self.bridge._responses)
        self.bridge.sink.on_event.assert_not_called()

    async def test_provider_completion_is_not_playback_completion(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.event("output_audio_buffer.started")
        self.event("response.done", response={"id": "notification_response", "status": "completed",
            "output": [{"type": "message"}]})
        self.assertNotIn(Evidence.OUTPUT_FINISHED,
            [call.args[0].state for call in self.bridge.sink.on_event.call_args_list])
        self.event("output_audio_buffer.stopped")
        evidence = self.bridge.sink.on_event.call_args.args[0]
        self.assertEqual(evidence.state, Evidence.OUTPUT_FINISHED)
        self.assertIsNone(evidence.playback)

    async def test_buffer_stops_before_response_done_is_also_correlated(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.event("output_audio_buffer.started")
        self.event("output_audio_buffer.stopped")
        self.event("response.done", response={"id": "notification_response", "status": "completed",
            "output": [{"type": "message"}]})
        self.assertEqual(self.bridge.sink.on_event.call_args.args[0].state, Evidence.OUTPUT_FINISHED)

    async def test_speech_commit_gap_and_output_buffer_are_not_idle(self):
        self.setup_offer()
        self.event("input_audio_buffer.speech_started")
        self.session._user_audio_pending = False
        self.assertFalse(self.bridge._idle())
        self.bridge.observe({"type": "response.created", "response": {"id": "ordinary"}})
        self.event("output_audio_buffer.started", response_id="ordinary")
        self.assertFalse(self.bridge._idle())
        self.event("output_audio_buffer.stopped", response_id="ordinary")
        self.assertTrue(self.bridge._idle())
        self.delivery.request_stop.assert_called_once_with("user_interrupted")

    async def test_unsubmitted_speech_fence_does_not_cancel_anything(self):
        self.setup_offer()
        command = InterruptSpeechCommand(route=self.bridge.request.route, command_id=uuid4(),
            requested_at=self.session._now(), reason="user_interrupted")
        receipt = await self.bridge.fence_notification_output(command, delivery=self.delivery)
        self.assertTrue(receipt.output_fenced)
        self.assertEqual(self.provider.sent, [])

    async def test_old_result_fence_never_clears_new_ordinary_answer(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.session._current_response_id = "new_ordinary"
        self.event("output_audio_buffer.started", response_id="new_ordinary")
        original = self.provider.send_event
        async def send(event, **kwargs):
            result = await original(event, **kwargs)
            self.event("output_audio_buffer.cleared")
            return result
        self.provider.send_event = send
        command = InterruptSpeechCommand(route=self.bridge.request.route, command_id=uuid4(),
            requested_at=self.session._now(), reason="user_interrupted")
        receipt = await self.bridge.fence_notification_output(command, delivery=self.delivery)
        self.assertTrue(receipt.output_fenced)
        self.assertEqual(len(self.provider.sent), 2)

    async def test_owned_cancel_and_clear_require_actual_provider_stop(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.session._current_response_id = "notification_response"
        original = self.provider.send_event
        async def send(event, **kwargs):
            written = await original(event, **kwargs)
            if self.provider.sent[-1]["type"] == "output_audio_buffer.clear":
                self.event("output_audio_buffer.cleared")
            return written
        self.provider.send_event = send
        command = InterruptSpeechCommand(route=self.bridge.request.route, command_id=uuid4(),
            requested_at=self.session._now(), reason="user_interrupted")
        receipt = await self.bridge.fence_notification_output(command, delivery=self.delivery)
        self.assertTrue(receipt.output_fenced)
        self.assertEqual(self.provider.sent[-2], {"type": "response.cancel", "response_id": "notification_response"})
        self.assertEqual(self.provider.sent[-1], {"type": "output_audio_buffer.clear"})


if __name__ == "__main__":
    unittest.main()
