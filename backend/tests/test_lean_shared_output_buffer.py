"""Captured shared-buffer sequences, not proof of client playback or hearing."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from app.voice_routes.contracts import DeliveryEvidenceState as Evidence
import test_lean_native_attention as fixtures


class SharedOutputBufferTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.LeanNativeAttentionTests.asyncSetUp
    asyncTearDown = fixtures.LeanNativeAttentionTests.asyncTearDown
    setup_offer = fixtures.LeanNativeAttentionTests.setup_offer
    event = fixtures.LeanNativeAttentionTests.event
    created = fixtures.LeanNativeAttentionTests.created

    def ordinary(self, response_id):
        self.bridge.observe({"type": "response.created", "response": {"id": response_id}})

    async def test_captured_continuation_drains_without_its_own_start(self):
        self.setup_offer()
        self.session._attention_delivery = None
        self.session._realtime_context_ready.set()
        self.ordinary("first")
        self.event("output_audio_buffer.started", "first")
        self.event("output_audio_buffer.stopped", "first")
        self.ordinary("tool_continuation")
        self.event("output_audio_buffer.started", "tool_continuation")
        self.ordinary("final_continuation")
        self.event("response.done", "final_continuation", response={
            "id": "final_continuation", "status": "completed", "output": [{"type": "message"}],
        })
        self.assertFalse(self.bridge._idle())
        self.event("output_audio_buffer.stopped", "final_continuation")
        self.assertTrue(self.bridge._idle())
        repository = SimpleNamespace(claim_next=AsyncMock(return_value=None))
        await self.bridge._admit(repository, None, worker_id="fixture", receipt_seconds=1)
        repository.claim_next.assert_awaited_once()

    async def test_delayed_old_and_unknown_drains_preserve_newer_output(self):
        self.setup_offer()
        for response_id in ("old", "new"):
            self.ordinary(response_id)
            self.event("output_audio_buffer.started", response_id)
        self.event("output_audio_buffer.stopped", "unknown")
        self.assertEqual(set(self.bridge._playing), {"old", "new"})
        self.event("output_audio_buffer.stopped", "old")
        self.assertEqual(set(self.bridge._playing), {"new"})
        self.event("output_audio_buffer.stopped", "old")
        self.event("output_audio_buffer.cleared", "prior_connection_response")
        self.assertEqual(set(self.bridge._playing), {"new"})

    async def test_duplicate_creation_and_start_do_not_reopen_drained_output(self):
        self.setup_offer()
        self.ordinary("old")
        self.event("output_audio_buffer.started", "old")
        self.event("output_audio_buffer.stopped", "old")
        self.ordinary("new")
        self.event("output_audio_buffer.started", "new")
        self.ordinary("old")
        self.event("output_audio_buffer.started", "old")
        self.event("output_audio_buffer.stopped", "old")
        self.assertEqual(set(self.bridge._playing), {"new"})

    async def test_drain_does_not_override_user_or_tool_admission(self):
        self.setup_offer()
        self.ordinary("ordinary")
        self.event("output_audio_buffer.started", "ordinary")
        self.session._user_audio_pending = True
        self.event("output_audio_buffer.stopped", "ordinary")
        self.assertFalse(self.bridge._idle())
        self.session._user_audio_pending = False
        self.session._lean_responses["pending_tool"] = SimpleNamespace(pending=True)
        self.assertFalse(self.bridge._idle())
        self.session._lean_responses.clear()
        self.assertTrue(self.bridge._idle())

    async def test_successor_drain_is_not_owned_notification_completion(self):
        self.setup_offer()
        await self.bridge.offer_notification(self.offer)
        self.created()
        self.event("output_audio_buffer.started")
        self.event("response.done", response={"id": "notification_response", "status": "completed",
                                             "output": [{"type": "message"}]})
        self.ordinary("ordinary_successor")
        self.event("output_audio_buffer.stopped", "ordinary_successor")
        self.assertFalse(self.bridge._playing)
        self.assertFalse(self.bridge._offers[self.offer.offer_id].stopped.is_set())
        self.assertNotIn(Evidence.OUTPUT_FINISHED,
                         [call.args[0].state for call in self.bridge.sink.on_event.call_args_list])

    async def test_creation_history_is_bounded_without_evicting_playback(self):
        self.setup_offer()
        self.ordinary("still_playing")
        self.event("output_audio_buffer.started", "still_playing")
        for index in range(300):
            self.ordinary(f"response_{index}")
        self.assertEqual(len(self.bridge._response_order), 256)
        self.assertIn("still_playing", self.bridge._response_order)
        self.event("output_audio_buffer.stopped", "response_0")
        self.assertEqual(set(self.bridge._playing), {"still_playing"})
