"""Playback deadlines use validated PCM duration, not generation wall time."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.tts import PreparedSpeech, SpeechFrame
from app.attention.voice import ExpertVoiceDelivery


class Speech:
    @asynccontextmanager
    async def open_stream(self, request, *, authorize_submission, is_current):
        assert await authorize_submission(request)
        assert await is_current()

        async def frames():
            yield SpeechFrame(b"\x00\x00" * 12000, 0)

        yield SimpleNamespace(frames=frames, samples_yielded=12000,
                              first_bytes_ms=1.0, provider_request_id="fixture")


class AttentionVoiceTimingTests(IsolatedAsyncioTestCase):
    def delivery(self, *, receipt_seconds=0.01):
        self.repository = SimpleNamespace(
            lease_seconds=3600,
            prepare=AsyncMock(return_value=(PreparedSpeech("First.", "fixture", "cedar"),
                                            PreparedSpeech("Second.", "fixture", "cedar"))),
            submit_chunk=AsyncMock(return_value=True), chunk_generated=AsyncMock(),
            generation_finished=AsyncMock(), playback_receipt=AsyncMock(return_value=False),
            renew=AsyncMock(), settle=AsyncMock())
        self.session = SimpleNamespace(
            _closed=asyncio.Event(), _context_capture=None,
            remember_attention_item=Mock(), record_attention_speech=AsyncMock(),
            release_attention=AsyncMock())
        delivery = ExpertVoiceDelivery(
            self.repository, Speech(),
            SimpleNamespace(attempt_id=uuid4(), source_namespace="task_result"),
            self.session, receipt_seconds=receipt_seconds)
        self.session._attention_delivery = delivery
        delivery._send = AsyncMock()
        delivery._clear = AsyncMock()
        return delivery

    def event(self, delivery, index, samples, *, event_type="playback.progress"):
        return dict(type=event_type, item_id=delivery.item_id(index),
                    response_id=delivery.response_id, played_samples=samples, content_index=0)

    async def test_each_chunk_has_its_own_generation_deadline_and_pcm_tail_budget(self):
        delivery = self.delivery(receipt_seconds=2)
        budgets = []
        real_timeout = asyncio.timeout

        def timeout(seconds):
            budgets.append(seconds)
            return real_timeout(seconds)

        async def sent(index, event_type, **kwargs):
            if index == 1 and event_type == "audio.output.done":
                # Finish next tick, after run computes the queued-tail budget.
                asyncio.get_running_loop().call_soon(delivery._played.set)

        delivery._send.side_effect = sent
        with patch("app.attention.voice.asyncio.timeout", side_effect=timeout):
            await delivery.run()
        self.assertEqual(budgets, [300, 300, 300, 3])
        self.assertEqual(self.repository.chunk_generated.await_count, 2)
        self.repository.settle.assert_not_awaited()

    async def test_validated_partial_receipts_reduce_tail_without_context_capture(self):
        delivery = self.delivery(receipt_seconds=2)
        delivery._items = {0: {"samples": 24000}, 1: {"samples": 48000}}
        await delivery.receipt(self.event(delivery, 0, 24000), completed=False)
        await delivery.receipt(self.event(delivery, 1, 12000), completed=False)
        self.assertEqual(delivery._playback_timeout_seconds(), 3.5)
        self.assertFalse(delivery._played.is_set())

    async def test_invalid_receipt_cannot_shorten_deadline(self):
        delivery = self.delivery(receipt_seconds=2)
        delivery._items = {0: {"samples": 24000}}
        self.repository.playback_receipt.side_effect = AttentionError(
            "attention_playback_invalid", "Fixture invalid receipt")
        with self.assertRaises(AttentionError):
            await delivery.receipt(self.event(delivery, 0, 24000), completed=True)
        self.assertNotIn("played_samples", delivery._items[0])
        self.assertEqual(delivery._playback_timeout_seconds(), 3)
        self.assertFalse(delivery._played.is_set())

    async def test_receipt_before_local_send_callback_cannot_create_negative_budget(self):
        delivery = self.delivery(receipt_seconds=2)
        delivery._items = {0: {"samples": 0}}
        await delivery.receipt(self.event(delivery, 0, 12000), completed=False)
        self.assertEqual(delivery._playback_timeout_seconds(), 2)

    async def test_healthy_playback_may_finish_after_old_fixed_receipt_window(self):
        delivery = self.delivery()
        generated = asyncio.Event()

        async def sent(index, event_type, **kwargs):
            if index == 1 and event_type == "audio.output.done":
                generated.set()

        delivery._send.side_effect = sent
        task = delivery.task = asyncio.create_task(delivery.run())
        self.addAsyncCleanup(self.finish, delivery, task)
        await asyncio.wait_for(generated.wait(), 1)
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())
        await delivery.receipt(self.event(delivery, 0, 12000, event_type="playback.completed"),
                               completed=True)
        self.repository.playback_receipt.return_value = True
        await delivery.receipt(self.event(delivery, 1, 12000, event_type="playback.completed"),
                               completed=True)
        await asyncio.wait_for(task, 1)
        self.assertTrue(delivery._played.is_set())
        delivery._clear.assert_not_awaited()
        self.repository.settle.assert_not_awaited()

    async def test_missing_receipts_still_expire_and_never_claim_playback(self):
        delivery = self.delivery()
        # No chunks: only the finite receipt grace remains.
        self.repository.prepare.return_value = ()
        await asyncio.wait_for(delivery.run(), 1)
        self.assertFalse(delivery._played.is_set())
        self.assertTrue(delivery.revoked)
        delivery._clear.assert_awaited_once()
        self.repository.settle.assert_awaited_once_with(
            delivery.claim, reason="playback_or_generation_timeout", interrupted=False)

    async def test_interruption_during_tail_wait_does_not_wait_for_duration_budget(self):
        delivery = self.delivery(receipt_seconds=60)
        generated = asyncio.Event()

        async def sent(index, event_type, **kwargs):
            if index == 1 and event_type == "audio.output.done":
                generated.set()

        delivery._send.side_effect = sent
        task = delivery.task = asyncio.create_task(delivery.run())
        self.addAsyncCleanup(self.finish, delivery, task)
        await asyncio.wait_for(generated.wait(), 1)
        await delivery.receipt(self.event(delivery, 0, 120, event_type="playback.interrupted"),
                               completed=False)
        await asyncio.wait_for(task, 1)
        self.assertEqual(delivery.reason, "user_interrupted")
        self.assertFalse(delivery._played.is_set())
        self.repository.settle.assert_awaited_once_with(
            delivery.claim, reason="user_interrupted", interrupted=True)

    async def finish(self, delivery, task):
        if not task.done():
            await delivery.stop("fixture_cleanup")
        await asyncio.gather(task, return_exceptions=True)
