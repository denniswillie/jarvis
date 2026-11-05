"""Real canonical ledger plus controlled Live append and source-watch boundaries."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.attention.live_voice import LiveAttentionSink, commentary_fits
from app.attention.models import AttentionError
from app.voice_routes.contracts import CommandReceipt, DeliveryEvidence, OutputIdentity, RouteIdentity
from test_notification_ledger import CanonicalFixture, NOW, capabilities


class LiveAttentionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = CanonicalFixture()
        self.f.lease_seconds = 60
        self.f.renew = AsyncMock()
        self.f.settle = AsyncMock()
        self.route = RouteIdentity(scope=self.f.scope, route_id=uuid4(), route_generation=1)
        self.sources, self.writes = set(), []
        self.acknowledge = True
        self.policy = Mock(return_value=None)
        self.session = SimpleNamespace(ready=True, _closed=asyncio.Event(), _attention_delivery=None,
            try_attention=None, context=SimpleNamespace(owner_id=self.f.scope.owner_id,
                jarvis_session_id=self.f.scope.session_id, audio_session_id=str(uuid4())),
            request=SimpleNamespace(route=self.route),
            adapter=SimpleNamespace(capabilities=capabilities(attribution=False)))
        self.session._mark_closing = Mock(side_effect=lambda reason: self.session._closed.set())
        async def interrupt(command):
            return CommandReceipt(route=self.route, command_id=command.command_id, operation="interrupt_speech",
                observed_at=NOW, status="completed", output_fenced=True)
        self.session.coordinator = SimpleNamespace(interrupt_speech=AsyncMock(side_effect=interrupt))
        @asynccontextmanager
        async def source_guard(*, connection, extra_sources):
            self.assertIs(connection, self.f.connection)
            self.assertTrue(self.f.lock.locked())
            self.assertEqual(extra_sources, (self.f.claim.source_payload_id,))
            yield
        self.session._source_context = SimpleNamespace(context_guard=source_guard, observe=self.sources.update)
        async def claim(owner, session_id, worker, *, admission):
            self.assertEqual((owner, session_id), (self.f.scope.owner_id, self.f.scope.session_id))
            return None if admission(self.f.intent, self.f.output) else self.f.claim
        self.f.claim_next = AsyncMock(side_effect=claim)
        async def submit(offer):
            async with self.sink.guard(offer):
                self.writes.append(offer)
            if self.acknowledge:
                self.sink.on_event(self.evidence(offer, "provider_received"))
            return self.evidence(offer, "submitted")
        self.submit = AsyncMock(side_effect=submit)
        self.sink = LiveAttentionSink(self.session, self.f, clock=lambda: NOW,
                                      admission=self.policy, submit_offer=self.submit)
        self.sink.start()

    async def asyncTearDown(self):
        await self.sink.close()

    def evidence(self, offer, state, **kwargs):
        return DeliveryEvidence(evidence_id=uuid4(), route=offer.route, transport="native_voice",
            observed_at=NOW, state=state, attribution="explicit", offer_id=offer.offer_id,
            attempt=offer.attempt, **kwargs)

    async def deliver(self, *, deadline=1):
        delivery = await self.sink.try_attention(self.f, object(), worker_id="worker", receipt_seconds=deadline)
        self.assertIsNotNone(delivery)
        await asyncio.wait_for(delivery.task, 2)
        return delivery

    async def test_correlated_ack_settles_unknown_without_interrupting_or_tts(self):
        delivery = await self.deliver()
        self.assertTrue(delivery.provider_received)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "live_commentary_received_hearing_unknown")
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.assertEqual(len(self.writes), 1)
        self.assertEqual(self.sources, {self.f.claim.source_payload_id})
        self.assertIsNone(self.session._attention_delivery)
        self.session.coordinator.interrupt_speech.assert_not_awaited()
        self.session._mark_closing.assert_not_called()
        self.assertTrue(all(item["output_id"] is None for item in self.f.evidence.values()))
        self.assertEqual(self.f.attempt["forwarded_samples"], 0)
        self.assertIsNone(self.f.attempt["played_samples"])

    async def test_submission_without_ack_is_not_delivery_and_does_not_retry(self):
        self.acknowledge = False
        delivery = await self.deliver(deadline=0.03)
        self.assertFalse(delivery.provider_received)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "live_notification_receipt_timeout")
        self.submit.assert_awaited_once()
        self.session.coordinator.interrupt_speech.assert_awaited_once()

    async def test_source_withdrawal_after_claim_prevents_append(self):
        async def revoked(offer):
            self.f.source_problem = "withdrawn"
            async with self.sink.guard(offer):
                self.writes.append(offer)
            raise AssertionError("withdrawn source escaped")
        self.submit.side_effect = revoked
        delivery = await self.deliver()
        self.assertFalse(delivery.provider_received)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.sources, set())

    async def test_one_write_ticket_cannot_be_reused(self):
        async def duplicate(offer):
            async with self.sink.guard(offer):
                self.writes.append(offer)
            with self.assertRaises(AttentionError):
                async with self.sink.guard(offer):
                    self.writes.append(offer)
            return self.evidence(offer, "provider_received")
        self.submit.side_effect = duplicate
        await self.deliver()
        self.assertEqual(len(self.writes), 1)

    async def test_duplicate_provider_receipt_cannot_resubmit(self):
        delivery = await self.deliver()
        self.sink.on_event(self.evidence(delivery.offer, "provider_received"))
        self.submit.assert_awaited_once()
        self.assertIsNone(self.sink._delivery)
        self.assertEqual(self.f.operations.count("submit"), 1)

    async def test_route_not_ready_or_foreign_repository_never_claims(self):
        self.session.ready = False
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1))
        self.session.ready = True
        self.assertIsNone(await self.sink.try_attention(object(), worker_id="w", receipt_seconds=1))
        self.f.claim_next.assert_not_awaited()

    async def test_existing_policy_can_defer_without_creating_attempt(self):
        self.policy.return_value = "proactive_chain_limit"
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1))
        self.policy.assert_called_once()
        self.assertEqual(self.f.operations, [])
        self.submit.assert_not_awaited()

    async def test_oversized_exact_text_stays_unclaimed_and_unmodified(self):
        content = self.f.output.content.model_copy(update={"message": "Do not turn it on. " * 100})
        self.f.output = type(self.f.output)(content)
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1))
        self.assertEqual(self.f.operations, [])
        self.policy.assert_not_called()
        self.submit.assert_not_awaited()

    async def test_concurrent_admissions_have_one_claim_and_offer(self):
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.submit.side_effect
        async def blocked(offer):
            entered.set()
            await release.wait()
            return await original(offer)
        self.submit.side_effect = blocked
        first = await self.sink.try_attention(self.f, worker_id="a", receipt_seconds=1)
        await entered.wait()
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="b", receipt_seconds=1))
        release.set()
        await asyncio.wait_for(first.task, 1)
        self.f.claim_next.assert_awaited_once()

    async def test_disconnect_during_claim_settles_without_submission(self):
        original = self.f.claim_next.side_effect
        async def closing(*args, **kwargs):
            result = await original(*args, **kwargs)
            self.session._closed.set()
            return result
        self.f.claim_next.side_effect = closing
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1))
        self.f.settle.assert_awaited_once_with(self.f.claim, reason="live_notification_route_unavailable")
        self.submit.assert_not_awaited()

    async def test_imagined_audio_attribution_fails_closed(self):
        async def fake_output(offer):
            async with self.sink.guard(offer):
                self.writes.append(offer)
            return self.evidence(offer, "output_finished", output=OutputIdentity(output_id=uuid4()))
        self.submit.side_effect = fake_output
        delivery = await self.deliver()
        self.assertFalse(delivery.provider_received)
        self.assertEqual(self.sink.failure, "live_notification_evidence_mismatch")
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertTrue(self.session._closed.is_set())

    async def test_close_joins_pending_delivery_and_disables_voice_availability(self):
        self.acknowledge = False
        delivery = await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1)
        while not self.writes:
            await asyncio.sleep(0)
        await asyncio.wait_for(self.sink.close(), 1)
        self.assertTrue(delivery.task.done())
        self.assertIsNone(self.session.try_attention)
        self.assertIsNone(self.session._attention_delivery)
        self.assertEqual(self.f.attempt["phase"], "unknown")

    async def test_immediate_close_settles_claim_before_delivery_coroutine_starts(self):
        delivery = await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1)
        self.assertFalse(delivery._started)
        await asyncio.wait_for(self.sink.close(), 1)
        self.assertTrue(delivery.task.done())
        self.assertIsNone(self.sink._delivery)
        self.assertIsNone(self.session._attention_delivery)
        self.f.settle.assert_awaited_once_with(self.f.claim,
            reason="live_notification_sink_closed", interrupted=False)
        self.submit.assert_not_awaited()
        self.assertEqual(self.f.operations, [])
        self.session.coordinator.interrupt_speech.assert_not_awaited()

    async def test_pre_start_stop_is_cleanup_not_a_provider_submission(self):
        delivery = await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1)
        await delivery.stop("user_interrupted")
        await asyncio.wait_for(delivery.task, 1)
        self.f.settle.assert_awaited_once_with(self.f.claim, reason="user_interrupted", interrupted=True)
        self.assertIsNone(self.session._attention_delivery)
        self.submit.assert_not_awaited()
        self.session.coordinator.interrupt_speech.assert_not_awaited()

    async def test_repeated_pre_start_stop_preserves_single_settlement(self):
        delivery = await self.sink.try_attention(self.f, worker_id="w", receipt_seconds=1)
        await delivery.stop("source_withdrawn")
        await delivery.stop("source_withdrawn")
        await asyncio.wait_for(delivery.task, 1)
        self.f.settle.assert_awaited_once_with(self.f.claim, reason="source_withdrawn", interrupted=False)
        self.assertIsNone(self.sink._delivery)
        self.submit.assert_not_awaited()

    async def test_repeated_stop_and_close_cannot_cancel_owned_settlement(self):
        self.acknowledge = False
        entered, release = asyncio.Event(), asyncio.Event()
        original = self.sink.ledger.settle

        async def settling(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        self.sink.ledger.settle = AsyncMock(side_effect=settling)
        delivery = await self.sink.try_attention(self.f, worker_id="worker", receipt_seconds=1)
        while not self.writes:
            await asyncio.sleep(0)
        await delivery.stop("source_withdrawn")
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(delivery._settling)
            await delivery.stop("different_late_reason")
            closing = asyncio.create_task(self.sink.close())
            await asyncio.sleep(0)
            self.assertFalse(delivery.task.done())
            self.assertIs(self.session._attention_delivery, delivery)
            self.assertEqual(delivery.reason, "source_withdrawn")
        finally:
            release.set()
            await asyncio.wait_for(delivery.task, 1)
            if closing is not None:
                await asyncio.wait_for(closing, 1)
        self.sink.ledger.settle.assert_awaited_once()
        self.assertFalse(delivery.task.cancelled())
        self.assertIsNone(self.session._attention_delivery)
        self.assertIsNone(self.sink._delivery)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.submit.assert_awaited_once()

    async def test_close_during_ack_cleanup_neither_cancels_settlement_nor_fences_speech(self):
        entered, release = asyncio.Event(), asyncio.Event()
        delivery = await self.sink.try_attention(self.f, worker_id="worker", receipt_seconds=1)

        async def heartbeat():
            try:
                await asyncio.Event().wait()
            finally:
                entered.set()
                await release.wait()

        # Yield in the external submit fixture so the lease task enters first.
        original_submit = self.submit.side_effect

        async def submit(offer):
            await asyncio.sleep(0)
            return await original_submit(offer)

        delivery._heartbeat = heartbeat
        self.submit.side_effect = submit
        closing = None
        try:
            await asyncio.wait_for(entered.wait(), 1)
            self.assertTrue(delivery.provider_received)
            self.assertTrue(delivery._settling)
            closing = asyncio.create_task(self.sink.close())
            await asyncio.sleep(0)
            self.assertFalse(delivery.task.done())
            self.session.coordinator.interrupt_speech.assert_not_awaited()
        finally:
            release.set()
            await asyncio.wait_for(delivery.task, 1)
            if closing is not None:
                await asyncio.wait_for(closing, 1)
        self.assertFalse(delivery.task.cancelled())
        self.assertIsNone(self.session._attention_delivery)
        self.assertIsNone(self.sink._delivery)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "live_commentary_received_hearing_unknown")
        self.session.coordinator.interrupt_speech.assert_not_awaited()

    def test_budget_preserves_whole_negation_and_escaped_unicode(self):
        self.assertTrue(commentary_fits("The stove shutdown is NOT confirmed."))
        self.assertFalse(commentary_fits("\u00e9" * 100))
        self.assertFalse(commentary_fits(""))
        self.assertFalse(commentary_fits("bad\x00text"))


if __name__ == "__main__":
    unittest.main()
