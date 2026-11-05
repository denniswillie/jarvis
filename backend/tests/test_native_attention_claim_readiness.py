"""Pre-submission readiness races through real native delivery/ledger code.

Canonical storage and transport use the maintained offline fixture. These are
not PostgreSQL, provider, timing-performance, hearing or model acceptance tests.
"""
import asyncio
from datetime import timedelta
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

import test_native_attention_delivery as delivery_helpers
from app.attention.models import AttentionError
from test_notification_ledger import NOW


class NativeClaimReadinessTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = delivery_helpers.NativeDeliveryTests.asyncSetUp
    asyncTearDown = delivery_helpers.NativeDeliveryTests.asyncTearDown

    async def begin_wait(self, *, prepared=False):
        self.waiting = asyncio.Event()
        self.diagnostics = []
        delivery = self.sink.create_delivery(self.f, None, self.f.claim, self.session, receipt_seconds=1)
        self.session._attention_delivery = delivery
        diagnostic = delivery._readiness_diagnostic

        def observe(stage, reason):
            diagnostic(stage, reason)
            self.diagnostics.append((stage, reason))
            if reason == "route_not_ready":
                self.waiting.set()

        delivery._readiness_diagnostic = observe
        if prepared:
            prepare = self.sink.ledger.prepare

            async def pause_after_prepare(*args, **kwargs):
                offer = await prepare(*args, **kwargs)
                self.runtime.ready = False
                return offer

            self.sink.ledger.prepare = pause_after_prepare
        else:
            self.runtime.ready = False
        delivery.task = asyncio.create_task(delivery.run())
        await asyncio.wait_for(self.waiting.wait(), 1)
        self.assertFalse(delivery.task.done())
        self.assertIsNone(delivery.ticket)
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()
        self.runtime.coordinator.interrupt_speech.assert_not_awaited()
        return delivery

    async def finish(self, delivery):
        await asyncio.wait_for(delivery.task, 1)
        await asyncio.wait_for(self.sink._events.join(), 1)

    async def test_claimed_unready_waits_then_submits_exactly_once_without_new_deadline(self):
        self.clock = NOW
        self.sink.clock = lambda: self.clock
        self.f.intent["expires_at"] = NOW + timedelta(minutes=10)
        delivery = await self.begin_wait()
        original_claim = delivery.claim
        self.assertIsNone(delivery.offer)
        self.f.settle.assert_not_awaited()
        self.clock += timedelta(seconds=2)
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertIs(delivery.claim, original_claim)
        self.assertEqual(delivery.offer.expires_at, NOW + timedelta(seconds=300))
        self.assertEqual(self.f.operations.count("prepare"), 1)
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.runtime.offer_notification.assert_awaited_once()
        self.assertEqual(self.writes, ["offer", "response"])
        self.assertTrue(delivery.output_finished)
        self.assertIsNone(self.f.attempt["played_samples"])
        self.assertEqual(self.diagnostics, [("before_prepare", "route_not_ready"),
                                           ("before_prepare", "route_ready")])

    async def test_readiness_loss_during_prepare_waits_before_submission(self):
        delivery = await self.begin_wait(prepared=True)
        original_offer = delivery.offer
        self.assertIsNotNone(original_offer)
        self.assertIsNone(self.f.attempt["submitted_at"])
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertIs(delivery.offer, original_offer)
        self.assertEqual(self.f.operations.count("prepare"), 1)
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.runtime.offer_notification.assert_awaited_once()
        self.assertEqual(self.diagnostics[0], ("before_submission", "route_not_ready"))

    async def test_source_withdrawal_during_wait_still_rejects_before_submit(self):
        delivery = await self.begin_wait(prepared=True)
        self.f.source_problem = "withdrawn"
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_source_unavailable")
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()
        self.assertEqual(self.writes, [])

    async def test_generation_change_during_wait_cannot_mutate_new_intent(self):
        delivery = await self.begin_wait(prepared=True)
        self.f.intent.update(generation=self.f.claim.generation + 1, state="pending")
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_claim_lost")
        self.assertEqual(self.f.intent["state"], "pending")
        self.assertEqual(self.f.intent["generation"], self.f.claim.generation + 1)
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()

    async def test_lost_lease_during_wait_is_not_local_readiness_recovery(self):
        delivery = await self.begin_wait(prepared=True)
        self.f.lease_valid = False
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_claim_lost")
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()

    async def test_source_expiry_is_the_original_prepared_deadline(self):
        self.clock = NOW
        self.sink.clock = lambda: self.clock
        self.f.intent["expires_at"] = NOW + timedelta(seconds=1)
        delivery = await self.begin_wait(prepared=True)
        self.clock += timedelta(seconds=1)
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_offer_expired")
        self.assertEqual(delivery.offer.expires_at, NOW + timedelta(seconds=1))
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()
        self.assertIn(("before_submission", "original_deadline_expired"), self.diagnostics)

    async def test_original_window_expiry_before_prepare_never_prepares_or_submits(self):
        self.clock = NOW
        self.sink.clock = lambda: self.clock
        delivery = await self.begin_wait()
        self.clock += timedelta(seconds=300)
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_offer_expired")
        self.assertIsNone(delivery.offer)
        self.assertEqual(self.f.operations.count("prepare"), 0)
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()

    async def test_route_replacement_is_not_readiness_recovery(self):
        delivery = await self.begin_wait()
        self.runtime.request.route = self.runtime.request.route.model_copy(update={"route_generation": 2})
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_not_current")
        self.assertIn(("before_prepare", "route_changed"), self.diagnostics)
        self.runtime.offer_notification.assert_not_awaited()

    async def test_audio_connection_replacement_is_not_readiness_recovery(self):
        delivery = await self.begin_wait()
        self.session.context.audio_session_id = str(uuid4())
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_not_current")
        self.assertIn(("before_prepare", "audio_session_changed"), self.diagnostics)
        self.runtime.offer_notification.assert_not_awaited()

    async def test_new_local_owner_cannot_inherit_waiting_claim_or_old_clear(self):
        delivery = await self.begin_wait()
        newer = object()
        self.session._attention_delivery = newer
        self.runtime.ready = True
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_not_current")
        self.assertIs(self.session._attention_delivery, newer)
        self.runtime.coordinator.interrupt_speech.assert_not_awaited()
        self.assertIn(("before_prepare", "delivery_replaced"), self.diagnostics)

    async def test_heartbeat_keeps_canonical_renewal_while_unready(self):
        self.f.lease_seconds = 0.03
        renewed = asyncio.Event()

        async def renew(claim):
            async with self.f.guard(claim):
                renewed.set()

        self.f.renew = AsyncMock(side_effect=renew)
        delivery = await self.begin_wait()
        await asyncio.wait_for(renewed.wait(), 1)
        self.assertFalse(delivery.task.done())
        self.f.renew.assert_awaited()
        self.f.source_problem = "withdrawn"
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "native_source_revoked")
        self.runtime.offer_notification.assert_not_awaited()

    async def test_sink_shutdown_joins_claim_waiting_before_offer_registration(self):
        delivery = await self.begin_wait()
        self.assertEqual(self.sink._deliveries, {})
        await asyncio.wait_for(self.sink.close(), 1)
        self.assertTrue(delivery.task.done())
        self.assertTrue(self.sink._worker.done())
        self.assertIsNone(self.session._attention_delivery)
        self.assertEqual(delivery.reason, "native_attention_closed")
        self.runtime.offer_notification.assert_not_awaited()

    async def test_explicit_stop_during_wait_is_not_silently_retried(self):
        delivery = await self.begin_wait()
        await delivery.stop("user_interrupted")
        await self.finish(delivery)
        self.runtime.ready = True
        await asyncio.sleep(0)
        self.assertEqual(delivery.reason, "user_interrupted")
        self.assertTrue(delivery.revoked)
        self.assertEqual(self.f.operations.count("submit"), 0)
        self.runtime.offer_notification.assert_not_awaited()
        self.runtime.coordinator.interrupt_speech.assert_awaited_once()

    async def test_readiness_loss_after_submission_is_unknown_and_never_retried(self):
        commit = self.sink.ledger.commit_submission

        async def lose_readiness_after_commit(*args):
            ticket = await commit(*args)
            self.runtime.ready = False
            return ticket

        self.sink.ledger.commit_submission = lose_readiness_after_commit
        delivery = self.sink.create_delivery(self.f, None, self.f.claim, self.session, receipt_seconds=1)
        self.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_route_not_ready")
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertIsNotNone(self.f.attempt["submitted_at"])
        self.assertEqual(self.writes, [])
        self.runtime.ready = True
        await asyncio.sleep(0)
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.runtime.offer_notification.assert_awaited_once()

    async def test_readiness_diagnostics_are_bounded_identity_only(self):
        with self.assertLogs("app.attention.native_voice", level="INFO") as logs:
            delivery = await self.begin_wait()
            self.session._closed.set()
            await self.finish(delivery)
        self.assertEqual(delivery.reason, "attention_not_current")
        self.assertEqual(len(logs.output), 2)
        self.assertIn("stage=before_prepare reason=route_not_ready", logs.output[0])
        self.assertIn("reason=session_closed", logs.output[1])
        self.assertIn(str(self.f.claim.attempt_id), logs.output[0])
        self.assertIn(str(self.f.claim.intent_id), logs.output[0])
        for private in (self.f.claim.owner_id, "umbrella", "original"):
            self.assertNotIn(private, repr(logs.output))
        with self.assertLogs("app.attention.native_voice", level="INFO") as bounded:
            for _ in range(20):
                delivery._readiness_diagnostic("before_prepare", "route_not_ready")
        self.assertEqual(len(bounded.output), 6)
