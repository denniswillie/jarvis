"""Actual Live composer, ledger and adapter with scoped store/network boundaries.

The canonical fixture replaces PostgreSQL owner operations, not the ledger,
session policy, provider adapter, source watcher or native channel handshake.
Peer receipts are controlled reports, never hardware or human-hearing proof.
"""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.attention.notifications import NotificationOutput, VisualNotificationContent
from app.live_state.models import SessionScope
from app.runtime.feature_config import SecondBrainSettings
import test_live_session as composer_fixture
import test_notification_ledger as ledger_fixture


class SourceConnection(ledger_fixture.Connection):
    def is_in_transaction(self):
        return self.store.lock.locked()

    async def fetchval(self, sql, *args):
        if "lock_jarvis_copy_sources" in sql:
            return bool(args[1]) and all(self.store.pool.available(args[0], root) for root in args[1])
        return await super().fetchval(sql, *args)


class LiveNotificationCompositionTests(unittest.IsolatedAsyncioTestCase):
    append = composer_fixture.LiveSessionTests.append
    live = composer_fixture.LiveSessionTests.live
    until = composer_fixture.LiveSessionTests.until
    make = composer_fixture.LiveSessionTests.make
    start = composer_fixture.LiveSessionTests.start
    asyncTearDown = composer_fixture.LiveSessionTests.asyncTearDown

    async def asyncSetUp(self):
        await composer_fixture.LiveSessionTests.asyncSetUp(self)
        self.flags = SecondBrainSettings(_env_file=None, enabled="true", attention_enabled="true",
                                        attention_voice_enabled="true")
        self.patches.extend([patch("app.realtime.live_session.get_second_brain_settings", side_effect=lambda: self.flags),
                             patch.object(ledger_fixture, "NOW", datetime.now(UTC))])
        for item in self.patches[-2:]:
            item.start()
        self.f = ledger_fixture.CanonicalFixture()
        self.f.scope = SessionScope(owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id)
        self.f.claim = replace(self.f.claim, owner_id=self.context.owner_id,
                               session_id=self.context.jarvis_session_id)
        self.f.intent.update(owner_id=self.context.owner_id, preferred_session_id=self.context.jarvis_session_id,
                             state="pending")
        self.f.attempt.update(owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id)
        self.f.connection = SourceConnection(self.f)
        self.f.pool, self.f.lease_seconds = self.db, 60
        self.f.renew = AsyncMock()
        self.f.claim_next = AsyncMock(side_effect=self.claim)
        self.f.settle = AsyncMock(side_effect=self.settle_unsubmitted)
        original_available = self.db.available

        def available(owner, identity, seen=None):
            if identity not in self.f.payloads:
                return original_available(owner, identity, seen)
            row = self.f.payloads[identity]
            return (owner == self.context.owner_id and row["availability"] == "available"
                    and row["content"] is not None
                    and all(available(owner, root, seen) for root in self.f.dependencies.get(identity, ())))

        self.db.available = available
        self.attention_status = "available"
        self.services.attention_service = SimpleNamespace(repository=self.f, _closed=False,
            voice_status=lambda: {"state": self.attention_status})
        self.manager.attention_notifications_enabled = True
        self.manager.attention_repository = self.f
        self.provider_readers = self.maximum_provider_readers = 0
        original_recv = self.wire.socket.recv

        async def recv():
            self.provider_readers += 1
            self.maximum_provider_readers = max(self.maximum_provider_readers, self.provider_readers)
            try:
                return await original_recv()
            finally:
                self.provider_readers -= 1

        self.wire.socket.recv = recv

    async def claim(self, owner, session_id, worker_id, *, admission):
        self.assertEqual((owner, session_id), (self.context.owner_id, self.context.jarvis_session_id))
        if self.f.intent["state"] != "pending" or admission(self.f.intent, self.f.output):
            return None
        self.f.intent["state"] = "claimed"
        self.f.claim = replace(self.f.claim, worker_id=worker_id)
        self.f.attempt["worker_id"] = worker_id
        return self.f.claim

    async def settle_unsubmitted(self, claim, *, reason, interrupted=False):
        self.assertEqual(claim, self.f.claim)
        self.assertIsNone(self.f.attempt["submitted_at"])
        self.f.attempt.update(phase="interrupted" if interrupted else "failed", reason=reason)
        self.f.intent.update(state="pending", reason=reason)

    async def started(self):
        session = await self.start()
        await self.until(lambda: session._startup.done())
        return session

    async def deliver(self):
        return await self.manager.deliver_attention(owner_id=self.context.owner_id, repository=self.f,
            speech=None, worker_id="actual-live-composer-test", receipt_seconds=1)

    def commentary(self):
        return [item for item in self.wire.socket.sent if item["type"] == "session.commentary.append"]

    def visual(self, **changes):
        now = datetime.now(UTC)
        content = VisualNotificationContent(source_namespace="visual_alert", source_id=self.f.intent["source_id"],
            session_id=self.context.jarvis_session_id, observation_payload_id=uuid4(),
            message="The observed control state is uncertain.", supporting_frame_ids=(uuid4(),),
            source_captured_at=now - timedelta(seconds=1), source_received_at=now,
            expires_at=now + timedelta(seconds=30))
        self.f.output = NotificationOutput(content.model_copy(update=changes))
        self.f.claim = replace(self.f.claim, source_namespace="visual_alert")
        self.f.intent["source_namespace"] = "visual_alert"

    async def user_fragment(self):
        # A real provider fragment, not a fake finalized input or client event.
        await self.wire.socket.incoming.put(json.dumps({"type":"session.input_transcript.delta",
            "event_id":"observed-user-fragment", "delta":"Wait a moment",
            "start_ms":10, "end_ms":20}))
        await self.until(lambda: self.session._last_observed_user_at is not None)

    async def test_capability_appears_only_after_actual_ready_and_initialized_service(self):
        session = await self.make()
        self.assertIsNone(session.try_attention)
        self.assertIsNone(session._attention_sink)
        self.assertFalse(self.wire.requests)
        await self.started()
        self.assertTrue(await self.manager.has_connected_voice(self.context.owner_id))
        self.assertEqual(session.availability()["notification_voice"], "available_hearing_unknown")
        self.assertIs(session._attention_sink.repository, self.f)
        await session.close("test_close")
        self.assertIsNone(session.try_attention)
        self.assertFalse(await self.manager.has_connected_voice(self.context.owner_id))

    async def test_flag_alone_or_nonrunning_service_does_not_claim_capability(self):
        self.attention_status = "unavailable"
        session = await self.started()
        self.assertIsNone(session._attention_sink)
        self.assertIsNone(session.try_attention)
        self.assertIsNone(await self.deliver())
        self.f.claim_next.assert_not_awaited()

    async def test_live_loss_of_configuration_preserves_manager_fallback(self):
        session = await self.started()
        self.flags = self.flags.model_copy(update={"attention_voice_enabled":"false"})
        self.assertIsNone(session.try_attention)
        self.assertFalse(await self.manager.has_connected_voice(self.context.owner_id))
        self.flags = self.flags.model_copy(update={"attention_voice_enabled":"true"})
        self.services.attention_service._closed = True
        self.assertIsNone(await self.deliver())
        self.f.claim_next.assert_not_awaited()

    async def test_notification_reaches_real_commentary_and_ack_is_unknown_hearing(self):
        session = await self.started()
        delivery = await self.deliver()
        self.assertIsNotNone(delivery)
        await asyncio.wait_for(delivery.task, 2)
        self.assertTrue(delivery.provider_received, (session.failure, self.f.attempt))
        self.assertEqual(len(self.commentary()), 1)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "live_commentary_received_hearing_unknown")
        self.assertIsNone(self.f.attempt["played_samples"])
        self.assertEqual(self.f.attempt["forwarded_samples"], 0)
        self.assertTrue(all(item["output_id"] is None for item in self.f.evidence.values()))
        self.assertIn(self.f.claim.source_payload_id, session._source_context._sources)
        self.assertFalse(any(item["type"] == "voice_route.interrupt" for item in self.phone.messages))
        self.assertTrue(session.ready)
        self.assertEqual((self.maximum_provider_readers, self.phone.maximum_readers), (1, 1))
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.assertIsNone(await self.deliver())  # Unknown hearing is not replay permission.

    async def test_unsubmitted_source_withdrawal_blocks_actual_provider_write(self):
        session = await self.started()
        original = session._attention_sink.submit_offer

        async def withdrawn(offer):
            self.f.source_problem = "withdrawn"
            return await original(offer)

        session._attention_sink.submit_offer = withdrawn
        delivery = await self.deliver()
        await asyncio.wait_for(delivery.task, 2)
        self.assertFalse(delivery.provider_received)
        self.assertFalse(self.commentary())
        self.assertEqual(self.f.attempt["phase"], "unknown")

    async def test_withdrawal_after_ack_fences_real_route_and_joins_shutdown(self):
        session = await self.started()
        delivery = await self.deliver()
        await asyncio.wait_for(delivery.task, 2)
        self.assertTrue(delivery.provider_received)
        self.f.payloads[self.f.claim.source_payload_id].update(availability="deleted", content=None)
        await self.until(lambda: session._closed.is_set())
        await asyncio.wait_for(session.close("withdrawn"), 2)
        self.assertTrue(session.shutdown_confirmed, session.failure)
        self.assertIsNone(session.try_attention)
        self.assertEqual(len(self.commentary()), 1)

    async def test_oversized_notification_remains_unclaimed_without_truncation(self):
        await self.started()
        text = "This is not confirmed. " * 100
        self.f.output = NotificationOutput(self.f.output.content.model_copy(update={"message":text}))
        self.assertIsNone(await self.deliver())
        self.assertEqual(self.f.intent["state"], "pending")
        self.assertEqual(self.f.output.content.message, text)
        self.assertFalse(self.f.operations or self.commentary())

    async def test_generic_visual_policy_preserves_source_session_expiry_and_time(self):
        session = await self.started()
        now = datetime.now(UTC)
        for changes, reason in (
            ({"session_id":uuid4()}, "source_session_mismatch"),
            ({"source_received_at":None}, "source_time_unavailable"),
            ({"source_received_at":now + timedelta(minutes=1)}, "source_time_unavailable"),
            ({"expires_at":now - timedelta(seconds=1)}, "superseded_observation"),
        ):
            with self.subTest(reason=reason):
                self.visual(**changes)
                self.assertIsNone(await self.deliver())
                self.assertEqual(session.notification_omission, reason)
        self.assertFalse(self.f.operations or self.commentary())

    async def test_provisional_activity_defers_visuals_without_finality_or_interruption(self):
        session = await self.started()
        await self.user_fragment()
        self.visual()
        self.assertIsNone(await self.deliver())
        self.assertEqual(session.notification_omission, "live_observed_speech_recent")
        self.assertFalse(self.commentary())
        self.assertTrue(session.ready)
        self.assertFalse(any(item["type"] == "voice_route.interrupt" for item in self.phone.messages))
        await self.until(lambda: any(item.source_event_id.startswith("live-caption:") for item in self.captured))
        captions = [item for item in self.captured if item.source_event_id.startswith("live-caption:")]
        self.assertTrue(captions)
        self.assertTrue(all(item.finality == "provisional" for item in captions))

    async def test_proactive_chain_is_bounded_and_observed_user_speech_resets_only_gate(self):
        session = await self.started()
        while session._proactive_interrupt_gate.try_begin():
            pass
        self.assertIsNone(await self.deliver())
        self.assertEqual(session.notification_omission, "proactive_chain_limit")
        await self.user_fragment()
        self.assertTrue(session._proactive_interrupt_gate.can_begin())
        delivery = await self.deliver()
        await asyncio.wait_for(delivery.task, 2)
        self.assertTrue(delivery.provider_received)
        self.assertTrue(session.ready)

    async def test_close_before_delivery_task_enters_still_releases_claim(self):
        session = await self.started()
        delivery = await self.deliver()
        self.assertFalse(delivery._started)
        # The sink's pre-start stop latch runs synchronously in this call stack.
        await delivery.stop("source_withdrawn")
        await session.close("test_close")
        self.assertTrue(delivery.task.done())
        self.assertIsNone(session._attention_delivery)
        self.assertIsNone(session._attention_sink._delivery)
        self.f.settle.assert_awaited_once()
        self.assertFalse(self.commentary())
        self.assertTrue(session.shutdown_confirmed)

    async def test_immediate_composer_close_without_explicit_stop_releases_pre_entry_owner(self):
        session = await self.started()
        delivery = await self.deliver()
        self.assertFalse(delivery._started)
        await asyncio.wait_for(session.close("immediate_close"), 2)
        self.assertTrue(delivery.task.done())
        self.assertFalse(delivery.task.cancelled())
        self.assertIsNone(session._attention_delivery)
        self.assertIsNone(session._attention_sink._delivery)
        self.f.settle.assert_awaited_once()
        self.assertIsNone(self.f.attempt["submitted_at"])
        self.assertFalse(self.commentary())
        self.assertTrue(session.shutdown_confirmed)

    async def test_composer_shutdown_and_repeated_stop_join_inflight_settlement(self):
        session = await self.started()
        self.wire.socket.auto_ack = False
        entered, release = asyncio.Event(), asyncio.Event()
        sink = session._attention_sink
        original = sink.ledger.settle

        async def settling(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        sink.ledger.settle = AsyncMock(side_effect=settling)
        delivery = await self.deliver()
        await self.until(lambda: bool(self.commentary()))
        closing = asyncio.create_task(session.close("shutdown_during_ack_wait"))
        repeated_close = None
        try:
            await asyncio.wait_for(entered.wait(), 2)
            self.assertTrue(delivery._settling)
            first_reason = delivery.reason
            await delivery.stop("source_withdrawn")
            repeated_close = asyncio.create_task(sink.close())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            self.assertFalse(repeated_close.done())
            self.assertIs(session._attention_delivery, delivery)
            self.assertEqual(delivery.reason, first_reason)
        finally:
            release.set()
            await asyncio.wait_for(closing, 2)
            if repeated_close is not None:
                await asyncio.wait_for(repeated_close, 2)
        sink.ledger.settle.assert_awaited_once()
        self.assertTrue(delivery.task.done())
        self.assertFalse(delivery.task.cancelled())
        self.assertFalse(delivery.provider_received)
        self.assertIsNone(session._attention_delivery)
        self.assertIsNone(sink._delivery)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.assertEqual(len(self.commentary()), 1)
        self.assertIsNone(self.f.attempt["played_samples"])
        self.assertEqual(self.f.attempt["forwarded_samples"], 0)
        self.assertTrue(session.shutdown_confirmed)


if __name__ == "__main__":
    unittest.main()
