"""Native speech uses existing claims and canonical ledger; no provider/DB I/O."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.native_voice import NativeAttentionSink
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import LiveResponseSources, ResponseSourceSnapshot
from app.voice_routes.contracts import CommandReceipt, DeliveryEvidence, OutputIdentity, RouteIdentity
from test_notification_ledger import CanonicalFixture, NOW, capabilities


class NativeDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = CanonicalFixture()
        self.f.lease_seconds = 60
        self.f.renew = AsyncMock()
        self.f.settle = AsyncMock()
        self.session = SimpleNamespace(_closed=asyncio.Event(), _attention_delivery=None,
            context=SimpleNamespace(audio_session_id=str(uuid4())), try_attention=AsyncMock())
        self.session._mark_closing = lambda reason: self.session._closed.set()
        async def release(delivery):
            if self.session._attention_delivery is delivery:
                self.session._attention_delivery = None
        self.session.release_attention = release
        route = RouteIdentity(scope=self.f.scope, route_id=uuid4(), route_generation=1)
        async def interrupt(command):
            return CommandReceipt(route=route, command_id=command.command_id, operation="interrupt_speech",
                observed_at=NOW, status="completed", output_fenced=True)
        self.runtime = SimpleNamespace(ready=True, request=SimpleNamespace(route=route),
            adapter=SimpleNamespace(capabilities=capabilities()),
            coordinator=SimpleNamespace(interrupt_speech=AsyncMock(side_effect=interrupt)), _fail=Mock())
        self.admission = AsyncMock()
        self.sink = NativeAttentionSink(self.session, self.runtime, self.f, self.admission, clock=lambda: NOW)
        self.sink.start()
        self.writes = []
        self.withdraw = False

        async def offer_notification(offer):
            if self.withdraw:
                self.f.source_problem = "withdrawn"
            async with self.sink.guard(offer) as connection:
                self.assertIs(connection, self.f.connection)
                self.assertTrue(self.f.lock.locked())
                self.writes.append("offer")
            async with self.sink.guard(offer) as connection:
                self.assertIs(connection, self.f.connection)
                self.writes.append("response")
            return DeliveryEvidence(evidence_id=uuid4(), route=offer.route, transport="native_voice",
                observed_at=NOW, state="output_finished", attribution="explicit", offer_id=offer.offer_id,
                attempt=offer.attempt, output=OutputIdentity(output_id=uuid4(), provider_response_ref="response"))
        self.runtime.offer_notification = AsyncMock(side_effect=offer_notification)

    async def asyncTearDown(self):
        await self.sink.close()

    async def run_delivery(self):
        delivery = self.sink.create_delivery(self.f, None, self.f.claim, self.session, receipt_seconds=1)
        self.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        await asyncio.wait_for(delivery.task, 2)
        await asyncio.wait_for(self.sink._events.join(), 1)
        return delivery

    async def test_offer_and_response_share_one_submission_and_settle_unknown_hearing(self):
        delivery = await self.run_delivery()
        self.assertEqual(self.writes, ["offer", "response"])
        self.assertTrue(delivery.ticket._used)
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertTrue(delivery.output_finished)
        self.assertIsNone(self.session._attention_delivery)
        self.runtime.coordinator.interrupt_speech.assert_not_awaited()

    async def test_withdrawal_after_submission_prevents_external_write_and_fences(self):
        self.withdraw = True
        delivery = await self.run_delivery()
        self.assertEqual(self.writes, [])
        self.assertTrue(delivery.ticket._used)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.runtime.coordinator.interrupt_speech.assert_awaited_once()

    async def test_duplicate_evidence_does_not_resubmit_offer(self):
        delivery = await self.run_delivery()
        from app.voice_routes.contracts import DeliveryEvidence
        recorded = next(value for value in self.f.payloads.values()
            if value["content"].get("state") == "output_finished")
        self.sink.on_event(DeliveryEvidence.model_validate(recorded["content"]))
        await asyncio.wait_for(self.sink._events.join(), 1)
        self.runtime.offer_notification.assert_awaited_once()
        self.assertEqual(self.f.operations.count("submit"), 1)

    async def test_late_unavailable_evidence_cannot_interrupt_a_new_speaker(self):
        delivery = await self.run_delivery()
        # The first notification finished; an unrelated utterance now owns audio.
        newer = object()
        self.session._attention_delivery = newer
        self.f.source_problem = "withdrawn"
        offer = delivery.offer
        self.sink.on_event(DeliveryEvidence(evidence_id=uuid4(), route=offer.route,
            transport="native_voice", observed_at=NOW, state="unknown", attribution="explicit",
            offer_id=offer.offer_id, attempt=offer.attempt, reason="late_transport_observation"))
        await asyncio.wait_for(self.sink._events.join(), 1)
        self.runtime.coordinator.interrupt_speech.assert_not_awaited()
        self.assertIs(self.session._attention_delivery, newer)

    async def test_not_ready_does_not_claim(self):
        self.runtime.ready = False
        self.assertIsNone(await self.sink.try_attention(self.f, None, worker_id="fixture", receipt_seconds=1))
        self.admission.assert_not_awaited()

    async def test_foreign_repository_does_not_claim(self):
        self.assertIsNone(await self.sink.try_attention(object(), None, worker_id="fixture", receipt_seconds=1))
        self.admission.assert_not_awaited()

    async def test_no_legacy_speech_provider_is_passed_to_admission(self):
        await self.sink.try_attention(self.f, object(), worker_id="fixture", receipt_seconds=1)
        self.admission.assert_awaited_once_with(self.f, None, worker_id="fixture", receipt_seconds=1)

    async def test_cancel_is_speech_only_and_releases_claim(self):
        entered = asyncio.Event()
        async def blocked(offer):
            entered.set()
            await asyncio.Event().wait()
        self.runtime.offer_notification.side_effect = blocked
        delivery = self.sink.create_delivery(self.f, None, self.f.claim, self.session, receipt_seconds=1)
        self.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        await asyncio.wait_for(entered.wait(), 1)
        await delivery.stop("user_interrupted")
        await asyncio.wait_for(delivery.task, 1)
        self.assertEqual(self.f.attempt["phase"], "interrupted")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertIsNone(self.session._attention_delivery)

    async def test_repeated_stop_keeps_owner_until_shared_fence_finishes(self):
        offering, fencing, release_fence = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def blocked_offer(offer):
            offering.set()
            await asyncio.Event().wait()
        original_interrupt = self.runtime.coordinator.interrupt_speech.side_effect
        async def blocked_fence(command):
            fencing.set()
            await release_fence.wait()
            return await original_interrupt(command)
        self.runtime.offer_notification.side_effect = blocked_offer
        self.runtime.coordinator.interrupt_speech.side_effect = blocked_fence
        delivery = self.sink.create_delivery(self.f, None, self.f.claim, self.session, receipt_seconds=1)
        self.session._attention_delivery = delivery
        delivery.task = asyncio.create_task(delivery.run())
        await asyncio.wait_for(offering.wait(), 1)
        stop = asyncio.create_task(delivery.stop("source_withdrawn"))
        await asyncio.wait_for(fencing.wait(), 1)
        again = asyncio.create_task(delivery.stop("source_withdrawn"))
        await asyncio.sleep(0)
        self.assertIs(self.session._attention_delivery, delivery)
        self.assertFalse(delivery.task.done())
        release_fence.set()
        await asyncio.wait_for(asyncio.gather(stop, again, delivery.task), 1)
        self.runtime.coordinator.interrupt_speech.assert_awaited_once()
        self.assertIsNone(self.session._attention_delivery)


class NotificationAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.offer = SimpleNamespace(offer_id=uuid4(), attempt=SimpleNamespace(attempt_id=uuid4()))
        self.reservation = SimpleNamespace(task_instructions=None)
        self.session = SimpleNamespace(_response_admission_lock=asyncio.Lock(), _closed=asyncio.Event(),
            _realtime_context_ready=asyncio.Event(), _turn_generation=8, _user_audio_pending=False,
            _normal_turn_pending=False, _pending_response=None, _active_responses={}, _native_playback={},
            _attention_delivery=SimpleNamespace(accepts_offer=lambda offer: offer is self.offer),
            _reserve_response_locked=Mock(return_value=self.reservation),
            _send_reserved_response=AsyncMock(return_value=True))
        self.session._realtime_context_ready.set()

    async def test_no_user_turn_is_required_or_manufactured(self):
        self.assertTrue(await RealtimeAudioSession.request_route_notification(self.session, self.offer, turn_generation=8))
        self.assertFalse(self.session._normal_turn_pending)
        self.assertEqual(self.session._reserve_response_locked.call_args.args[0], "prospective")
        self.session._send_reserved_response.assert_awaited_once_with(self.reservation)
        self.assertIn("uncertainty", self.reservation.task_instructions)

    async def test_user_turn_has_priority(self):
        self.session._normal_turn_pending = True
        self.assertFalse(await RealtimeAudioSession.request_route_notification(self.session, self.offer, turn_generation=8))
        self.session._reserve_response_locked.assert_not_called()

    async def test_no_selected_delivery_means_no_notification(self):
        self.session._attention_delivery = None
        self.assertFalse(await RealtimeAudioSession.request_route_notification(self.session, self.offer, turn_generation=8))


class BorrowedTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_snapshot_guard_borrows_same_task_transaction(self):
        root, copy = uuid4(), uuid4()
        sources = LiveResponseSources(None, owner_id="owner", session_id=uuid4(), audio_session_id="audio")
        sources.observe((root,))
        snapshot = ResponseSourceSnapshot(copy, (root,), 1)
        connection = SimpleNamespace(is_in_transaction=lambda: True, fetchval=AsyncMock(return_value=True))
        async with sources.use_transaction(connection):
            async with sources.guard(snapshot):
                self.assertEqual(connection.fetchval.await_args.args[-1], [root, copy])
            async def child():
                return sources._current_connection()
            self.assertIsNone(await asyncio.create_task(child()))
        self.assertIsNone(sources._current_connection())
