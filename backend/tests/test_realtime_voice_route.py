"""Offline control-seam fixtures; no provider, database or phone connection."""

import asyncio
import inspect
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.live_state.models import SessionScope, SourceDependency, StateCursor
from app.voice_routes.contracts import (
    ClientPlaybackObservation, CloseRouteCommand, CommandReceipt, DeliveryAttemptReference,
    DeliveryEvidence, DeliveryEvidenceState, InterruptSpeechCommand, NotificationOffer,
    RouteIdentity, RouteStart, SourceRetirementCommand, Support, VoiceContextUpdate, VoiceTransport,
)
from app.voice_routes.realtime_adapter import RealtimeRouteBindings, RealtimeRouteError, RealtimeVoiceRoute


NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def make_identity(generation=1):
    return RouteIdentity(scope=SessionScope(owner_id="fixture", session_id=UUID(int=1), area_id=None),
                         route_id=UUID(int=100 + generation), route_generation=generation)


def make_context(identity=None, revision=0, **changes):
    values = dict(route=identity or make_identity(), update_id=uuid4(),
        cursor=StateCursor(epoch=UUID(int=50), revision=revision), mode="snapshot",
        text="Explicitly unknown context.", dependencies=(), created_at=NOW,
        expires_at=NOW + timedelta(minutes=5))
    values.update(changes)
    return VoiceContextUpdate(**values)


def make_start(identity=None, **context_changes):
    identity = identity or make_identity()
    return RouteStart(route=identity, command_id=uuid4(), authorization_ref=uuid4(),
        requested_at=NOW, initial_context=make_context(identity, **context_changes))


def make_offer(identity=None):
    return NotificationOffer(route=identity or make_identity(), offer_id=uuid4(),
        attempt=DeliveryAttemptReference(notification_intent_id=uuid4(), attempt_id=uuid4(),
            attempt_generation=4, source=SourceDependency(source_namespace="task_result",
                source_id=uuid4(), revision=3, generation=2), source_payload_id=uuid4()),
        text="The committed result is available.", offered_at=NOW,
        expires_at=NOW + timedelta(minutes=1))


class FakeControl:
    def __init__(self, identity=None):
        self.identity = identity or make_identity()
        self.lock = asyncio.Lock()
        self.writes = []
        self.guarded = []
        self.admissions = []
        self.fences = []
        self.block_id = None
        self.guard_entered = asyncio.Event()
        self.guard_release = asyncio.Event()
        self.valid = True
        self.fence_confirmed = True
        self.wrong_fence = False
        self.cancel_started = asyncio.Event()
        self.closed = False
        self.item_event = None

    async def send_event(self, event, *, write_guard=None):
        async with self.lock:
            if write_guard is None:
                return await self._write(event)
            async with write_guard():
                return await self._write(event)

    async def _write(self, event):
        payload = event() if callable(event) else event
        if inspect.isawaitable(payload):
            payload = await payload
        if payload is None:
            return False
        self.writes.append(payload)
        # Explicit fake-provider ACKs, not inferred socket-write success. Tests
        # of missing/rejected ACKs disable or replace this callback.
        if self.item_event is not None:
            if payload["type"] == "conversation.item.create":
                self.item_event({"type": "conversation.item.done", "item": payload["item"]})
            elif payload["type"] == "conversation.item.delete":
                self.item_event({"type": "conversation.item.deleted", "item_id": payload["item_id"]})
        return True

    @asynccontextmanager
    async def source_guard(self, value):
        self.guarded.append(value)
        if self.block_id is not None and getattr(value, "update_id", None) == self.block_id:
            self.guard_entered.set()
            await self.guard_release.wait()
        if not self.valid:
            raise PermissionError("source withdrawn")
        yield

    async def authorize(self, request):
        return request.route == self.identity

    async def ready(self):
        return True

    async def admit(self, offer, request_id, guard, current):
        self.admissions.append((offer, request_id, guard, current))
        return await self.send_event(lambda: {"type": "response.create", "request_id": request_id}
                                     if current() else None, write_guard=guard)

    async def cancel(self):
        self.cancel_started.set()
        await self.send_event({"type": "response.cancel"})

    async def close(self):
        self.closed = True

    async def fence(self, command):
        self.fences.append(command)
        return CommandReceipt(route=command.route,
            command_id=uuid4() if self.wrong_fence else command.command_id,
            operation="close" if isinstance(command, CloseRouteCommand) else "interrupt_speech",
            observed_at=NOW, status="completed" if self.fence_confirmed else "unknown",
            output_fenced=self.fence_confirmed)

    def bindings(self):
        return RealtimeRouteBindings(identity=self.identity, audio_session_id="audio-fixture", provider=self,
            wait_ready=self.ready, authorize_start=self.authorize, source_guard=self.source_guard,
            admit_response=self.admit, cancel_response=self.cancel, close_control=self.close,
            fence_output=self.fence)


def make_adapter(control=None, *, media_transport="webrtc", playback=False, clock=lambda: NOW):
    control = control or FakeControl()
    route = RealtimeVoiceRoute(control.bindings(), media_transport=media_transport,
        input_codec="opus" if media_transport == "webrtc" else "pcm16",
        output_codec="opus" if media_transport == "webrtc" else "pcm16",
        client_playback_supported=playback, io_timeout=0.15, start_timeout=0.15, clock=clock)
    control.item_event = lambda payload: route.observe_item_event(route.identity, payload)
    return route


async def stop_route(route):
    return await route.close(CloseRouteCommand(route=route.identity, command_id=uuid4(),
                                              requested_at=NOW, reason="fixture_cleanup"))


class RealtimeRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.control = FakeControl()
        self.route = make_adapter(self.control)
        receipt = await self.route.start(make_start())
        self.assertEqual(receipt.status, "completed")

    async def asyncTearDown(self):
        self.control.guard_release.set()
        await stop_route(self.route)

    async def test_native_manifest_and_context_are_not_playback(self):
        self.assertEqual(self.route.capabilities.client_playback_evidence, Support.UNSUPPORTED)
        self.assertEqual(self.route.capabilities.provider_context_receipts, Support.UNSUPPORTED)
        with self.assertRaises(ValueError):
            make_adapter(self.control, playback=True)
        update = make_context(revision=1)
        receipt = await self.route.publish_context(update)
        self.assertEqual(receipt.status, "written")
        self.assertIsNone(receipt.provider_receipt_ref)
        self.assertIs(self.control.guarded[-1], update)
        self.assertFalse(any(event["type"] == "response.create" for event in self.control.writes))
        self.assertTrue(any(event["type"] == "conversation.item.delete" for event in self.control.writes))

    async def test_context_rejects_delta_expired_older_and_wrong_route(self):
        await self.route.publish_context(make_context(revision=2))
        for update in (make_context(revision=3, mode="delta"), make_context(revision=1),
                       make_context(revision=3, created_at=NOW - timedelta(seconds=2),
                                    expires_at=NOW - timedelta(seconds=1))):
            with self.subTest(update=update.update_id):
                self.assertEqual((await self.route.publish_context(update)).status, "rejected")
        with self.assertRaises(RealtimeRouteError):
            await self.route.publish_context(make_context(make_identity(2), revision=3))

    async def test_guard_rejection_prevents_context_write(self):
        self.control.valid = False
        before = sum(event["type"] == "conversation.item.create" for event in self.control.writes)
        receipt = await self.route.publish_context(make_context(revision=1))
        self.assertEqual(receipt.status, "unknown")
        self.assertEqual(sum(event["type"] == "conversation.item.create" for event in self.control.writes), before)

    async def test_interruption_does_not_wait_for_context_guard(self):
        update = make_context(revision=1)
        self.control.block_id = update.update_id
        publication = asyncio.create_task(self.route.publish_context(update))
        await asyncio.wait_for(self.control.guard_entered.wait(), 0.5)
        command = InterruptSpeechCommand(route=self.route.identity, command_id=uuid4(),
                                         requested_at=NOW, reason="barge_in")
        receipt = await asyncio.wait_for(self.route.interrupt_speech(command), 0.1)
        self.assertEqual(receipt.status, "completed")
        self.assertTrue(receipt.output_fenced)
        self.assertIsNone(receipt.provider_stopped)
        self.assertFalse(publication.done())
        self.control.guard_release.set()
        self.assertEqual((await publication).status, "written")

    async def test_retirement_fences_blocked_write_and_requires_reconnect(self):
        update = make_context(revision=1)
        self.control.block_id = update.update_id
        publication = asyncio.create_task(self.route.publish_context(update))
        await asyncio.wait_for(self.control.guard_entered.wait(), 0.5)
        command = SourceRetirementCommand(route=self.route.identity, command_id=uuid4(),
            requested_at=NOW, cursor=update.cursor, sources=(make_offer().attempt.source,), reason="withdrawn")
        receipt = await self.route.retire_context(command)
        self.assertEqual(receipt.status, "reconnect_required")
        await asyncio.gather(publication, return_exceptions=True)
        self.control.guard_release.set()
        self.assertEqual((await self.route.publish_context(make_context(revision=2))).status, "rejected")
        self.assertEqual(sum(event["type"] == "conversation.item.create" for event in self.control.writes), 1)

    async def test_notification_uses_admission_guard_and_explicit_response_binding(self):
        offer = make_offer()
        evidence = await self.route.offer_notification(offer)
        self.assertEqual(evidence.state, DeliveryEvidenceState.SUBMITTED)
        self.assertEqual(len(self.control.admissions), 1)
        admitted_offer, request_id, _, _ = self.control.admissions[0]
        self.assertIs(admitted_offer, offer)
        self.assertIs(self.control.guarded[-1], offer)
        self.assertFalse(self.route.bind_response(self.route.identity, offer.offer_id, "wrong", "r1"))
        self.assertTrue(self.route.bind_response(self.route.identity, offer.offer_id, request_id, "r1"))
        event = {"type": "response.output_item.added", "response_id": "r1",
                 "item": {"id": "i1", "type": "message", "phase": "final_answer"}}
        self.route.observe_provider_event(self.route.identity, event, admitted=False)
        self.route.observe_provider_event(self.route.identity, event, admitted=True)
        self.route.observe_provider_event(self.route.identity,
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}}, admitted=True)
        # Repeating an identical offer cannot submit another response.
        await self.route.offer_notification(offer)
        self.assertEqual(len(self.control.admissions), 1)
        await stop_route(self.route)
        observations = [event async for event in self.route.events() if isinstance(event, DeliveryEvidence)]
        self.assertEqual(sum(event.state == DeliveryEvidenceState.OUTPUT_STARTED for event in observations), 1)
        self.assertEqual(sum(event.state == DeliveryEvidenceState.OUTPUT_FINISHED for event in observations), 1)
        self.assertTrue(all(event.playback is None for event in observations))

    async def test_commentary_suppression_and_stale_response_binding(self):
        offer = make_offer()
        await self.route.offer_notification(offer)
        request_id = self.control.admissions[0][1]
        self.assertTrue(self.route.bind_response(self.route.identity, offer.offer_id, request_id, "r1"))
        self.route.observe_provider_event(self.route.identity,
            {"type": "response.output_item.added", "response_id": "r1",
             "item": {"id": "i1", "type": "message", "phase": "commentary"}}, admitted=True)
        await self.route.interrupt_speech(InterruptSpeechCommand(route=self.route.identity,
            command_id=uuid4(), requested_at=NOW, reason="barge_in"))
        self.assertFalse(self.route.bind_response(self.route.identity, offer.offer_id, request_id, "r2"))
        await stop_route(self.route)
        observations = [event async for event in self.route.events() if isinstance(event, DeliveryEvidence)]
        self.assertFalse(any(event.state == DeliveryEvidenceState.OUTPUT_STARTED for event in observations))

    async def test_close_requires_matching_client_fence_receipt(self):
        self.control.wrong_fence = True
        receipt = await stop_route(self.route)
        self.assertEqual(receipt.status, "unknown")
        self.assertFalse(receipt.output_fenced)
        self.assertTrue(self.control.closed)

    async def test_pcm_render_evidence_requires_actual_attempt_output_and_audio_binding(self):
        control = FakeControl(make_identity(2))
        route = make_adapter(control, media_transport="websocket", playback=True)
        try:
            await route.start(make_start(control.identity))
            offer = make_offer(control.identity)
            await route.offer_notification(offer)
            route.bind_response(route.identity, offer.offer_id, control.admissions[0][1], "r-pcm")
            route.observe_provider_event(route.identity,
                {"type": "response.output_item.added", "response_id": "r-pcm",
                 "item": {"id": "i-pcm", "type": "message", "phase": "final_answer"}}, admitted=True)
            iterator = route.events()
            output = None
            while output is None:
                event = await anext(iterator)
                if isinstance(event, DeliveryEvidence) and event.state == DeliveryEvidenceState.OUTPUT_STARTED:
                    output = event.output
            await iterator.aclose()
            evidence = DeliveryEvidence(evidence_id=uuid4(), route=route.identity, transport=VoiceTransport.NATIVE_VOICE,
                observed_at=NOW, state=DeliveryEvidenceState.CLIENT_PLAYBACK_OBSERVED, attribution="explicit",
                offer_id=offer.offer_id, attempt=offer.attempt, output=output,
                playback=ClientPlaybackObservation(receipt_id=uuid4(), observed_at=NOW, rendered_ms=20))
            self.assertFalse(route.accept_client_playback(evidence, audio_session_id="wrong"))
            self.assertTrue(route.accept_client_playback(evidence, audio_session_id="audio-fixture"))
        finally:
            await stop_route(route)


if __name__ == "__main__":
    unittest.main()
