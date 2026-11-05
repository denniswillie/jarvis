"""Offline wiring against one existing-owner-shaped session; no external I/O."""

import asyncio
import inspect
import json
import unittest
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

from app.live_state.models import SessionScope, SourceDependency, StateCursor
from app.realtime.models import PendingRealtimeResponse, RealtimeResponseState
from app.realtime.session import RealtimeAudioSession
from app.realtime.source_context import ResponseSourceSnapshot
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime
from app.voice_routes.contracts import (
    CommandReceipt, DeliveryAttemptReference, DeliveryEvidenceState, NotificationOffer,
    RouteIdentity, RouteStart, VoiceContextUpdate,
)


NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
ROOT = UUID(int=77)
CLIENT_CAPS = {
    "supported_versions": [1], "supported_transports": ["native_voice"],
    "supported_media_topologies": ["phone_to_provider"], "input_codecs": ["opus"],
    "output_codecs": ["opus"], "local_output_fencing": True, "client_playback_boundaries": [],
}


class Sources:
    def __init__(self):
        self.roots = set()
        self.entries = 0

    def observe(self, roots):
        self.roots.update(roots)

    @asynccontextmanager
    async def context_guard(self, *, extra_sources=()):
        self.entries += 1
        yield

    @asynccontextmanager
    async def output_guard(self, snapshot):
        assert snapshot is self.snapshot
        self.entries += 1
        yield


class Provider:
    transport = "webrtc"

    def __init__(self):
        self.lock = asyncio.Lock()
        self.writes = []
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
        # A deliberate fake-provider receipt, not evidence inferred by the
        # production adapter from a successful socket write.
        if self.item_event is not None:
            if payload["type"] == "conversation.item.create":
                await self.item_event({"type": "conversation.item.done", "item": payload["item"]})
            elif payload["type"] == "conversation.item.delete":
                await self.item_event({"type": "conversation.item.deleted", "item_id": payload["item_id"]})
        return True

    async def close(self):
        self.closed = True


class Sender:
    def __init__(self, session):
        self.session = session
        self.messages = []
        self.fence_ack = True
        self.lock = asyncio.Lock()
        self.forwarded_marks = 0

    async def send_event(self, payload):
        self.messages.append(payload)
        if payload["type"] == "voice_route.selected":
            await self.session._handle_client_event({"type": "voice_route.ready", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "route": payload["route"],
                "client_event_id": "ready-fixture"})
        elif payload["type"] in {"voice_route.interrupt", "voice_route.close"} and self.fence_ack:
            command = payload["payload"]
            receipt = CommandReceipt(route=command["route"], command_id=command["command_id"],
                operation="close" if payload["type"] == "voice_route.close" else "interrupt_speech",
                observed_at=NOW, status="completed", output_fenced=True)
            await self.session._handle_client_event({"type": "voice_route.event", "protocol_version": 1,
                "audio_session_id": payload["audio_session_id"], "payload": receipt.model_dump(mode="json"),
                "client_event_id": "receipt-fixture"})
        return True

    async def send_text(self, text, *, should_send=None, write_guard=None):
        async with self.lock:
            if write_guard is not None:
                async with write_guard() as marker:
                    if should_send is not None and not should_send():
                        return False
                    self.messages.append(json.loads(text))
                    if marker is not None:
                        marker()
                    return True
            if should_send is not None and not should_send():
                return False
            self.messages.append(json.loads(text))
            return True


class Session:
    def __init__(self):
        self.context = SimpleNamespace(owner_id="owner", jarvis_session_id=UUID(int=1), audio_session_id="audio-1")
        self.openai = Provider()
        # Resolve the current installed receiver, not a stale pre-runtime bound
        # method. Missing/rejected-ACK fixtures may set this to None or replace it.
        self.openai.item_event = lambda payload: self._handle_openai_event(payload)
        self._app_sender = Sender(self)
        self._source_context = Sources()
        self._realtime_context_ready = asyncio.Event()
        self._closed = asyncio.Event()
        self._context_revoked = False
        self._voice_route_runtime = None
        self._attention_delivery = None
        self._response_admission_lock = asyncio.Lock()
        self._turn_generation = 2
        self._user_audio_pending = False
        self._unacknowledged_response_requests = {}
        self._active_responses = {}
        self._direct_output_pending = set()
        self._close_reason = "unknown"
        self.request_calls = self.legacy_speech_calls = self.cancels = self.original_guards = 0
        self.handled = []
        self.resources_closed = False

    async def _handle_client_event(self, event):
        self.handled.append(event)
        if event["type"] == "response.cancel":
            async with self._response_admission_lock:
                self._turn_generation += 1
                self.cancels += 1

    async def _handle_openai_event(self, payload):
        if payload["type"] == "response.created":
            response = payload["response"]
            request_id = response["metadata"]["jarvis_request_id"]
            pending = self._unacknowledged_response_requests.pop(request_id, None)
            if pending is not None:
                self._active_responses[response["id"]] = RealtimeResponseState(
                    response_id=response["id"], kind=pending.kind, metadata=response["metadata"],
                    source_snapshot=getattr(self._source_context, "snapshot", None))
                await self._send_app_event("audio.webrtc.response", response_id=response["id"])
        elif payload["type"] == "response.output_item.added":
            await self._send_app_event("audio.webrtc.output_item", response_id=payload["response_id"], item=payload["item"])

    async def _send_app_event(self, event_type, **payload):
        @asynccontextmanager
        async def guard():
            self.original_guards += 1
            def mark():
                self._app_sender.forwarded_marks += 1
            yield mark
        return await self._app_sender.send_text(json.dumps({"type": event_type, **payload}),
                                                should_send=lambda: not self._closed.is_set(), write_guard=guard)

    async def _request_normal_response(self, *, turn_generation):
        async with self._response_admission_lock:
            if turn_generation != self._turn_generation:
                return False
            self.request_calls += 1
            identity = uuid4().hex
            pending = PendingRealtimeResponse(request_id=identity, kind="normal_user_turn", turn_generation=turn_generation)
            self._unacknowledged_response_requests[identity] = pending
        @asynccontextmanager
        async def guard():
            self.original_guards += 1
            yield
        sent = await self.openai.send_event({"type": "response.create", "event_id": identity,
            "response": {"metadata": {"jarvis_request_id": identity, "turn_generation": str(turn_generation)}}},
            write_guard=guard)
        pending.submitted = sent
        return sent

    async def request_route_notification(self, offer, *, turn_generation):
        # Transport fixture only. Real no-user-turn admission is exercised by
        # test_native_attention_delivery against RealtimeAudioSession itself.
        return await self._request_normal_response(turn_generation=turn_generation)

    async def try_attention(self, *args, **kwargs):
        self.legacy_speech_calls += 1
        return "legacy"

    def _mark_closing(self, reason):
        self._close_reason = reason
        self._closed.set()

    async def _close_resources(self, reason):
        await self.openai.close()
        self.resources_closed = True


def request_for(session):
    identity = RouteIdentity(scope=SessionScope(owner_id="owner", session_id=session.context.jarvis_session_id,
        area_id=None), route_id=uuid4(), route_generation=1)
    context = VoiceContextUpdate(route=identity, update_id=uuid4(), cursor=StateCursor(epoch=uuid4(), revision=0),
        mode="snapshot", text="No source-specific context is selected.", dependencies=(), created_at=NOW,
        expires_at=NOW + timedelta(minutes=5))
    return RouteStart(route=identity, command_id=uuid4(), requested_at=NOW,
                      authorization_ref=uuid4(), initial_context=context)


async def eventually(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


class VoiceRouteRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.session = Session()
        self.request = request_for(self.session)
        self.guarded = []
        self.evidence = []
        @asynccontextmanager
        async def atomic(payload):
            self.guarded.append(payload)
            yield
        async def valid(_):
            return True
        self.runtime = RealtimeVoiceRouteRuntime(self.session, self.request, CLIENT_CAPS,
            atomic_guard=atomic, payload_roots=lambda _: (ROOT,), validity=valid,
            authorize_start=valid, on_event=self.evidence.append, clock=lambda: NOW,
            ready_timeout=0.2, control_timeout=0.05)
        RealtimeAudioSession.install_voice_route_runtime(self.session, self.runtime)

    async def asyncTearDown(self):
        await RealtimeAudioSession.close(self.session, "fixture_cleanup")

    async def ready(self):
        self.session._realtime_context_ready.set()
        await eventually(lambda: self.runtime.ready or self.runtime.failure is not None)
        self.assertTrue(self.runtime.ready, self.runtime.failure)

    async def test_opt_in_waits_for_provider_and_blocks_legacy_attention(self):
        self.assertFalse(self.runtime.ready)
        self.assertEqual(self.session._app_sender.messages, [])
        self.assertIsNone(await self.session.try_attention(None, None, worker_id="fixture", receipt_seconds=1))
        self.assertEqual(self.session.legacy_speech_calls, 0)
        await self.ready()
        self.assertIs(self.guarded[0], self.request.initial_context)
        self.assertIn(ROOT, self.session._source_context.roots)
        self.assertTrue(self.session._source_context.entries)

    async def test_fixture_item_acknowledgments_use_the_installed_receiver(self):
        await self.ready()
        old_item = self.runtime.adapter._context_item
        self.assertEqual(self.runtime.adapter._items[old_item].state, "created")
        initial = self.request.initial_context
        update = initial.model_copy(update={"update_id": uuid4(),
            "cursor": StateCursor(epoch=initial.cursor.epoch, revision=1),
            "text": "New fixture context."})
        self.assertEqual((await self.runtime.adapter.publish_context(update)).status, "written")
        self.assertNotIn(old_item, self.runtime.adapter._items)
        self.assertEqual(self.runtime.adapter._items[self.runtime.adapter._context_item].state, "created")
        deletes = [event for event in self.session.openai.writes if event["type"] == "conversation.item.delete"]
        self.assertEqual([event["item_id"] for event in deletes], [old_item])

    async def test_existing_handler_and_serialized_output_guard_remain_in_use(self):
        await self.ready()
        await self.session._handle_client_event({"type": "audio.input.begin", "segment_id": "original-handler"})
        self.assertEqual(self.session.handled[-1]["segment_id"], "original-handler")
        self.assertTrue(await self.session._send_app_event("audio.webrtc.response", response_id="ordinary"))
        message = self.session._app_sender.messages[-1]
        self.assertEqual(message["protocol_version"], 1)
        self.assertEqual(message["audio_session_id"], self.session.context.audio_session_id)
        self.assertEqual(message["route"], self.request.route.model_dump(mode="json"))
        self.assertEqual(self.session._app_sender.forwarded_marks, 1)

    async def test_native_mode_rejects_direct_legacy_pcm_writer(self):
        await self.ready()
        before = len(self.session._app_sender.messages)
        self.assertFalse(await self.session._app_sender.send_text(json.dumps({
            "type": "audio.output.delta", "audio_origin": "expert_tts", "audio_base64": "AAAA"})))
        self.assertEqual(len(self.session._app_sender.messages), before)

    async def test_native_buffer_event_names_match_existing_direct_session(self):
        await self.ready()
        for kind in ("audio.webrtc.started", "audio.webrtc.stopped", "audio.webrtc.cleared"):
            with self.subTest(kind=kind):
                self.assertTrue(await self.session._send_app_event(kind, response_id="ordinary"))
                self.assertEqual(self.session._app_sender.messages[-1]["type"], kind)

    async def test_degraded_route_forwards_only_response_scoped_end_controls(self):
        await self.ready()
        self.runtime.coordinator._publication_failed(self.request.route, "context_publication_deadline")
        self.assertFalse(self.runtime.ready)
        for kind in ("audio.webrtc.stopped", "audio.webrtc.cleared"):
            with self.subTest(kind=kind):
                self.assertTrue(await self.session._send_app_event(kind, response_id="ordinary"))
                message = self.session._app_sender.messages[-1]
                self.assertEqual(message["route"], self.request.route.model_dump(mode="json"))
                self.assertEqual(message["audio_session_id"], self.session.context.audio_session_id)
        for kind in ("audio.webrtc.response", "audio.webrtc.output_item", "audio.webrtc.started"):
            with self.subTest(kind=kind):
                self.assertFalse(await self.session._send_app_event(kind, response_id="ordinary"))
        self.assertFalse(await self.session.openai.send_event({"type": "response.create"}))
        self.assertEqual(self.session._app_sender.forwarded_marks, 2)

    async def test_degraded_end_controls_do_not_reauthorize_expired_notification_content(self):
        await self.ready()
        self.runtime.coordinator._publication_failed(self.request.route, "context_publication_deadline")
        self.runtime._response_offers["ordinary"] = SimpleNamespace(current=lambda: False, offer=object())
        before = len(self.guarded)
        self.assertTrue(await self.session._send_app_event("audio.webrtc.stopped", response_id="ordinary"))
        self.assertEqual(len(self.guarded), before)
        self.assertFalse(await self.session._app_sender.send_text(json.dumps({
            "type": "audio.webrtc.stopped", "response_id": "ordinary", "text": "not lifecycle data"})))
        self.assertFalse(await self.session._app_sender.send_text(json.dumps({"type": "audio.webrtc.stopped"})))
        self.assertFalse(await self.session._app_sender.send_text(json.dumps({
            "type": "audio.webrtc.stopped", "response_id": "ordinary"}), should_send=lambda: False))

    async def test_degraded_end_control_rechecks_channel_at_serialized_write(self):
        await self.ready()
        self.runtime.coordinator._publication_failed(self.request.route, "context_publication_deadline")
        await self.session._app_sender.lock.acquire()
        pending = asyncio.create_task(self.session._send_app_event("audio.webrtc.stopped", response_id="ordinary"))
        try:
            await asyncio.sleep(0)
            self.runtime.channel.close()
        finally:
            self.session._app_sender.lock.release()
        self.assertFalse(await pending)

    async def test_degraded_end_control_rejects_replacement_route_identity(self):
        await self.ready()
        self.runtime.coordinator._publication_failed(self.request.route, "context_publication_deadline")
        previous = self.runtime.coordinator._identity
        try:
            self.runtime.coordinator._identity = previous.model_copy(update={"route_id": uuid4()})
            self.assertFalse(await self.session._send_app_event("audio.webrtc.stopped", response_id="ordinary"))
        finally:
            self.runtime.coordinator._identity = previous

    async def test_notification_uses_existing_admission_and_keeps_original_metadata(self):
        await self.ready()
        offer = NotificationOffer(route=self.request.route, offer_id=uuid4(),
            attempt=DeliveryAttemptReference(notification_intent_id=uuid4(), attempt_id=uuid4(), attempt_generation=2,
                source=SourceDependency(source_namespace="task_result", source_id=uuid4(), revision=1, generation=1),
                source_payload_id=uuid4()), text="Result is ready.", offered_at=NOW, expires_at=NOW + timedelta(minutes=1))
        # This transport-only fixture must supply the same ownership and pin
        # shape required by native delivery; it does not qualify SQL authority.
        self.session._attention_delivery = SimpleNamespace(runtime=self.runtime,
            accepts_offer=lambda candidate: candidate == offer,
            owns_offer=lambda candidate: candidate == offer)
        self.session._source_context.snapshot = ResponseSourceSnapshot(
            uuid4(), (offer.attempt.source_payload_id,), 0, NOW)
        evidence = await self.runtime.offer_notification(offer)
        self.assertEqual(evidence.state, DeliveryEvidenceState.SUBMITTED)
        self.assertEqual(self.session.request_calls, 1)
        response = next(item for item in self.session.openai.writes if item["type"] == "response.create")
        metadata = response["response"]["metadata"]
        self.assertEqual(metadata["voice_route_offer_id"], str(offer.offer_id))
        self.assertIn(metadata["jarvis_request_id"], self.session._unacknowledged_response_requests)
        self.assertEqual(metadata["turn_generation"], "2")
        self.assertGreaterEqual(self.session.original_guards, 1)
        await self.session._handle_openai_event({"type": "response.created", "response": {"id": "response-1", "metadata": metadata}})
        await self.session._handle_openai_event({"type": "response.output_item.added", "response_id": "response-1",
            "item": {"id": "item-1", "type": "message", "phase": "final_answer"}})
        await eventually(lambda: any(item["type"] == "voice_route.output_bound" for item in self.session._app_sender.messages))
        self.assertTrue(any(item is offer for item in self.guarded))

    async def test_interrupt_intake_does_not_wait_for_receipt_reader(self):
        await self.ready()
        await asyncio.wait_for(self.session._handle_client_event({"type": "response.cancel"}), 0.05)
        await eventually(lambda: self.session.cancels == 1)
        await eventually(lambda: any(item["type"] == "voice_route.event" and
            item["payload"].get("operation") == "interrupt_speech" and item["payload"].get("status") == "completed"
            for item in self.session._app_sender.messages))

    async def test_shutdown_fences_before_provider_resources_and_joins_intake(self):
        await self.ready()
        await RealtimeAudioSession.close(self.session, "requested_close")
        self.assertTrue(self.runtime.shutdown_confirmed)
        self.assertTrue(self.session.resources_closed)
        self.assertTrue(self.session.openai.closed)
        self.assertFalse(self.runtime.ready)
        self.assertTrue(await self.runtime.after_session_close())

    async def test_missing_shutdown_fence_is_not_success(self):
        await self.ready()
        self.session._app_sender.fence_ack = False
        await RealtimeAudioSession.close(self.session, "unconfirmed_close")
        self.assertFalse(self.runtime.shutdown_confirmed)
        self.assertTrue(self.session.resources_closed)


if __name__ == "__main__":
    unittest.main()
