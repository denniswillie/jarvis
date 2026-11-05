"""Controlled real native delivery/ledger/runtime/writer composition, not SQL/native.

The canonical repository fixture supplies source/lease state; transport and
copied-source locking are controlled. No provider or client hearing is inferred.
"""
import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from app.attention.models import AttentionError
from app.attention.native_voice import NativeAttentionSink
from app.realtime.direct_session import DirectRealtimeAudioSession
from app.realtime.models import PendingRealtimeResponse
from app.realtime.socket_sender import WebSocketEventSender
from app.realtime.source_context import LiveContextError
from app.realtime.voice_route_runtime import RealtimeVoiceRouteRuntime, _Dispatch
from app.voice_routes.contracts import CommandReceipt, DeliveryEvidence, RouteIdentity
from app.voice_routes.coordinator import CoordinationPhase
from test_notification_ledger import CanonicalFixture, NOW, capabilities
from test_realtime_voice_route import FakeControl


class BorrowedSources:
    """Only the downstream copied-source DB boundary is controlled here."""
    def __init__(self, fixture):
        self.fixture = fixture
        self.borrowed = False
        self.entries = 0
        self.failure = None
        self.snapshot = object()

    def snapshot_for(self, request_id):
        return self.snapshot

    @asynccontextmanager
    async def use_transaction(self, connection):
        assert connection is self.fixture.connection and self.fixture.lock.locked()
        self.borrowed = True
        try:
            yield
        finally:
            self.borrowed = False

    @asynccontextmanager
    async def context_guard(self, *, extra_sources):
        assert self.borrowed and self.fixture.lock.locked()
        self.entries += 1
        if self.failure:
            raise self.failure
        yield

    @asynccontextmanager
    async def output_guard(self, snapshot):
        assert self.borrowed and self.fixture.lock.locked()
        yield


class NativeAttentionReadinessRaceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = CanonicalFixture()
        self.f.lease_seconds = 60
        self.f.renew = AsyncMock()
        self.f.settle = AsyncMock()
        self.route = RouteIdentity(scope=self.f.scope, route_id=uuid4(), route_generation=1)
        self.voice = voice = object.__new__(DirectRealtimeAudioSession)
        voice._closed, voice._context_revoked = asyncio.Event(), False
        voice.context = NS(owner_id=self.f.scope.owner_id, jarvis_session_id=self.f.scope.session_id,
                           audio_session_id=str(uuid4()))
        self.provider = voice.openai = FakeControl(self.route)
        voice._trace, voice._native_control_diagnostic = Mock(), Mock()
        voice._direct_output_pending, voice._direct_audio_responses = set(), set()
        voice._active_responses = {}
        voice._retired_response_ids = set()
        voice._turn_generation = 1
        voice._response_admission_lock = asyncio.Lock()
        voice._unacknowledged_response_requests = {}
        voice._latest_response_request_id = None
        voice._aborted_continuation_requests = set()
        voice._tool_chain_by_response_id, voice._tool_budgets = {}, {}
        voice._context_capture = voice._visual_delivery = voice._pending_response = None
        voice._user_audio_pending = False
        voice._send_speech_notification = AsyncMock()
        voice._send_response_cancel_locked = AsyncMock()
        voice.settings = NS(JARVIS_MAX_TOOL_CALLS_PER_RESPONSE=4, JARVIS_MAX_REPEATED_TOOL_CALLS_PER_RESPONSE=2)
        voice._attention_delivery = None
        voice._response_controls = NS(hold_fence=Mock())
        voice._source_context = self.sources = BorrowedSources(self.f)
        voice._begin_context_rebuild = Mock()
        voice._mark_closing = Mock(side_effect=lambda _: voice._closed.set())

        async def release(delivery):
            if voice._attention_delivery is delivery:
                voice._attention_delivery = None
        voice.release_attention = release

        async def cancel(*, request_id, offer_id):
            self.assertEqual(request_id, "request_fixture")
            self.assertEqual(offer_id, str(self.delivery.offer.offer_id))
            voice._active_responses["resp_fixture"].interrupted = True
            voice._direct_output_pending.discard("resp_fixture")
        voice.cancel_route_notification = AsyncMock(side_effect=cancel)

        self.runtime = runtime = object.__new__(RealtimeVoiceRouteRuntime)
        runtime.session, runtime.request = voice, NS(route=self.route)
        runtime._installed, runtime._closing, runtime._closed = True, False, False
        runtime._context_replacement_active = False
        runtime._unavailability_reported = False
        runtime._dispatches, runtime._response_offers = {}, {}
        runtime._dispatch_context = ContextVar("fixture_dispatch", default=None)
        runtime._notification_output_context = ContextVar("fixture_output", default=None)
        runtime._input_provider = voice.openai
        runtime._input_audio_session_id = voice.context.audio_session_id
        voice._voice_route_runtime = runtime
        runtime.channel = NS(ready=True, identity=self.route, audio_session_id=voice.context.audio_session_id,
                             media_envelope=lambda payload: payload)
        runtime.coordinator = NS(status=NS(phase=CoordinationPhase.ACTIVE, identity=self.route, reason=None),
                                 _bridge=NS(awaiting_fresh_context=False))
        runtime._payload_roots = lambda offer: (offer.attempt.source_payload_id,)
        self.socket = NS(send_text=AsyncMock())
        self.sender = WebSocketEventSender(self.socket, trace=voice._trace)
        runtime._owner_send_text = self.sender.send_text
        self.fence_confirmed = True

        async def fence(command):
            # No network/control handoff while the app writer or owner lock is held.
            self.assertFalse(self.sender._lock.locked())
            self.assertFalse(self.f.lock.locked())
            return CommandReceipt(route=self.route, command_id=command.command_id, operation="interrupt_speech",
                observed_at=NOW, status="completed" if self.fence_confirmed else "unknown",
                output_fenced=self.fence_confirmed)
        runtime.adapter = NS(capabilities=capabilities(), fence_output_only=AsyncMock(side_effect=fence),
            context_reconciliation_pending=False, observe_item_event=Mock(),
            bind_response=Mock(return_value=True), observe_provider_event=Mock())
        self.sink = NativeAttentionSink(voice, runtime, self.f, AsyncMock(), clock=lambda: NOW)
        runtime._atomic_guard = self.sink.guard
        self.admitted = asyncio.Event()
        self.epoch_current = True
        self.write_outcome = True
        self.write_error = None

        async def provider_send(event, *, write_guard):
            if self.write_outcome is False:
                return False
            result = await self.provider.send_event(event, write_guard=write_guard)
            if self.write_outcome == "unknown":
                raise TimeoutError("Controlled loss of transport completion")
            return result
        runtime._owner_provider_send = provider_send
        runtime._owner_provider_handler = voice._handle_openai_event

        async def request_notification(offer, *, turn_generation):
            # Controlled source preparation, followed by the REAL dispatch writer
            # and REAL response registration. No written bit is preseeded.
            metadata = {"jarvis_request_id": "request_fixture", "jarvis_kind": "prospective",
                        "jarvis_turn_generation": str(turn_generation)}
            pending = PendingRealtimeResponse(
                request_id="request_fixture", kind="prospective", metadata=metadata,
                submitted=True, turn_generation=turn_generation)
            self.assertIsNone(pending.request_sources)
            voice._unacknowledged_response_requests[pending.request_id] = pending
            voice._latest_response_request_id = pending.request_id
            return await runtime._provider_send({"type": "response.create", "event_id": pending.request_id,
                                                "response": {"metadata": dict(metadata)}})
        voice.request_route_notification = request_notification

        async def offer_notification(offer):
            # Real ledger submission/ticket consumption, with controlled external admission.
            async with self.sink.guard(offer):
                pass
            try:
                accepted = await runtime._admit_response(offer, "offer_request",
                    lambda: runtime._source_guard(offer), lambda: self.epoch_current)
            except TimeoutError as error:
                self.write_error, accepted = error, False
            self.admitted.set()
            return DeliveryEvidence(evidence_id=uuid4(), route=offer.route, transport="native_voice",
                observed_at=NOW, state="submitted" if accepted else "unknown", attribution="explicit", offer_id=offer.offer_id,
                attempt=offer.attempt)
        runtime.offer_notification = AsyncMock(side_effect=offer_notification)

        async def app_event(kind, **fields):
            return await runtime._app_send_text(json.dumps(dict(type=kind, **fields)))
        voice._send_app_event = app_event
        self.sink.start()
        self.addAsyncCleanup(self.sink.close)

    async def start_delivery(self, *, correlate=True, write_outcome=True):
        self.write_outcome = write_outcome
        voice = self.voice
        self.delivery = self.sink.create_delivery(self.f, None, self.f.claim, voice, receipt_seconds=1)
        voice._attention_delivery = self.delivery
        self.delivery.task = asyncio.create_task(self.delivery.run())
        await asyncio.wait_for(self.admitted.wait(), 1)
        await asyncio.wait_for(self.sink._events.join(), 1)
        if correlate:
            await self.correlate()
        self.socket.send_text.reset_mock()
        self.sources.entries = 0

    def created(self, **metadata_changes):
        dispatch = self.runtime._dispatches["offer_request"]
        metadata = {"jarvis_request_id": "request_fixture", "jarvis_kind": "prospective",
                    "jarvis_turn_generation": "1", "voice_route_id": str(self.route.route_id),
                    "voice_route_generation": str(self.route.route_generation),
                    "voice_route_offer_id": str(dispatch.offer.offer_id), "voice_route_request_id": dispatch.request_id}
        return {"type": "response.created", "response": {"id": "resp_fixture", "metadata": metadata | metadata_changes}}

    async def correlate(self, **metadata_changes):
        await self.runtime._provider_event(self.created(**metadata_changes))

    async def audio_start(self):
        await self.voice._handle_openai_event({"type": "output_audio_buffer.started", "response_id": "resp_fixture"})

    def guard_mutation(self, mutation):
        original = self.f.guard

        @asynccontextmanager
        async def mutate(claim):
            async with original(claim) as value:
                await asyncio.sleep(0)
                mutation()
                yield value
        self.f.guard = mutate

    def expire_context(self):
        self.runtime.coordinator._bridge.awaiting_fresh_context = True

    async def assert_unknown_once(self):
        await asyncio.wait_for(self.delivery.task, 1)
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.assertEqual(self.f.intent["state"], "unknown")
        self.assertEqual(self.f.attempt["reason"], "native_context_not_ready")
        self.assertIsNone(self.f.attempt["played_samples"])
        self.assertEqual(self.f.operations.count("submit"), 1)
        self.runtime.offer_notification.assert_awaited_once()
        self.runtime.adapter.fence_output_only.assert_awaited_once()
        self.voice.cancel_route_notification.assert_awaited_once()
        self.socket.send_text.assert_not_awaited()
        self.assertFalse(self.voice._closed.is_set())
        self.assertFalse(self.f.lock.locked())

    async def test_audio_start_readiness_flip_inside_canonical_guard_keeps_exact_inflight_output(self):
        await self.start_delivery()
        self.guard_mutation(self.expire_context)
        await self.audio_start()
        self.socket.send_text.assert_awaited_once()
        self.assertEqual(self.sources.entries, 1)
        self.assertFalse(self.runtime.ready)
        self.assertFalse(self.voice._closed.is_set())
        self.assertEqual(self.f.attempt["phase"], "submitted")
        self.assertIsNone(self.f.attempt["played_samples"])
        self.runtime.adapter.fence_output_only.assert_not_awaited()
        self.assertEqual(self.f.operations.count("submit"), 1)

    async def test_already_unready_output_is_not_identity_loss_or_new_admission(self):
        await self.start_delivery()
        self.expire_context()
        self.assertTrue(self.delivery.owns_offer(self.delivery.offer))
        self.assertFalse(self.delivery.accepts_offer(self.delivery.offer))
        self.assertIsNone(await self.sink.try_attention(self.f, worker_id="fixture", receipt_seconds=1))
        self.sink.admission.assert_not_awaited()
        await self.audio_start()
        self.socket.send_text.assert_awaited_once()
        self.runtime.adapter.fence_output_only.assert_not_awaited()

    async def test_expiry_before_response_created_preserves_real_written_request_correlation(self):
        await self.start_delivery(correlate=False)
        dispatch = self.runtime._dispatches["offer_request"]
        self.assertTrue(dispatch.written)
        self.assertEqual(len(self.provider.writes), 1)
        self.expire_context()
        await self.correlate()
        state = self.voice._active_responses["resp_fixture"]
        self.assertIs(state.source_snapshot, self.sources.snapshot)
        self.assertFalse(state.interrupted)
        self.runtime.adapter.bind_response.assert_called_once()
        self.assertIs(self.runtime._response_offers["resp_fixture"], dispatch)
        self.socket.send_text.assert_awaited_once()  # The existing response permit.
        await self.audio_start()
        self.assertEqual(self.socket.send_text.await_count, 2)
        self.assertEqual(len(self.provider.writes), 1)
        self.runtime.adapter.fence_output_only.assert_not_awaited()

    async def test_consumed_ticket_without_actual_request_write_cannot_authorize_output(self):
        await self.start_delivery(correlate=False, write_outcome=False)
        self.assertTrue(self.delivery.ticket._used)
        self.assertFalse(self.runtime._dispatches["offer_request"].written)
        self.assertEqual(self.provider.writes, [])
        self.expire_context()
        await self.correlate()
        self.socket.send_text.assert_not_awaited()
        self.runtime.adapter.bind_response.assert_not_called()
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.runtime.offer_notification.assert_awaited_once()

    async def test_unknown_request_write_is_never_upgraded_by_a_late_response(self):
        await self.start_delivery(correlate=False, write_outcome="unknown")
        self.assertIsInstance(self.write_error, TimeoutError)
        self.assertEqual(len(self.provider.writes), 1)
        self.assertFalse(self.runtime._dispatches["offer_request"].written)
        await self.correlate()
        self.socket.send_text.assert_not_awaited()
        self.runtime.adapter.bind_response.assert_not_called()
        self.runtime.offer_notification.assert_awaited_once()

    async def test_foreign_response_metadata_cannot_fall_back_to_normal_output(self):
        await self.start_delivery(correlate=False)
        await self.correlate(voice_route_generation="999")
        self.socket.send_text.assert_not_awaited()
        self.runtime.adapter.bind_response.assert_not_called()
        self.assertNotIn("resp_fixture", self.runtime._response_offers)
        await self.audio_start()
        self.socket.send_text.assert_not_awaited()

    async def test_healthy_control_and_forward_marker_still_write_once(self):
        await self.start_delivery()
        marker = Mock()

        @asynccontextmanager
        async def guard():
            yield marker
        self.assertTrue(await self.runtime._app_send_text(
            json.dumps({"type": "audio.webrtc.started", "response_id": "resp_fixture"}), write_guard=guard))
        self.socket.send_text.assert_awaited_once()
        marker.assert_called_once()
        self.assertEqual(self.sources.entries, 1)
        self.runtime.adapter.fence_output_only.assert_not_awaited()

    async def test_real_route_generation_change_remains_foreign(self):
        await self.start_delivery()
        new_route = self.route.model_copy(update={"route_generation": 2})
        self.guard_mutation(lambda: setattr(self.runtime.request, "route", new_route))
        with self.assertRaises(AttentionError) as caught:
            await self.audio_start()
        self.assertEqual(caught.exception.code, "attention_offer_not_current")
        self.socket.send_text.assert_not_awaited()
        self.runtime.adapter.fence_output_only.assert_not_awaited()
        self.runtime.request.route = self.route

    async def test_replaced_provider_or_audio_connection_is_not_recoverable_readiness(self):
        await self.start_delivery()
        for change in ("provider", "audio", "channel", "closed"):
            with self.subTest(change=change):
                old_provider, old_audio, old_identity = self.voice.openai, self.runtime.channel.audio_session_id, self.runtime.channel.identity
                def replace():
                    self.expire_context()
                    if change == "provider":
                        self.voice.openai = object()
                    elif change == "audio":
                        self.runtime.channel.audio_session_id = str(uuid4())
                    elif change == "channel":
                        self.runtime.channel.identity = self.route.model_copy(update={"route_generation": 2})
                    else:
                        self.runtime._closing = True
                original = self.f.guard
                self.guard_mutation(replace)
                with self.assertRaises(AttentionError):
                    await self.audio_start()
                self.socket.send_text.assert_not_awaited()
                self.runtime.adapter.fence_output_only.assert_not_awaited()
                self.f.guard = original
                self.voice.openai, self.runtime.channel.audio_session_id, self.runtime.channel.identity = old_provider, old_audio, old_identity
                self.runtime._closing = False
                self.runtime.coordinator._bridge.awaiting_fresh_context = False

    async def test_source_withdrawal_and_lost_lease_are_not_swallowed(self):
        await self.start_delivery()
        for code in ("attention_source_unavailable", "attention_claim_lost"):
            with self.subTest(code=code):
                self.f.source_problem = "withdrawn" if code == "attention_source_unavailable" else None
                self.f.lease_valid = code != "attention_claim_lost"
                with self.assertRaises(AttentionError) as caught:
                    await self.audio_start()
                self.assertEqual(caught.exception.code, code)
                self.socket.send_text.assert_not_awaited()
                self.runtime.adapter.fence_output_only.assert_not_awaited()
        self.f.source_problem, self.f.lease_valid = None, True

    async def test_copied_source_withdrawal_still_rebuilds_and_never_writes(self):
        await self.start_delivery()
        self.sources.failure = LiveContextError("live_context_revoked")
        with self.assertRaises(LiveContextError):
            await self.audio_start()
        self.socket.send_text.assert_not_awaited()
        self.voice._begin_context_rebuild.assert_called_once()
        self.runtime.adapter.fence_output_only.assert_not_awaited()

    async def test_same_typed_error_after_guard_yield_is_unknown_not_recovered_or_replayed(self):
        await self.start_delivery()
        self.socket.send_text.side_effect = AttentionError("attention_route_not_ready", "PRIVATE_AFTER_WRITE")
        with self.assertRaises(AttentionError):
            await self.audio_start()
        self.socket.send_text.assert_awaited_once()
        self.runtime.adapter.fence_output_only.assert_not_awaited()
        self.assertEqual(self.f.operations.count("submit"), 1)

    async def test_unconfirmed_fence_keeps_existing_session_close_policy(self):
        await self.start_delivery()
        self.fence_confirmed = False
        def unsafe():
            self.expire_context()
            self.runtime.adapter.context_reconciliation_pending = True
        self.guard_mutation(unsafe)
        await self.audio_start()
        await asyncio.wait_for(self.delivery.task, 1)
        self.voice._mark_closing.assert_called_once_with("native_attention_fence_unconfirmed")
        self.assertTrue(self.voice._closed.is_set())
        self.assertEqual(self.f.attempt["phase"], "unknown")
        self.socket.send_text.assert_not_awaited()

    async def test_cancellation_is_not_converted_to_readiness_recovery(self):
        await self.start_delivery()
        original = self.f.guard

        @asynccontextmanager
        async def cancelled(claim):
            async with original(claim):
                raise asyncio.CancelledError
                yield
        self.f.guard = cancelled
        with self.assertRaises(asyncio.CancelledError):
            await self.audio_start()
        self.f.guard = original
        self.socket.send_text.assert_not_awaited()
        self.runtime.adapter.fence_output_only.assert_not_awaited()

    async def test_diagnostic_only_preserves_typed_static_codes(self):
        await self.start_delivery()
        for error, expected in ((AttentionError("attention_offer_not_current", "PRIVATE_MESSAGE"), "attention_offer_not_current"),
                                (AttentionError("PRIVATE_UNKNOWN_CODE", "PRIVATE_MESSAGE"), "unrecognized_code"),
                                (NS(code="attention_route_not_ready"), "unrecognized_code")):
            with self.subTest(expected=expected), self.assertLogs("app.realtime.voice_route_runtime", level="WARNING") as logs:
                self.runtime._readiness_diagnostic("source_guard_rejected", source_error=error, guard_yielded=False)
            record = json.loads(logs.output[-1].split("voice_route_readiness_diagnostic ", 1)[1])
            self.assertEqual(record["error_code"], expected)
            self.assertNotIn("PRIVATE", repr(logs.output))

    async def test_unsafe_replacement_uses_existing_confirmed_fence_without_resend(self):
        await self.start_delivery()
        def unsafe():
            self.expire_context()
            self.runtime._context_replacement_active = True
        self.guard_mutation(unsafe)
        await self.audio_start()
        await self.assert_unknown_once()
        for _ in range(3):
            await self.audio_start()
        self.runtime.adapter.fence_output_only.assert_awaited_once()

    async def test_new_turn_and_adapter_epoch_change_remain_output_fences(self):
        await self.start_delivery()
        for boundary in ("turn", "epoch"):
            with self.subTest(boundary=boundary):
                if boundary == "turn":
                    self.voice._turn_generation = 2
                else:
                    self.epoch_current = False
                await self.audio_start()
                self.socket.send_text.assert_not_awaited()
                self.voice._turn_generation, self.epoch_current = 1, True

    async def test_output_guard_scope_does_not_leak_to_new_provider_writes_or_child_tasks(self):
        await self.start_delivery()
        self.expire_context()
        observed = []

        @asynccontextmanager
        async def guard():
            observed.append(self.runtime.notification_output_current(self.delivery.offer))
            async def child():
                return self.runtime.notification_output_current(self.delivery.offer)
            observed.append(await asyncio.create_task(child()))
            yield
        await self.runtime._app_send_text(json.dumps({"type": "audio.webrtc.started", "response_id": "resp_fixture"}),
                                          write_guard=guard)
        self.assertEqual(observed, [True, False])
        self.assertFalse(self.runtime.notification_output_current(self.delivery.offer))
        with self.assertRaises(AttentionError) as caught:
            async with self.sink.guard(self.delivery.offer):
                self.fail("A consumed ticket authorized a new unready provider write")
        self.assertEqual(caught.exception.code, "attention_route_not_ready")

    async def test_provider_completion_does_not_manufacture_pending_playback_authority(self):
        await self.start_delivery()
        await self.audio_start()
        voice = self.voice
        voice._responses_with_tool_calls = set()
        voice._persist_assistant_transcript = AsyncMock()
        voice._finish_response_admission = AsyncMock(return_value=None)
        voice._finish_tool_response = Mock()
        voice._finish_prospective_response = AsyncMock()
        voice._send_error = AsyncMock()
        await voice._finish_provider_response("resp_fixture", {"status": "completed"})
        self.assertIn("resp_fixture", voice._direct_output_pending)
        self.assertNotIn("resp_fixture", voice._active_responses)
        self.socket.send_text.reset_mock()
        self.expire_context()
        await self.audio_start()
        self.socket.send_text.assert_not_awaited()
        await voice._handle_openai_event({"type": "output_audio_buffer.stopped", "response_id": "resp_fixture"})
        self.socket.send_text.assert_awaited_once()  # End-only control, not fresh content.
        self.assertNotIn("resp_fixture", voice._direct_output_pending)
        self.assertIsNone(self.f.attempt["played_samples"])
