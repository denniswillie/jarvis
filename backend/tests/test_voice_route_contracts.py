"""Offline P0 contract fixtures, not provider, database or device qualification."""

import unittest
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from app.live_state.models import GuardedPayloadBinding, SessionScope, SourceDependency, StateCursor
from app.voice_routes.contracts import (
    ClientPlaybackObservation, CloseRouteCommand, CommandReceipt, ContextReceipt,
    DeliveryAttemptReference, DeliveryEvidence, DeliveryEvidenceState, HistoryTurn,
    InterruptSpeechCommand, NotificationOffer, OperationCapability, OutputIdentity,
    RouteCapabilities, RouteIdentity, RouteOperation, RouteStart,
    SourceRetirementCommand, SourceRetirementReceipt, Support, UnsupportedOperation,
    VoiceContextUpdate, VoiceRouteEvent, VoiceTransport,
)
from app.voice_routes.protocol import VoiceRoute


NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
OFFER_ID = UUID(int=33)


def route_identity(generation=3):
    return RouteIdentity(
        scope=SessionScope(owner_id="fixture-owner", session_id=UUID(int=1), area_id=None),
        route_id=UUID(int=2), route_generation=generation,
    )


def dependency():
    return SourceDependency(source_namespace="task_result", source_id=UUID(int=20),
                            revision=11, generation=5)


def attempt_reference(generation=7):
    return DeliveryAttemptReference(notification_intent_id=UUID(int=30),
        attempt_id=UUID(int=31), attempt_generation=generation,
        source=dependency(), source_payload_id=UUID(int=32))


def context_update(route=None, **changes):
    values = dict(route=route or route_identity(), update_id=UUID(int=60),
        cursor=StateCursor(epoch=UUID(int=61), revision=9), mode="snapshot",
        text="The background task has a committed result.", dependencies=(dependency(),),
        created_at=NOW, expires_at=NOW + timedelta(seconds=30))
    values["guard_binding"] = GuardedPayloadBinding(scope=values["route"].scope,
        payload_root_ids=(UUID(int=64),), dependencies=(dependency(),))
    values.update(changes)
    return VoiceContextUpdate(**values)


def notification_offer(route=None):
    return NotificationOffer(route=route or route_identity(), offer_id=OFFER_ID,
        attempt=attempt_reference(), text="The result is available in the inbox.",
        offered_at=NOW, expires_at=NOW + timedelta(seconds=30))


def output_identity():
    return OutputIdentity(output_id=UUID(int=40), provider_response_ref="response-1",
                          provider_item_ref="item-1")


def playback_evidence(**changes):
    values = dict(evidence_id=UUID(int=50), route=route_identity(),
        transport=VoiceTransport.NATIVE_VOICE, observed_at=NOW,
        state=DeliveryEvidenceState.CLIENT_PLAYBACK_OBSERVED, attribution="explicit",
        offer_id=OFFER_ID, attempt=attempt_reference(), output=output_identity(),
        playback=ClientPlaybackObservation(receipt_id=UUID(int=51), observed_at=NOW,
                                           rendered_ms=25))
    values.update(changes)
    return DeliveryEvidence(**values)


def capabilities():
    operations = tuple(OperationCapability(operation=operation,
        support=Support.UNSUPPORTED if operation == RouteOperation.RETIRE_CONTEXT else Support.SUPPORTED,
        reason="Fixture cannot retire provider context." if operation == RouteOperation.RETIRE_CONTEXT else None)
        for operation in RouteOperation)
    return RouteCapabilities(transport=VoiceTransport.NATIVE_VOICE,
        media_topology="phone_to_provider", control_owner="backend",
        credential_mode="server_only", input_codec="opus", output_codec="opus",
        vad_owner="provider", explicit_input_commit=Support.UNSUPPORTED,
        explicit_response_create=Support.SUPPORTED, full_duplex=Support.SUPPORTED,
        image_input=Support.UNSUPPORTED, provider_context_receipts=Support.SUPPORTED,
        notification_output_attribution=Support.SUPPORTED,
        client_playback_evidence=Support.SUPPORTED, provider_interruption=Support.UNSUPPORTED,
        operations=operations)


def start_request(route=None):
    identity = route or route_identity()
    return RouteStart(route=identity, command_id=UUID(int=62), requested_at=NOW,
        authorization_ref=UUID(int=63), initial_context=context_update(identity),
        relevant_history=(HistoryTurn(role="user", text="Tell me when the result is ready."),))


class FakeVoiceRoute(VoiceRoute):
    """Finite in-memory fixture. No ledger, provider, media or scheduling implementation."""

    def __init__(self, *, fence_confirmed=True):
        self._identity = route_identity()
        self._capabilities = capabilities()
        self._events = []
        self._offers = {}
        self._outputs = {}
        self._started = False
        self._closed = False
        self._fence_confirmed = fence_confirmed

    @property
    def identity(self):
        return self._identity

    @property
    def capabilities(self):
        return self._capabilities

    def _check(self, identity, *, ready=True):
        if identity != self.identity:
            raise ValueError("Wrong route scope, identity or generation.")
        if ready and (not self._started or self._closed):
            raise ValueError("Route is not ready.")

    def _emit(self, event):
        self._events.append(event)
        return event

    async def start(self, request):
        self._check(request.route, ready=False)
        self._started = True
        return self._emit(CommandReceipt(route=self.identity, command_id=request.command_id,
            operation="start", observed_at=NOW, status="completed"))

    async def publish_context(self, update):
        self._check(update.route)
        if update.expires_at is not None and update.expires_at <= NOW:
            raise ValueError("Expired context.")
        return self._emit(ContextReceipt(route=self.identity, update_id=update.update_id,
            cursor=update.cursor, observed_at=NOW, status="provider_acknowledged",
            provider_receipt_ref="context-ack-1"))

    async def offer_notification(self, offer):
        self._check(offer.route)
        if offer.expires_at <= NOW:
            raise ValueError("Expired offer.")
        self._offers[offer.offer_id] = offer
        return self._emit(DeliveryEvidence(evidence_id=UUID(int=100 + len(self._events)),
            route=self.identity, transport=self.capabilities.transport, observed_at=NOW,
            state=DeliveryEvidenceState.OFFERED, attribution="explicit",
            offer_id=offer.offer_id, attempt=offer.attempt))

    async def retire_context(self, command):
        self._check(command.route)
        return self._emit(UnsupportedOperation(route=self.identity,
            request_id=command.command_id, operation=RouteOperation.RETIRE_CONTEXT,
            observed_at=NOW, reason="Fixture cannot retire provider context."))

    async def interrupt_speech(self, command):
        self._check(command.route)
        return self._stop_receipt(command, "interrupt_speech")

    async def close(self, command):
        self._check(command.route, ready=False)
        receipt = self._stop_receipt(command, "close")
        self._closed = self._fence_confirmed
        return receipt

    def _stop_receipt(self, command, operation):
        return self._emit(CommandReceipt(route=self.identity, command_id=command.command_id,
            operation=operation, observed_at=NOW,
            status="completed" if self._fence_confirmed else "unknown",
            output_fenced=self._fence_confirmed, provider_stopped=None))

    def bind_observed_output(self, offer_id, output):
        """Test-only input representing an independently established output binding."""
        if offer_id not in self._offers:
            raise ValueError("Unknown offer.")
        self._outputs[offer_id] = output

    def accept_delivery(self, evidence):
        """Shape validation alone cannot establish correlation to accepted work."""
        self._check(evidence.route)
        offer = self._offers.get(evidence.offer_id)
        if offer is None or evidence.attempt != offer.attempt:
            raise ValueError("Attempt identity/generation does not match the offer.")
        if evidence.transport != self.capabilities.transport:
            raise ValueError("Wrong transport.")
        if evidence.output is not None and self._outputs.get(evidence.offer_id) != evidence.output:
            raise ValueError("Output is not bound to this attempt.")
        return self._emit(evidence)

    async def events(self) -> AsyncIterator[VoiceRouteEvent]:
        for event in tuple(self._events):
            yield event


class VoiceRouteModelTests(unittest.TestCase):
    def test_json_round_trip_preserves_nested_identities_and_tuples(self):
        request = start_request()
        restored = RouteStart.model_validate_json(request.model_dump_json())
        self.assertEqual(restored, request)
        self.assertIsInstance(restored.relevant_history, tuple)
        self.assertIsInstance(restored.initial_context.dependencies, tuple)
        evidence = playback_evidence()
        restored_event = TypeAdapter(VoiceRouteEvent).validate_json(evidence.model_dump_json())
        self.assertEqual(restored_event, evidence)
        self.assertIsInstance(restored_event, DeliveryEvidence)

    def test_frozen_contracts_and_forbidden_extra_fields(self):
        offer = notification_offer()
        with self.assertRaises(ValidationError):
            offer.text = "changed"
        with self.assertRaises(ValidationError):
            offer.attempt.attempt_generation = 99
        with self.assertRaises(ValidationError):
            NotificationOffer.model_validate({**offer.model_dump(), "human_heard": True})

    def test_context_bounds_aware_time_and_expiry(self):
        invalid = ({"text": "x" * 4097}, {"created_at": NOW.replace(tzinfo=None)},
                   {"expires_at": NOW}, {"dependencies": (dependency(),) * 65})
        for changes in invalid:
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValidationError):
                context_update(**changes)

    def test_start_and_offer_identity_and_time_constraints(self):
        request = start_request().model_dump()
        request["initial_context"] = context_update(route_identity(4)).model_dump()
        with self.assertRaises(ValidationError):
            RouteStart.model_validate(request)
        with self.assertRaises(ValidationError):
            NotificationOffer.model_validate({**notification_offer().model_dump(), "expires_at": NOW})
        with self.assertRaises(ValidationError):
            route_identity(0)

    def test_capabilities_declare_every_operation_once(self):
        manifest = capabilities()
        self.assertEqual(len(manifest.operations), len(RouteOperation))
        values = manifest.model_dump()
        values["operations"] = manifest.operations[:-1] + (manifest.operations[0],)
        with self.assertRaises(ValidationError):
            RouteCapabilities.model_validate(values)
        with self.assertRaises(ValidationError):
            OperationCapability(operation=RouteOperation.RETIRE_CONTEXT, support=Support.UNSUPPORTED)
        with self.assertRaises(ValidationError):
            RouteCapabilities.model_validate({**manifest.model_dump(), "transport": "automatic_fallback"})

    def test_attributable_playback_rejects_missing_or_guessed_correlation(self):
        invalid = ({"attempt": None, "offer_id": None}, {"offer_id": None},
                   {"output": None}, {"playback": None}, {"attribution": "unattributed"})
        for changes in invalid:
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValidationError):
                playback_evidence(**changes)

    def test_native_evidence_cannot_invent_pcm_or_human_hearing(self):
        for changes in ({"played_samples": 24000}, {"state": "played"}, {"human_heard": True}):
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValidationError):
                DeliveryEvidence.model_validate({**playback_evidence().model_dump(), **changes})
        with self.assertRaises(ValidationError):
            ClientPlaybackObservation(receipt_id=UUID(int=51), observed_at=NOW,
                                      boundary="human_heard", rendered_ms=25)
        with self.assertRaises(ValidationError):
            ClientPlaybackObservation(receipt_id=UUID(int=51), observed_at=NOW, rendered_ms=0)

    def test_output_completion_and_unknown_are_not_playback(self):
        for state in (DeliveryEvidenceState.OUTPUT_FINISHED, DeliveryEvidenceState.INTERRUPTED,
                      DeliveryEvidenceState.FAILED, DeliveryEvidenceState.UNKNOWN):
            with self.subTest(state=state):
                event = playback_evidence(state=state, playback=None)
                self.assertIsNone(event.playback)
        with self.assertRaises(ValidationError):
            playback_evidence(state=DeliveryEvidenceState.OUTPUT_FINISHED)
        event = playback_evidence(output=OutputIdentity(output_id=UUID(int=40)))
        self.assertIsNone(event.output.provider_response_ref)

    def test_context_acknowledgment_is_not_notification_evidence(self):
        update = context_update()
        receipt = ContextReceipt(route=update.route, update_id=update.update_id,
            cursor=update.cursor, observed_at=NOW, status="provider_acknowledged")
        event = TypeAdapter(VoiceRouteEvent).validate_json(receipt.model_dump_json())
        self.assertIsInstance(event, ContextReceipt)
        self.assertNotIsInstance(event, DeliveryEvidence)
        with self.assertRaises(ValidationError):
            DeliveryEvidence.model_validate(receipt.model_dump())

    def test_retirement_ack_does_not_assert_completed_handover(self):
        receipt = SourceRetirementReceipt(route=route_identity(), command_id=UUID(int=70),
            cursor=context_update().cursor, observed_at=NOW, status="provider_acknowledged")
        self.assertNotIn("output_fenced", receipt.model_dump())
        with self.assertRaises(ValidationError):
            SourceRetirementReceipt.model_validate({**receipt.model_dump(), "status": "completed"})
        with self.assertRaises(ValidationError):
            SourceRetirementCommand(route=route_identity(), command_id=UUID(int=70),
                requested_at=NOW, cursor=context_update().cursor, sources=(), reason="withdrawn")

    def test_completed_interruption_and_close_require_output_fencing(self):
        for operation in ("interrupt_speech", "close"):
            for fenced in (None, False):
                with self.subTest(operation=operation, fenced=fenced), self.assertRaises(ValidationError):
                    CommandReceipt(route=route_identity(), command_id=UUID(int=71),
                        operation=operation, observed_at=NOW, status="completed",
                        output_fenced=fenced, provider_stopped=True)
            receipt = CommandReceipt(route=route_identity(), command_id=UUID(int=71),
                operation=operation, observed_at=NOW, status="unknown", output_fenced=False)
            self.assertEqual(receipt.status, "unknown")


class VoiceRouteAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_interface_and_explicit_unsupported_operation(self):
        route: VoiceRoute = FakeVoiceRoute()
        await route.start(start_request())
        await route.publish_context(context_update())
        offered = await route.offer_notification(notification_offer())
        self.assertEqual(offered.state, DeliveryEvidenceState.OFFERED)
        unsupported = await route.retire_context(SourceRetirementCommand(
            route=route.identity, command_id=UUID(int=70), requested_at=NOW,
            cursor=context_update().cursor, sources=(dependency(),), reason="source withdrawn"))
        self.assertIsInstance(unsupported, UnsupportedOperation)
        self.assertEqual(unsupported.operation, RouteOperation.RETIRE_CONTEXT)
        self.assertIsInstance(TypeAdapter(VoiceRouteEvent).validate_json(unsupported.model_dump_json()),
                              UnsupportedOperation)
        stopped = await route.interrupt_speech(InterruptSpeechCommand(route=route.identity,
            command_id=UUID(int=71), requested_at=NOW, reason="user barge-in"))
        self.assertTrue(stopped.output_fenced)
        self.assertIsNone(stopped.provider_stopped)
        await route.close(CloseRouteCommand(route=route.identity, command_id=UUID(int=72),
                                          requested_at=NOW, reason="route replacement"))
        events = [event async for event in route.events()]
        self.assertEqual([event.kind for event in events], ["command_receipt", "context_receipt",
            "delivery_evidence", "unsupported_operation", "command_receipt", "command_receipt"])
        self.assertEqual(sum(isinstance(event, DeliveryEvidence) for event in events), 1)

    async def test_actual_offer_attempt_generation_and_output_binding_are_required(self):
        route = FakeVoiceRoute()
        await route.start(start_request())
        await route.offer_notification(notification_offer())
        event = playback_evidence()
        with self.assertRaises(ValueError):
            route.accept_delivery(event)
        route.bind_observed_output(OFFER_ID, output_identity())
        route.accept_delivery(event)
        invalid = ({"route": route_identity(4)}, {"attempt": attempt_reference(8)},
                   {"offer_id": UUID(int=999)}, {"output": OutputIdentity(output_id=UUID(int=999))},
                   {"transport": VoiceTransport.LEGACY_PCM_TTS})
        for changes in invalid:
            with self.subTest(changes=tuple(changes)), self.assertRaises(ValueError):
                route.accept_delivery(playback_evidence(**changes))
        self.assertEqual(event.route.route_generation, 3)
        self.assertEqual(event.attempt.attempt_generation, 7)
        self.assertEqual(event.attempt.source.revision, 11)
        self.assertEqual(event.attempt.source.generation, 5)

    async def test_unconfirmed_fence_does_not_complete_handover(self):
        route = FakeVoiceRoute(fence_confirmed=False)
        await route.start(start_request())
        receipt = await route.close(CloseRouteCommand(route=route.identity,
            command_id=UUID(int=72), requested_at=NOW, reason="connection uncertain"))
        self.assertEqual(receipt.status, "unknown")
        self.assertFalse(receipt.output_fenced)

    async def test_retired_route_rejects_late_delivery_and_expired_input(self):
        route = FakeVoiceRoute()
        await route.start(start_request())
        expired = context_update(created_at=NOW - timedelta(seconds=2),
                                 expires_at=NOW - timedelta(seconds=1))
        with self.assertRaises(ValueError):
            await route.publish_context(expired)
        await route.offer_notification(notification_offer())
        route.bind_observed_output(OFFER_ID, output_identity())
        await route.close(CloseRouteCommand(route=route.identity,
            command_id=UUID(int=72), requested_at=NOW, reason="route replacement"))
        with self.assertRaises(ValueError):
            route.accept_delivery(playback_evidence())


if __name__ == "__main__":
    unittest.main()
