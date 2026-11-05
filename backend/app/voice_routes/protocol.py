"""Candidate whole-route interface, pending P0 fixtures and integration decisions."""

from collections.abc import AsyncIterator
from typing import Protocol

from app.voice_routes.contracts import (
    CloseRouteCommand,
    CommandReceipt,
    ContextReceipt,
    DeliveryEvidence,
    InterruptSpeechCommand,
    NotificationOffer,
    RouteCapabilities,
    RouteIdentity,
    RouteStart,
    SourceRetirementCommand,
    SourceRetirementReceipt,
    UnsupportedOperation,
    VoiceContextUpdate,
    VoiceRouteEvent,
)


class VoiceRoute(Protocol):
    """One route generation owns conversational speech, not application policy.

Implementations reject mismatched identities and expired input, honor declared
capabilities, and report unsupported operations explicitly without fallback.
The caller owns authorization, source validity, admission, eligibility, claims,
ledger mutations, replay policy and exclusive handover to another route.

Provider/transport evidence must be tied to the original generation. No method
completion proves human hearing. Local interruption must not wait behind context
writes, and closing must not cancel durable work. These are candidate obligations,
not implemented scheduling, transport or qualification guarantees.
"""

    @property
    def identity(self) -> RouteIdentity: ...

    @property
    def capabilities(self) -> RouteCapabilities: ...

    async def start(self, request: RouteStart) -> CommandReceipt | UnsupportedOperation: ...

    async def publish_context(
        self, update: VoiceContextUpdate,
    ) -> ContextReceipt | UnsupportedOperation: ...

    async def offer_notification(
        self, offer: NotificationOffer,
    ) -> DeliveryEvidence | UnsupportedOperation:
        """Report an offer boundary; this neither claims nor settles its intent."""
        ...

    async def retire_context(
        self, command: SourceRetirementCommand,
    ) -> SourceRetirementReceipt | UnsupportedOperation: ...

    async def interrupt_speech(
        self, command: InterruptSpeechCommand,
    ) -> CommandReceipt | UnsupportedOperation: ...

    async def close(
        self, command: CloseRouteCommand,
    ) -> CommandReceipt | UnsupportedOperation: ...

    def events(self) -> AsyncIterator[VoiceRouteEvent]:
        """Stream typed observations; returned evidence is not an exactly-once bus."""
        ...
