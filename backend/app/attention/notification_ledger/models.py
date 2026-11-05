"""Immutable persisted native plans and conservative v1 settlement rules."""
from dataclasses import dataclass
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from app.attention.models import ACTIVE_PHASES, AttentionError, DeliveryClaim
from app.live_state.models import ContractModel
from app.voice_routes.contracts import (
    DeliveryEvidenceState, NotificationOffer, RouteCapabilities, RouteOperation,
    Support, VoiceTransport,
)


PLAN_FORMAT = "jarvis.attention.voice_route_plan.v1"
SCHEMA_VERSION = 39


class VoiceRoutePlan(ContractModel):
    format: Literal["jarvis.attention.voice_route_plan.v1"] = PLAN_FORMAT
    protocol_version: Literal[1] = 1
    audio_session_id: UUID
    offer: NotificationOffer
    capabilities: RouteCapabilities
    submission_boundary: str = Field(min_length=1, max_length=256)
    source_payload_root_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def qualified_new_mode(self) -> Self:
        if self.capabilities.transport == VoiceTransport.LEGACY_PCM_TTS:
            raise ValueError("Legacy PCM retains its existing speech plan and ledger path")
        operations = {value.operation: value.support for value in self.capabilities.operations}
        if operations[RouteOperation.OFFER_NOTIFICATION] != Support.SUPPORTED:
            raise ValueError("The selected route cannot accept notification offers")
        if not self.submission_boundary.strip() or "\x00" in self.submission_boundary:
            raise ValueError("A concrete speech-eligible external write boundary is required")
        if self.source_payload_root_ids != (self.offer.attempt.source_payload_id,):
            raise ValueError("The plan must retain its exact canonical attention source root")
        return self


@dataclass(frozen=True, slots=True)
class Settlement:
    attempt_phase: str
    intent_state: str
    reason: str


def settlement_for(phase: str, *, submitted: bool, state: DeliveryEvidenceState,
                   reason: str | None = None) -> Settlement | None:
    """Lower-bound playback and provider output can never establish `played`."""
    if phase not in ACTIVE_PHASES:
        return None
    if state == DeliveryEvidenceState.OUTPUT_FINISHED:
        if not submitted:
            raise AttentionError("attention_submission_state", "Output requires prior submission.")
        return Settlement("unknown", "unknown", "native_completion_unconfirmed")
    if state == DeliveryEvidenceState.INTERRUPTED:
        return Settlement("interrupted" if submitted else "failed",
                          "unknown" if submitted else "failed", reason or "native_interrupted")
    if state in {DeliveryEvidenceState.FAILED, DeliveryEvidenceState.UNKNOWN}:
        terminal = "unknown" if submitted else "failed"
        return Settlement(terminal, terminal, reason or "native_delivery_unconfirmed")
    return None


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    inserted: bool
    attempt_phase: str
    hearing: Literal["not_established"] = "not_established"


@dataclass(slots=True)
class SubmissionTicket:
    """Process-local one-use permit issued only after a durable submission commit.

    Losing this ticket leaves an uncertain attempt; never reconstruct it to
    retry a possibly exposed offer after a crash.
    """
    claim: DeliveryClaim
    offer: NotificationOffer
    _issuer: object
    _used: bool = False

