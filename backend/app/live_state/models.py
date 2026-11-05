"""P0 candidate contracts, not a persistence engine or an authority to act.

Collections are tuples and nested records are frozen. Adapters must supply the
canonical lifecycle generation for sources that have one; content revision is
not a substitute. Reducer ordering and authorization belong to the next layer.
"""
from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator


Revision = Annotated[int, Field(ge=1, strict=True)]
Counter = Annotated[int, Field(ge=0, strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9_.:-]+$")]
ShortText = Annotated[str, Field(min_length=1, max_length=240)]
SummaryText = Annotated[str, Field(min_length=1, max_length=2000)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class SessionScope(ContractModel):
    owner_id: str = Field(min_length=1, max_length=200, strict=True)
    session_id: UUID
    area_id: UUID | None = None

    @field_validator("owner_id")
    @classmethod
    def canonical_owner_id(cls, value: str) -> str:
        # Match PostgreSQL's default btrim check; never normalize an identity.
        if value != value.strip(" ") or "\x00" in value:
            raise ValueError("Owner identity must match the canonical trimmed text domain.")
        return value


class StateCursor(ContractModel):
    epoch: UUID
    revision: Counter


class SourceDependency(ContractModel):
    source_namespace: Identifier
    source_id: UUID
    revision: Revision
    # None means the authority has no lifecycle counter, not "ignore its counter".
    generation: Revision | None = None


class GuardedPayloadBinding(ContractModel):
    """Committed context roots, not availability proof or a new payload revision.

    Only the canonical binding/guard path may construct this receipt. Consumers
    must still enforce current read/copy/output guards and retire remote context.
    """

    scope: SessionScope
    payload_root_ids: tuple[UUID, ...] = Field(min_length=1, max_length=64)
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def unique_roots_and_dependencies(self) -> GuardedPayloadBinding:
        if len(set(self.payload_root_ids)) != len(self.payload_root_ids):
            raise ValueError("Guarded roots must be unique.")
        keys = [(source.source_namespace, source.source_id) for source in self.dependencies]
        if len(set(keys)) != len(keys):
            raise ValueError("Guarded dependencies must have one version per source.")
        return self


class AggregateRef(ContractModel):
    namespace: Identifier
    aggregate_id: UUID
    revision: Revision


class ProducerCursor(ContractModel):
    producer_id: Identifier
    producer_epoch: UUID
    aggregate: AggregateRef
    source_generation: Revision | None = None


class SourceWatermark(ContractModel):
    source: SourceDependency
    availability: Literal["active", "retracted", "unknown"]
    recorded_at: AwareDatetime


class LiveCapabilities(ContractModel):
    durable_responsibilities: Literal["available", "disabled", "unavailable"] = "unavailable"
    historical_retrieval: Literal["available", "disabled", "unavailable"] = "unavailable"


class FrameRef(ContractModel):
    frame_id: UUID
    stream_epoch: UUID
    captured_at: AwareDatetime


class EntryBase(ContractModel):
    entry_id: Identifier
    scope: SessionScope
    received_at: AwareDatetime
    observed_at: AwareDatetime | None = None
    fresh_until: AwareDatetime | None = None
    expires_at: AwareDatetime | None = None
    sources: tuple[SourceDependency, ...] = Field(default=(), max_length=16)
    guard_binding: GuardedPayloadBinding | None = None
    evidence_kind: Literal["observed", "user_reported", "inferred", "canonical_record"]
    validity: Literal["active", "superseded", "retracted"] = "active"

    @model_validator(mode="after")
    def coherent_provenance(self) -> EntryBase:
        keys = [(source.source_namespace, source.source_id) for source in self.sources]
        if len(keys) != len(set(keys)):
            raise ValueError("An entry cannot contain duplicate or conflicting source dependencies.")
        if self.guard_binding is not None:
            if self.guard_binding.scope != self.scope:
                raise ValueError("Guarded payload roots must belong to the projected scope.")
            if set(self.guard_binding.dependencies) != set(self.sources):
                raise ValueError("Guarded payload dependencies must match the entry sources exactly.")
        if (self.fresh_until is not None and self.expires_at is not None
                and self.fresh_until > self.expires_at):
            raise ValueError("Evidence cannot remain fresh beyond entry expiry.")
        return self


class ObservationEntry(EntryBase):
    kind: Literal["observation"] = "observation"
    summary: SummaryText
    frames: tuple[FrameRef, ...] = Field(default=(), max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    unknowns: tuple[ShortText, ...] = Field(default=(), max_length=8)


class SelectedMemoryEntry(EntryBase):
    kind: Literal["selected_memory"] = "selected_memory"
    summary: SummaryText
    sources: tuple[SourceDependency, ...] = Field(min_length=1, max_length=16)


class FocusEntry(EntryBase):
    kind: Literal["focus"] = "focus"
    focus: AggregateRef
    goal: SummaryText
    stage: ShortText | None = None


class TaskEntry(EntryBase):
    kind: Literal["task"] = "task"
    task: AggregateRef
    status: Identifier
    summary: SummaryText


class NotificationEntry(EntryBase):
    kind: Literal["notification"] = "notification"
    intent_id: UUID
    intent_generation: Revision
    status: Identifier
    summary: SummaryText


class CheckEntry(EntryBase):
    """Compact projection of an owner's check, not a lifecycle command."""

    kind: Literal["situational_check"] = "situational_check"
    check_id: UUID
    check_revision: Revision
    lifecycle: Literal["candidate", "tracking", "resolved", "dismissed", "expired"]
    question: SummaryText
    canonical_owner: AggregateRef | None = None
    durable: bool = False

    @model_validator(mode="after")
    def durable_owner_required(self) -> CheckEntry:
        if self.durable and self.canonical_owner is None:
            raise ValueError("Durable checks require a canonical owner.")
        return self


class FutureCueEntry(EntryBase):
    kind: Literal["future_cue"] = "future_cue"
    check_id: UUID
    relevance_cue: SummaryText


LiveEntry = Annotated[
    ObservationEntry | SelectedMemoryEntry | FocusEntry | TaskEntry
    | NotificationEntry | CheckEntry | FutureCueEntry,
    Field(discriminator="kind"),
]


class ConversationView(ContractModel):
    turn_id: UUID | None = None
    turn_revision: Counter = 0
    topic: ShortText | None = None
    focus: AggregateRef | None = None


class LiveSnapshot(ContractModel):
    schema_version: Literal[1] = 1
    scope: SessionScope
    cursor: StateCursor
    as_of: AwareDatetime
    readiness: Literal["warming", "ready", "degraded"]
    conversation: ConversationView = Field(default_factory=ConversationView)
    capabilities: LiveCapabilities = Field(default_factory=LiveCapabilities)
    entries: tuple[LiveEntry, ...] = Field(default=(), max_length=128)
    producer_cursors: tuple[ProducerCursor, ...] = Field(default=(), max_length=256)

    @model_validator(mode="after")
    def unique_scoped_entries(self) -> LiveSnapshot:
        identities = [entry.entry_id for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("Snapshot entries must have unique identities.")
        if any(entry.scope != self.scope for entry in self.entries):
            raise ValueError("Entries must be projected into the snapshot's explicit scope.")
        return self


class EventBase(ContractModel):
    event_id: UUID
    scope: SessionScope
    producer: ProducerCursor
    emitted_at: AwareDatetime


class EntryUpserted(EventBase):
    kind: Literal["entry_upserted"] = "entry_upserted"
    entry: LiveEntry

    @model_validator(mode="after")
    def same_scope(self) -> EntryUpserted:
        if self.entry.scope != self.scope:
            raise ValueError("An event cannot insert an entry from another scope.")
        return self


class EntryRetracted(EventBase):
    kind: Literal["entry_retracted"] = "entry_retracted"
    entry_id: Identifier
    reason: Literal["superseded", "source_withdrawn", "cancelled", "expired"]


class SourceStateChanged(EventBase):
    kind: Literal["source_state_changed"] = "source_state_changed"
    watermark: SourceWatermark


class TurnRevised(EventBase):
    kind: Literal["turn_revised"] = "turn_revised"
    conversation: ConversationView
    request_source: SourceDependency


class CapabilitiesChanged(EventBase):
    kind: Literal["capabilities_changed"] = "capabilities_changed"
    capabilities: LiveCapabilities


LiveEvent = Annotated[
    EntryUpserted | EntryRetracted | SourceStateChanged | TurnRevised | CapabilitiesChanged,
    Field(discriminator="kind"),
]


class RequestIdentity(ContractModel):
    """Canonical admission inputs, never an independently generated task ID."""

    scope: SessionScope
    request_source: SourceDependency
    operation: Identifier
    slot: Identifier
