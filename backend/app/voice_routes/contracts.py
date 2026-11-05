"""Candidate P0 wire contracts; admission, ledger writes and qualification live elsewhere.

Character bounds are not provider token budgets. Adapters must enforce their own
limits and validate route/command correlation against the active connection.
"""

from collections.abc import Callable
from enum import Enum
from typing import Annotated, Literal, Self, TypeAlias
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator

from app.live_state.models import ContractModel, GuardedPayloadBinding, SessionScope, SourceDependency, StateCursor


BoundedText: TypeAlias = Annotated[str, Field(min_length=1, max_length=4096)]
ReasonText: TypeAlias = Annotated[str, Field(min_length=1, max_length=512)]
ReferenceText: TypeAlias = Annotated[str, Field(min_length=1, max_length=256)]


class VoiceTransport(str, Enum):
    LEGACY_PCM_TTS = "legacy_pcm_tts"
    NATIVE_VOICE = "native_voice"
    CHAINED_VOICE = "chained_voice"


class Support(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


class RouteOperation(str, Enum):
    START = "start"
    PUBLISH_CONTEXT = "publish_context"
    OFFER_NOTIFICATION = "offer_notification"
    RETIRE_CONTEXT = "retire_context"
    INTERRUPT_SPEECH = "interrupt_speech"
    CLOSE = "close"


class RouteIdentity(ContractModel):
    scope: SessionScope
    route_id: UUID
    route_generation: int = Field(ge=1)


class OperationCapability(ContractModel):
    operation: RouteOperation
    support: Support
    reason: ReasonText | None = None

    @model_validator(mode="after")
    def explain_unsupported(self) -> Self:
        if self.support == Support.UNSUPPORTED and self.reason is None:
            raise ValueError("Unsupported operations require a reason.")
        return self


class RouteCapabilities(ContractModel):
    """Explicit adapter declarations, not evidence of provider qualification."""

    transport: VoiceTransport
    media_topology: Literal["phone_to_provider", "phone_to_backend"]
    control_owner: Literal["backend", "authenticated_client_bridge"]
    credential_mode: Literal["server_only", "server_issued_client_token"]
    input_codec: ReferenceText
    output_codec: ReferenceText
    vad_owner: Literal["phone", "provider", "route", "none"]
    explicit_input_commit: Support
    explicit_response_create: Support
    full_duplex: Support
    image_input: Support
    provider_context_receipts: Support
    notification_output_attribution: Support
    client_playback_evidence: Support
    provider_interruption: Support
    operations: tuple[OperationCapability, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def declare_every_operation_once(self) -> Self:
        names = tuple(item.operation for item in self.operations)
        if len(set(names)) != len(names) or set(names) != set(RouteOperation):
            raise ValueError("Declare each route operation exactly once.")
        return self


class ObservationWindow(ContractModel):
    """Ordinary evidence freshness, never source authorization or retention.

    The application may retain an authorized observation as timestamped history
    after this window. Canonical correction/deletion and ownership guards still
    govern every remote copy, including copies no longer in the current view.
    """

    observed_at: AwareDatetime
    fresh_until: AwareDatetime
    dependencies: tuple[SourceDependency, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bounded_evidence_window(self) -> Self:
        if self.fresh_until <= self.observed_at:
            raise ValueError("Observation freshness must follow capture.")
        keys = [(source.source_namespace, source.source_id) for source in self.dependencies]
        if len(keys) != len(set(keys)):
            raise ValueError("Observation sources must have one exact version each.")
        return self


class VoiceContextUpdate(ContractModel):
    """Untrusted factual context, never an instruction or permission to speak."""

    route: RouteIdentity
    update_id: UUID
    cursor: StateCursor
    mode: Literal["snapshot", "delta"]
    text: BoundedText
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=64)
    guard_binding: GuardedPayloadBinding | None = None
    created_at: AwareDatetime
    # Hard remote-context lifetime. Ordinary visual freshness is NOT a remote
    # erasure requirement when the explicitly opted-in history contract is used.
    expires_at: AwareDatetime | None = None
    observation_history: tuple[ObservationWindow, ...] = Field(default=(), max_length=16)

    @model_validator(mode="after")
    def ordered_expiry(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("Context expiry must follow creation.")
        if self.dependencies and self.guard_binding is None:
            raise ValueError("Source-bearing voice context requires committed guarded payload roots.")
        if self.guard_binding is not None:
            if self.guard_binding.scope != self.route.scope:
                raise ValueError("Voice context guard scope must match the selected route.")
            if set(self.guard_binding.dependencies) != set(self.dependencies):
                raise ValueError("Voice context dependencies must match their guarded binding.")
        if len(set(self.observation_history)) != len(self.observation_history):
            raise ValueError("Observation history windows must be unique.")
        for observation in self.observation_history:
            if observation.observed_at > self.created_at:
                raise ValueError("Context cannot claim a future observation.")
            if not set(observation.dependencies) <= set(self.dependencies):
                raise ValueError("Historical observations require exact guarded context sources.")
        return self


class HistoryTurn(ContractModel):
    role: Literal["user", "assistant"]
    text: BoundedText
    dependencies: tuple[SourceDependency, ...] = Field(default=(), max_length=64)
    guard_binding: GuardedPayloadBinding | None = None

    @model_validator(mode="after")
    def require_guarded_history(self) -> Self:
        if self.dependencies and self.guard_binding is None:
            raise ValueError("Source-bearing history requires committed guarded payload roots.")
        if self.guard_binding is not None and set(self.guard_binding.dependencies) != set(self.dependencies):
            raise ValueError("History dependencies must match their guarded binding.")
        return self


class RouteStart(ContractModel):
    """Authorization is an external opaque reference, never a provider secret."""

    route: RouteIdentity
    command_id: UUID
    requested_at: AwareDatetime
    authorization_ref: UUID
    initial_context: VoiceContextUpdate
    publish_initial_context: bool = True
    relevant_history: tuple[HistoryTurn, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def same_initial_route(self) -> Self:
        if self.initial_context.route != self.route:
            raise ValueError("Initial context must belong to the starting route.")
        if any(turn.guard_binding is not None and turn.guard_binding.scope != self.route.scope
               for turn in self.relevant_history):
            raise ValueError("Historical payload bindings must belong to the starting route scope.")
        return self


class ContextWriteEvidence:
    """Local, one-way publication bookkeeping, not a provider receipt itself.

    Uninstrumented routes remain unknown. A cancelled factory cannot reopen its
    attempt; no-write proof additionally requires all owned work to be joined.
    A matching observed delete ACK can prove a resolved partial mutation, not
    whole-attempt no-write. This never replaces source/write authorization.
    """

    def __init__(self, *, write_budget: Callable[[], bool] | None = None):
        # Optional synchronous veto at the existing writer factory. This grants
        # time only, never source/route authority or acknowledgment of an effect.
        if write_budget is not None and not callable(write_budget):
            raise TypeError("A publication write budget must be callable.")
        self._write_budget = write_budget
        self._tracked = self._closed = self._joined = self._possible = False
        self.first_operation = None
        self._delete_attempts = 0
        self._delete_item_id = self._deleted_item_id = None
        self._create_possible = False
        self._create_attempts = 0
        self._create_item_id = self._created_item_id = None

    def begin_tracking(self):
        self._tracked = True

    def close(self):
        self._closed = True
        self._write_budget = None

    def mark_joined(self):
        self._joined = True

    def before_write(self, operation: Literal["context_create", "context_delete"], *,
                     item_id: str | None = None) -> bool:
        if operation not in {"context_create", "context_delete"}:
            raise ValueError("Unsupported context mutation boundary")
        if self._closed:
            return False
        if self._write_budget is not None and self._write_budget() is not True:
            self.close()
            return False
        self._possible = True
        if self.first_operation is None:
            self.first_operation = operation
        if operation == "context_create":
            self._create_possible = True
            self._create_attempts += 1
            self._create_item_id = item_id if isinstance(item_id, str) and 0 < len(item_id) <= 160 else None
            self._created_item_id = None
        else:
            self._delete_attempts += 1
            self._delete_item_id = item_id if isinstance(item_id, str) and 0 < len(item_id) <= 160 else None
            self._deleted_item_id = None
        return True

    def record_delete_acknowledged(self, item_id: str) -> bool:
        """Called only after the owned provider item's matching deleted ACK."""
        if (not self._tracked or self._closed or self._delete_attempts != 1
                or self._create_possible or self._delete_item_id is None
                or item_id != self._delete_item_id):
            return False
        self._deleted_item_id = item_id
        return True

    @property
    def deleted_context_acknowledged(self) -> bool:
        return (self._delete_attempts == 1 and self._delete_item_id is not None
                and self._deleted_item_id == self._delete_item_id)

    def record_create_acknowledged(self, item_id: str) -> bool:
        """Exact owned final ACK only; never a send return or an `added` event."""
        if (not self._tracked or self._closed or self._create_attempts != 1
                or self._create_item_id is None or item_id != self._create_item_id):
            return False
        self._created_item_id = item_id
        return True

    @property
    def created_context_acknowledged(self) -> bool:
        return (self._create_attempts == 1 and self._create_item_id is not None
                and self._created_item_id == self._create_item_id)

    @property
    def confirmed_created(self) -> bool:
        # Mutation knowledge, NOT current content, source authority or readiness.
        return (self._tracked and self._closed and self._joined
                and self.created_context_acknowledged
                and (self._delete_attempts == 0 or self.deleted_context_acknowledged))

    @property
    def replacement_create_possible(self) -> bool:
        return self._create_possible

    @property
    def confirmed_deleted_without_create(self) -> bool:
        # This is a resolved partial mutation, never whole-attempt no-write.
        return (self._tracked and self._closed and self._joined
                and self.deleted_context_acknowledged and not self._create_possible)

    @property
    def work_joined(self) -> bool:
        return self._joined

    @property
    def state(self) -> Literal["unknown", "prewrite", "unwritten", "write_possible"]:
        if self._possible:
            return "write_possible"
        if not self._tracked:
            return "unknown"
        return "unwritten" if self._closed and self._joined else "prewrite"

    @property
    def proven_unwritten(self) -> bool:
        return self.state == "unwritten"


class ContextReceipt(ContractModel):
    """Provider acknowledgment is not comprehension, speech or playback evidence."""

    kind: Literal["context_receipt"] = "context_receipt"
    route: RouteIdentity
    update_id: UUID
    cursor: StateCursor
    observed_at: AwareDatetime
    status: Literal["queued", "written", "provider_acknowledged", "rejected", "failed", "unknown"]
    provider_receipt_ref: ReferenceText | None = None
    reason: ReasonText | None = None


class DeliveryAttemptReference(ContractModel):
    """Existing canonical identities; attempt generation is not route generation."""

    notification_intent_id: UUID
    attempt_id: UUID
    attempt_generation: int = Field(ge=1)
    source: SourceDependency
    source_payload_id: UUID


class NotificationOffer(ContractModel):
    """Caller-authorized offer; the route neither claims nor reauthorizes an intent."""

    route: RouteIdentity
    offer_id: UUID
    attempt: DeliveryAttemptReference
    text: BoundedText
    offered_at: AwareDatetime
    expires_at: AwareDatetime
    context_cursor: StateCursor | None = None

    @model_validator(mode="after")
    def ordered_expiry(self) -> Self:
        if self.expires_at <= self.offered_at:
            raise ValueError("Offer expiry must follow its creation.")
        return self


class OutputIdentity(ContractModel):
    """Identity assigned to actual observed output, not an inferred notification.

Provider references are copied only when available. Local output identity also
supports chained/legacy output without mislabeling synthetic IDs as provider IDs.
"""

    output_id: UUID
    provider_response_ref: ReferenceText | None = None
    provider_item_ref: ReferenceText | None = None


class ClientPlaybackObservation(ContractModel):
    """Attributable client-render evidence only; no human-heard success or PCM claim."""

    receipt_id: UUID
    observed_at: AwareDatetime
    boundary: Literal["client_render_lower_bound"] = "client_render_lower_bound"
    rendered_ms: int | None = Field(default=None, gt=0)


class DeliveryEvidenceState(str, Enum):
    OFFERED = "offered"
    PROVIDER_RECEIVED = "provider_received"
    SUBMITTED = "submitted"
    OUTPUT_STARTED = "output_started"
    OUTPUT_FINISHED = "output_finished"
    CLIENT_PLAYBACK_OBSERVED = "client_playback_observed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DeliveryEvidence(ContractModel):
    """Observations for an external ledger owner, never ledger transitions.

Submission is the adapter's declared speech-offer boundary, not generic context
receipt. Combined output needs one explicitly correlated record per attempt.
Unattributed output may be reported without an attempt; timestamps are not proof.
"""

    kind: Literal["delivery_evidence"] = "delivery_evidence"
    evidence_id: UUID
    route: RouteIdentity
    transport: VoiceTransport
    observed_at: AwareDatetime
    state: DeliveryEvidenceState
    attribution: Literal["explicit", "unattributed"] = "unattributed"
    offer_id: UUID | None = None
    attempt: DeliveryAttemptReference | None = None
    output: OutputIdentity | None = None
    playback: ClientPlaybackObservation | None = None
    reason: ReasonText | None = None

    @model_validator(mode="after")
    def require_evidence_correlation(self) -> Self:
        if (self.attempt is None) != (self.offer_id is None):
            raise ValueError("Attempt and offer identity must be supplied together.")
        if self.attribution == "explicit" and self.attempt is None:
            raise ValueError("Explicit notification attribution requires an attempt.")
        if self.attempt is not None and self.attribution != "explicit":
            raise ValueError("Do not attach a guessed attempt to unattributed evidence.")
        if self.state in {
            DeliveryEvidenceState.OFFERED,
            DeliveryEvidenceState.PROVIDER_RECEIVED,
            DeliveryEvidenceState.SUBMITTED,
        } and self.attempt is None:
            raise ValueError("Offer/submission evidence requires the actual attempt.")
        if self.state in {
            DeliveryEvidenceState.OUTPUT_STARTED,
            DeliveryEvidenceState.OUTPUT_FINISHED,
            DeliveryEvidenceState.CLIENT_PLAYBACK_OBSERVED,
        } and self.output is None:
            raise ValueError("Output evidence requires an actual output identity.")
        if self.state == DeliveryEvidenceState.CLIENT_PLAYBACK_OBSERVED:
            if self.attempt is None or self.playback is None:
                raise ValueError("Attributable playback requires attempt, output and client receipt.")
        elif self.playback is not None:
            raise ValueError("Client playback observations require the playback evidence state.")
        return self


class SourceRetirementCommand(ContractModel):
    """Retire the named dependency versions/generations, not newer replacements."""

    route: RouteIdentity
    command_id: UUID
    requested_at: AwareDatetime
    cursor: StateCursor
    sources: tuple[SourceDependency, ...] = Field(min_length=1, max_length=64)
    reason: ReasonText


class SourceRetirementReceipt(ContractModel):
    """An append/control acknowledgment alone does not establish effective retirement.

Reconnect-required/unknown leaves retirement unresolved. Even route closure needs
separate confirmed output fencing before a replacement speaker is admitted.
"""

    kind: Literal["source_retirement_receipt"] = "source_retirement_receipt"
    route: RouteIdentity
    command_id: UUID
    cursor: StateCursor
    observed_at: AwareDatetime
    status: Literal[
        "accepted", "local_fenced", "provider_acknowledged", "reconnect_required",
        "route_closed", "failed", "unknown",
    ]
    provider_receipt_ref: ReferenceText | None = None
    reason: ReasonText | None = None


class InterruptSpeechCommand(ContractModel):
    """Stop speech only, never the durable task or an already-submitted action."""

    route: RouteIdentity
    command_id: UUID
    requested_at: AwareDatetime
    reason: ReasonText
    attempt: DeliveryAttemptReference | None = None
    output: OutputIdentity | None = None


class CloseRouteCommand(ContractModel):
    """Retire this route owner, not its durable conversation, tasks or notifications."""

    route: RouteIdentity
    command_id: UUID
    requested_at: AwareDatetime
    reason: ReasonText


class CommandReceipt(ContractModel):
    kind: Literal["command_receipt"] = "command_receipt"
    route: RouteIdentity
    command_id: UUID
    operation: Literal["start", "interrupt_speech", "close"]
    observed_at: AwareDatetime
    status: Literal["accepted", "completed", "failed", "unknown"]
    output_fenced: bool | None = None
    provider_stopped: bool | None = None
    reason: ReasonText | None = None

    @model_validator(mode="after")
    def completed_stop_requires_fence(self) -> Self:
        if (self.operation in {"interrupt_speech", "close"}
                and self.status == "completed" and self.output_fenced is not True):
            raise ValueError("Completed interruption/close requires confirmed output fencing.")
        return self


class UnsupportedOperation(ContractModel):
    """No silent downgrade, alternative speaker or fabricated acknowledgment."""

    kind: Literal["unsupported_operation"] = "unsupported_operation"
    route: RouteIdentity
    request_id: UUID
    operation: RouteOperation
    observed_at: AwareDatetime
    reason: ReasonText


class TranscriptRevision(ContractModel):
    kind: Literal["transcript_revision"] = "transcript_revision"
    route: RouteIdentity
    transcript_id: UUID
    revision: int = Field(ge=1)
    observed_at: AwareDatetime
    role: Literal["user", "assistant"]
    text: BoundedText
    final: bool
    output: OutputIdentity | None = None


class WorkProposal(ContractModel):
    """Data for the shared admission coordinator, never permission to execute."""

    kind: Literal["work_proposal"] = "work_proposal"
    route: RouteIdentity
    proposal_id: UUID
    transcript_id: UUID
    transcript_revision: int = Field(ge=1)
    cursor: StateCursor
    observed_at: AwareDatetime
    text: BoundedText
    provider_delegation_ref: ReferenceText | None = None


class RouteConnectionEvent(ContractModel):
    kind: Literal["connection_state"] = "connection_state"
    route: RouteIdentity
    observed_at: AwareDatetime
    state: Literal["starting", "ready", "closing", "closed", "failed"]
    reason: ReasonText | None = None


class RouteError(ContractModel):
    kind: Literal["route_error"] = "route_error"
    route: RouteIdentity
    observed_at: AwareDatetime
    code: ReferenceText
    message: ReasonText
    recoverable: bool


VoiceRouteEvent: TypeAlias = Annotated[
    ContextReceipt | DeliveryEvidence | SourceRetirementReceipt | CommandReceipt
    | UnsupportedOperation | TranscriptRevision | WorkProposal | RouteConnectionEvent
    | RouteError,
    Field(discriminator="kind"),
]
