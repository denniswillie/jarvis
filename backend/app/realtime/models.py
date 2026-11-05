from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import re
from typing import Any, Literal
from uuid import UUID

from app.sessions.models import JarvisSession
from app.realtime.output_policy import RealtimeOutputItemState
from app.tasks.context_events import ContextRecord
from app.tools.provenance import SourceBoundToolResult


AudioClientKind = Literal["rayban_audio", "phone_audio"]
ResponseKind = Literal[
    "normal_user_turn",
    "visual_interrupt",
    "reminder",
    "prospective",
]


@dataclass(frozen=True)
class RealtimeSessionContext:
    audio_session_id: str
    owner_id: str
    jarvis_session_id: UUID
    client_kind: AudioClientKind
    session: JarvisSession


def _followup_token(value: str, maximum: int = 160) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1," + str(maximum) + r"}", value) is not None


TOOL_FOLLOWUP_USER_CONTENT_MAX_ENCODED_BYTES = 1024


def _followup_user_content(value):
    """Retain a whole transient quotation, never a meaning-changing prefix."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        return None, "unavailable"
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return None, "unavailable"
    # Count the actual ASCII-escaped JSON representation used by the renderer.
    if len(json.dumps(value, ensure_ascii=True).encode("utf-8")) > TOOL_FOLLOWUP_USER_CONTENT_MAX_ENCODED_BYTES:
        return None, "omitted_byte_budget"
    return value, "available"


_FOLLOWUP_OUTCOME_VALUES = {
    "memory_status": frozenset({"stored", "updated", "corrected", "deleted", "unchanged", "unavailable"}),
    "source_ingestion_status": frozenset({"pending", "indexed", "retained", "failed", "cancelled", "excluded", "blocked"}),
    "quiet_effective_state": frozenset({"inactive", "quiet", "resumed", "expired", "unavailable"}),
    "quiet_delivery_propagation": frozenset({"checked", "unconfirmed"}),
    "error_code": frozenset({"tool_not_found", "tool_execution_failed", "tool_execution_persistence_failed",
        "tool_receipt_unavailable_after_commit", "tool_result_source_unavailable", "tool_execution_in_progress",
        "task_provider_not_configured", "tool_unavailable_for_response", "request_cancelled", "invalid_tool_arguments",
        "revision_conflict", "request_evidence_required", "request_evidence_unavailable", "request_conflict",
        "quiet_receipt_unavailable", "quiet_expiry_not_future", "attention_quiet_unavailable", "invalid_tool_input"}),
}


def _followup_outcome_valid(key, value):
    if key in {"replayed", "quiet_suppresses_unsolicited", "quiet_playback_stop_confirmed"}:
        return type(value) is bool
    if key in {"quiet_revision", "quiet_accepted_revision"}:
        return type(value) is int and 0 <= value < 2**63
    return isinstance(value, str) and value in _FOLLOWUP_OUTCOME_VALUES.get(key, ())


def _followup_owner_outcome(result, tool_name):
    """Selected owner states only; no result prose, inferred effect or new read."""
    fields = []
    if tool_name in {"get_attention_quiet", "set_attention_quiet"}:
        # These normal owner tools return repository dictionaries, not bound
        # reader wrappers. The continuation's canonical user request is still
        # guarded. Preserve the receipt's reported state, without inventing a
        # canonical result ID, doing a new read, or claiming it remains current.
        quiet = result.get("quiet")
        if (result.get("status") == "success" and isinstance(quiet, dict)
                and quiet.get("scope") == "owner_unsolicited_attention"):
            selected = [("quiet_effective_state", quiet.get("effective_state")),
                        ("quiet_revision", quiet.get("revision")),
                        ("quiet_suppresses_unsolicited", quiet.get("suppresses_unsolicited"))]
            if tool_name == "set_attention_quiet":
                propagation = result.get("delivery_propagation")
                propagation = propagation if isinstance(propagation, dict) else {}
                selected += [("quiet_accepted_revision", result.get("accepted_revision")),
                             ("replayed", result.get("request_replayed")),
                             ("quiet_delivery_propagation", propagation.get("state")),
                             ("quiet_playback_stop_confirmed", propagation.get("playback_stop_confirmed"))]
            fields.extend((key, value) for key, value in selected if _followup_outcome_valid(key, value))
    elif isinstance(result, SourceBoundToolResult):
        source = result.get("source")
        for key, value in (
            ("memory_status", result.get("memory_status")),
            ("source_ingestion_status", source.get("ingestion_status") if isinstance(source, dict) else None),
            ("replayed", result.get("replayed")),
        ):
            if _followup_outcome_valid(key, value):
                fields.append((key, value))
    error = result.get("error")
    if result.get("status") == "failed" and isinstance(error, dict):
        code = error.get("code")
        if _followup_outcome_valid("error_code", code):
            fields.append(("error_code", code))
    return tuple(fields)


@dataclass(frozen=True, slots=True)
class RealtimeToolRequest:
    """Source identity plus bounded transient USER content, excluded from repr.

    This object is not a ledger/telemetry payload. Source materialization still
    records only roots; the quotation is rendered only inside the final guard.
    """

    owner_id: str
    session_id: UUID
    turn_generation: int
    event_id: UUID
    payload_id: UUID
    revision: int
    provider_item_id: str
    user_text: str | None = field(default=None, repr=False)
    user_content_status: Literal["available", "unavailable", "omitted_byte_budget"] = "unavailable"

    def __post_init__(self):
        if self.user_content_status not in {"available", "unavailable", "omitted_byte_budget"}:
            raise ValueError("invalid_tool_followup_user_content")
        if self.user_text is None:
            if self.user_content_status == "available":
                raise ValueError("invalid_tool_followup_user_content")
        elif _followup_user_content(self.user_text) != (self.user_text, self.user_content_status):
            raise ValueError("invalid_tool_followup_user_content")

    @classmethod
    def from_record(cls, record, *, owner_id, session_id, turn_generation, source_namespace):
        if (not isinstance(record, ContextRecord) or not record.authorizes_request
                or record.owner_id != owner_id or record.session_id != session_id
                or record.source_namespace != source_namespace or record.revision < 1):
            return None
        item_id = (record.content or {}).get("provider_item_id")
        if not _followup_token(item_id):
            return None
        text, availability = _followup_user_content((record.content or {}).get("text"))
        return cls(owner_id, session_id, turn_generation, record.id, record.payload_id,
                   record.revision, item_id, text, availability)


@dataclass(frozen=True, slots=True)
class RealtimeToolReceipt:
    call_id: str
    tool_name: str
    status: Literal["success", "failed", "unknown"]
    event_id: UUID | None = None
    payload_id: UUID | None = None
    local_effect_reported: bool = False
    owner_outcome: tuple[tuple[str, str | bool | int], ...] = field(default=(), repr=False)

    def __post_init__(self):
        if (not _followup_token(self.call_id) or not _followup_token(self.tool_name, 80)
                or self.status not in {"success", "failed", "unknown"}
                or (self.event_id is None) != (self.payload_id is None)
                or (self.event_id is not None and not all(isinstance(value, UUID)
                    for value in (self.event_id, self.payload_id)))
                or type(self.local_effect_reported) is not bool
                or len(self.owner_outcome) > 7
                or len({key for key, _ in self.owner_outcome}) != len(self.owner_outcome)
                or any(not _followup_outcome_valid(key, value) for key, value in self.owner_outcome)):
            raise ValueError("invalid_tool_followup_receipt")

    @classmethod
    def from_result(cls, call_id, tool_name, result):
        bound = isinstance(result, SourceBoundToolResult)
        error = result.get("error")
        return cls(call_id, tool_name,
            result.get("status") if result.get("status") in {"success", "failed"} else "unknown",
            result.event_id if bound else None, result.payload_id if bound else None,
            isinstance(error, dict) and error.get("local_effect_reported") is True,
            _followup_owner_outcome(result, tool_name))


@dataclass(frozen=True, slots=True)
class RealtimeToolFollowup:
    request: RealtimeToolRequest | None = None
    receipts: tuple[RealtimeToolReceipt, ...] = ()
    omitted_receipts: int = 0
    binding_conflict: bool = False
    # Same-turn source context only. Offering is not fulfillment; the next
    # guarded response must revalidate and issue fresh per-call references.
    source_offer: Any = field(default=None, repr=False, kw_only=True)
    additional_requests: tuple[RealtimeToolRequest, ...] = ()
    conversation_request: RealtimeToolRequest | None = None
    call_request_ids: tuple[tuple[str, UUID], ...] = ()

    def append(self, receipt: RealtimeToolReceipt):
        if any(previous.call_id == receipt.call_id for previous in self.receipts):
            return self
        if len(self.receipts) >= 16:
            return replace(self, omitted_receipts=self.omitted_receipts + 1)
        return replace(self, receipts=(*self.receipts, receipt))


@dataclass(slots=True)
class PendingRealtimeResponse:
    """One admitted response awaiting its correlated provider acknowledgement."""

    request_id: str
    kind: ResponseKind
    turn_generation: int
    metadata: dict[str, str] = field(default_factory=dict)
    submitted: bool = False
    task_instructions: str | None = None
    requested_image_call_ids: tuple[str, ...] = ()
    unavailable_requested_images: tuple[dict[str, Any], ...] = ()
    tool_followup: RealtimeToolFollowup | None = None
    user_admission_scope: Any = None
    preserve_unwritten_request: bool = False
    request_sources: Any = None

    @property
    def is_delivery_continuation(self) -> bool:
        return self.kind == "normal_user_turn" and self.metadata.get("jarvis_delivery_kind") in {
            "visual_interrupt", "reminder", "prospective",
        }


@dataclass(frozen=True, slots=True)
class RealtimeResponseContinuation:
    turn_generation: int
    metadata: dict[str, str]
    task_instructions: str | None = None
    source_response_id: str | None = None
    requested_image_call_ids: tuple[str, ...] = ()
    tool_followup: RealtimeToolFollowup | None = None


@dataclass(slots=True)
class RealtimeResponseState:
    response_id: str
    kind: ResponseKind
    output_item_id: str | None = None
    output_items: dict[str, RealtimeOutputItemState] = field(default_factory=dict)
    interrupted: bool = False
    dismissed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    task_instructions: str | None = None
    tool_followup: RealtimeToolFollowup = field(default_factory=RealtimeToolFollowup)
    pending_tool_call_ids: set[str] = field(default_factory=set)
    requested_image_call_ids: list[str] = field(default_factory=list)
    completed_response: dict[str, Any] | None = None
    completion_started: bool = False
    # Pinned after a matching submitted request is acknowledged. Original source
    # timestamps remain in canonical lineage; this does not authorize new tools.
    source_snapshot: Any = None
    request_sources: Any = None
    # Local initial-admission proof, never reconstructed from provider metadata.
    user_admission_scope: Any = field(default=None, repr=False)
    user_admission_generation: int | None = None

    @property
    def ordered_output_items(self) -> list[RealtimeOutputItemState]:
        indexed = enumerate(self.output_items.values())
        return [
            item for _, item in sorted(
                indexed,
                key=lambda pair: (
                    pair[1].output_index if pair[1].output_index is not None else pair[0]
                ),
            )
        ]

    @property
    def has_forwarded_answer(self) -> bool:
        return any(item.has_forwarded_answer for item in self.output_items.values())

    @property
    def delivery_kind(self) -> ResponseKind:
        # A tool continuation is admitted as a normal turn but still owns the
        # original reminder/prospective delivery until its terminal answer.
        kind = self.metadata.get("jarvis_delivery_kind", self.kind)
        return kind if kind in ("visual_interrupt", "reminder", "prospective") else self.kind

    def delivery_continuation_metadata(self) -> dict[str, str]:
        if self.delivery_kind == "normal_user_turn":
            return {}
        keys = (
            "visual_interrupt_id", "reminder_id", "reminder_event_id",
            "prospective_event_id", "prospective_subject_type", "prospective_subject_id",
        )
        return {
            "jarvis_delivery_kind": self.delivery_kind,
            **{
                key: value for key in keys
                if isinstance(value := self.metadata.get(key), str)
            },
        }
