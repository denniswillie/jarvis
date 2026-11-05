"""Eval-only canonical readiness, not a scheduler, loader or authority producer.

The host must supply a fresh, committed, narrowly scoped snapshot for EACH consume
nonce. Use current canonical repositories, not old exports or caller-invented
availability. This pure adapter performs no I/O, polling, writes, clock override,
tool dispatch, retention or playback control. EvidenceRef validates a reference's
shape, not its authenticity. The opt-in replay observer owns admission/source
observations and its lifecycle joins. This pure module has no host or IO side effects.

response_input_association is reused unchanged. Its null admission-time canonical
fields are intentional; the resolver supplies the actual finalized revision.
Neither a shared input nor temporal proximity proves tool-result continuation.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

from app.knowledge.models import SourceVersion
from app.tasks.context_events import ContextRecord
from app.tasks.models import digest
from evals.responsive_voice.native_memory_replay_control import EvidenceRef, Scope
from evals.responsive_voice.response_input_association import resolve_response_input


def _identity(value, limit=512):
    return isinstance(value, str) and 0 < len(value) <= limit and value.strip() == value and "\x00" not in value


def _aware(value):
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


@dataclass(frozen=True)
class ReadinessRequest:
    """Projection of the declared manual control request, not a launch grant.

    earliest at_seconds and playback none/authenticated_buffer_stop remain frontend
    predicates. This adapter never moves an enqueue or asserts decoded/audio drain.
    """
    scope: Scope
    invocation_id: str
    manifest_sha256: str
    session_id: UUID
    audio_session_id: str
    connection_id: str
    prior_input_ordinal: int
    target_input_ordinal: int
    segment_id: str
    client_event_id: str
    response_id: str
    item_id: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    canonical_requirement: str = "retain_searchable"

    def __post_init__(self):
        if not isinstance(self.scope, Scope) or not isinstance(self.session_id, UUID):
            raise ValueError("Exact fixture scope and actual session are required")
        for name in ("invocation_id", "audio_session_id", "connection_id", "segment_id",
                     "client_event_id", "response_id", "item_id", "nonce"):
            if not _identity(getattr(self, name)):
                raise ValueError("Invalid control identity")
        if (not isinstance(self.manifest_sha256, str) or len(self.manifest_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.manifest_sha256)):
            raise ValueError("Exact manifest SHA256 required")
        if (type(self.prior_input_ordinal) is not int or type(self.target_input_ordinal) is not int
                or not 0 <= self.prior_input_ordinal < self.target_input_ordinal <= 1024):
            raise ValueError("Invalid declared input ordinals")
        if (not _aware(self.issued_at) or not _aware(self.expires_at)
                or not timedelta(0) < self.expires_at - self.issued_at <= timedelta(seconds=120)):
            raise ValueError("A positive real-clock deadline of at most 120 seconds is required")
        if self.canonical_requirement not in {"finalized_input", "retain_searchable"}:
            raise ValueError("Unsupported canonical prerequisite")


@dataclass(frozen=True)
class CanonicalMemorySnapshot:
    """Fixed-size input from a fresh scoped read; no full-table export is needed.

    admission is the explicit input_commit_admitted projection, including the
    control binding below, request_id, turn_generation and an EvidenceRef.
    continuation is an explicit observed tool-result -> admitted response source
    snapshot -> output edge, NOT a shared-input or temporal inference. It proves
    inclusion in the response's source set, not that this tool caused the response
    or that the model used the result. Missing evidence is not synthesized.
    At most one source/retain row is admitted; duplicates are ambiguous.
    """
    scope: Scope
    nonce: str
    observed_at: datetime
    request: ContextRecord | None = None
    output: ContextRecord | None = None
    tool_outcome: ContextRecord | None = None
    memory_event: ContextRecord | None = None
    sources: tuple[SourceVersion, ...] = ()
    retains: tuple[dict, ...] = ()
    admission: dict | None = None
    continuation: dict | None = None


def _control(request):
    return {"run_id": request.scope.run_id, "owner_id": request.scope.owner_id,
            "invocation_id": request.invocation_id, "manifest_sha256": request.manifest_sha256,
            "session_id": str(request.session_id), "audio_session_id": request.audio_session_id,
            "connection_id": request.connection_id, "prior_input_ordinal": request.prior_input_ordinal,
            "target_input_ordinal": request.target_input_ordinal, "segment_id": request.segment_id,
            "client_event_id": request.client_event_id}


def _matches(value, expected):
    # Reject bool-as-int and partial scope projections, not just unequal values.
    return (isinstance(value, Mapping) and len(value) <= 40
            and all(key in value and type(value[key]) is type(item) and value[key] == item
                    for key, item in expected.items()))


def _current(record, request, kind, trust):
    return (isinstance(record, ContextRecord) and record.owner_id == request.scope.owner_id
            and record.session_id == request.session_id and record.kind == kind and record.trust == trust
            and record.is_current is True and record.availability == "available"
            and record.finality in {"final", "revised"} and type(record.revision) is int
            and record.revision > 0 and isinstance(record.content, dict) and len(record.content) <= 40)


def evaluate_memory_readiness(request: ReadinessRequest, snapshot: CanonicalMemorySnapshot, *, now: datetime) -> dict:
    """Evaluate exactly this consume nonce. A returned ready receipt is not sticky.

    Future consumption must request a new nonce and reread canonical state; copying
    old records into a new envelope violates the caller contract. No cryptographic
    authenticity, native input mapping, playback or semantic truth is invented here.
    """
    base = {"format": "jarvis.native_memory_readiness.v1", **_control(request),
            "nonce": request.nonce, "canonical_requirement": request.canonical_requirement,
            "response_id": request.response_id, "item_id": request.item_id,
            "expires_at": request.expires_at.isoformat(), "canonical_input": None,
            "full_chain_proven": False, "heard_state": "unknown", "playback": "not_evaluated",
            "semantic_acceptance": "not_evaluated", "evidence_authenticity": "supplied_by_scoped_host",
            "control_label_basis": "frontend_declared_connection_and_ordinals_not_backend_observed",
            "model_use_of_tool_result": "not_established"}

    def result(status, reason, **evidence):
        return base | {"status": status, "reason": reason, **evidence}

    if not _aware(now) or now < request.issued_at:
        return result("unavailable", "invalid_real_clock")
    if now >= request.expires_at:
        return result("expired", "control_deadline")
    if snapshot.scope != request.scope or snapshot.nonce != request.nonce:
        return result("unavailable", "snapshot_scope_or_nonce")
    if not _aware(snapshot.observed_at) or not request.issued_at <= snapshot.observed_at <= now:
        return result("unavailable", "snapshot_not_fresh_for_request")
    if len(snapshot.sources) > 1 or len(snapshot.retains) > 1:
        return result("unavailable", "ambiguous_source_or_retain")
    admission = snapshot.admission
    if not _matches(admission, _control(request) | {"event": "input_commit_admitted"}):
        return result("unavailable", "manual_commit_admission_unavailable")
    if (not _identity(admission.get("request_id")) or type(admission.get("turn_generation")) is not int
            or not 0 <= admission["turn_generation"] < 2**63
            or not isinstance(admission.get("evidence"), EvidenceRef)):
        return result("unavailable", "manual_commit_admission_invalid")
    if snapshot.output is None:
        return result("waiting", "canonical_output_missing")
    association = resolve_response_input(output=snapshot.output, request=snapshot.request,
        owner_id=request.scope.owner_id, session_id=request.session_id,
        audio_session_id=request.audio_session_id, response_id=request.response_id, item_id=request.item_id)
    if association["status"] != "resolved":
        return result("unavailable", association["reason"])
    binding = snapshot.output.content["response_input_association"]
    if (binding["input_id"] != "commit:" + admission["request_id"]
            or binding["turn_generation"] != admission["turn_generation"]):
        return result("unavailable", "manual_commit_response_mismatch")
    canonical = association["canonical_input"]
    base["canonical_input"] = canonical
    if not _current(snapshot.output, request, "assistant_output", "assistant_output"):
        return result("waiting", "assistant_output_not_terminal")
    completion = snapshot.output.content.get("completion_state")
    if completion != "completed":
        return result("unavailable" if completion == "interrupted" else "waiting", "assistant_output_not_completed")
    response = {"event_id": str(snapshot.output.id), "revision": snapshot.output.revision,
                "response_id": request.response_id, "item_id": request.item_id,
                "completion_state": "completed"}
    if request.canonical_requirement == "finalized_input":
        return result("ready", "finalized_input_and_terminal_response", response=response,
                      admission_evidence=asdict(admission["evidence"]))

    tool, memory = snapshot.tool_outcome, snapshot.memory_event
    if tool is None:
        return result("waiting", "committed_remember_receipt_missing")
    if not _current(tool, request, "tool_outcome", "tool_receipt"):
        return result("unavailable", "remember_receipt_not_current_available")
    content = tool.content
    call = content.get("provider_call_id")
    if (not _identity(call) or tool.source_namespace != snapshot.request.source_namespace
            or tool.source_event_id != "bound-tool:" + digest({"identity": call})
            or content.get("tool_name") != "remember" or content.get("input_id") != binding["input_id"]
            or content.get("audio_provider") != "openai" or content.get("audio_session_id") != request.audio_session_id):
        return result("unavailable", "remember_tool_input_mismatch")
    saved = content.get("result")
    if (not isinstance(saved, dict) or len(saved) > 16 or saved.get("status") != "success"
            or saved.get("memory_status") != "stored" or not isinstance(saved.get("source"), dict)
            or len(saved["source"]) > 40):
        return result("unavailable", "remember_tool_result_not_stored")
    if not _current(memory, request, "tool_outcome", "tool_receipt"):
        return result("unavailable", "remember_original_not_current_available")
    statement = memory.content
    if (memory.source_namespace != "conversation-explicit-memory"
            or set(statement) != {"memory_type", "content", "occurred_at", "attribution"}
            or statement.get("memory_type") not in {"fact", "experience", "procedure", "goal"}
            or not isinstance(statement.get("content"), str) or not 0 < len(statement["content"]) <= 8000
            or (statement.get("occurred_at") is not None and not _identity(statement["occurred_at"], 100))
            or not _matches(statement.get("attribution"), {
                "kind": "model_formulated_from_finalized_user_request",
                "request_event_id": canonical["event_id"], "request_revision": canonical["revision"]})):
        return result("unavailable", "remember_request_revision_mismatch")
    if memory.source_event_id != digest({"session_id": str(request.session_id), "statement": statement}):
        return result("unavailable", "remember_statement_identity_mismatch")
    if not snapshot.sources:
        return result("unavailable", "memory_source_unavailable")
    source = snapshot.sources[0]
    if (not isinstance(source, SourceVersion) or source.owner_id != request.scope.owner_id
            or str(source.area_id) != request.scope.area_id or source.bank_name != request.scope.memory_bank_id
            or source.session_id != request.session_id or source.context_session_id != request.session_id
            or source.is_current is not True or source.source_type != "tool_outcome" or source.trust != "tool_receipt"
            or source.context_event_id != memory.id or source.context_event_revision != memory.revision
            or source.payload_id != memory.payload_id or source.content_digest != memory.content_digest
            or source.content != statement or source.semantic_index_exclusion is not None
            or saved.get("memory_id") != str(source.source_id)):
        return result("unavailable", "memory_source_scope_or_binding")
    pinned = {"owner_id": source.owner_id, "area_id": str(source.area_id), "source_id": str(source.source_id),
              "revision": source.revision, "generation": source.generation, "payload_id": str(source.payload_id),
              "document_id": source.document_id, "content_digest": source.content_digest}
    if not _matches(saved["source"], pinned):
        return result("unavailable", "saved_source_revision_changed")
    if not snapshot.retains:
        return result("waiting", "exact_retain_missing")
    retain = snapshot.retains[0]
    if (not _matches(retain, {"owner_id": source.owner_id, "document_id": source.document_id,
                             "generation": source.generation, "operation": "retain"})
            or not isinstance(retain.get("id"), UUID) or type(retain.get("dispatch_count")) is not int):
        return result("unavailable", "retain_identity_mismatch")
    statuses = {retain.get("status"), source.ingestion_status}
    if statuses & {"failed", "cancelled", "done"}:
        return result("unavailable", "retain_not_searchable_terminal")
    if statuses != {"searchable"}:
        return result("waiting", "retain_not_searchable")
    if retain["dispatch_count"] < 1:
        return result("unavailable", "searchable_without_dispatch")
    continuation = snapshot.continuation
    expected = {"kind": "observed_tool_result_continuation", "owner_id": request.scope.owner_id,
                "session_id": str(request.session_id), "audio_session_id": request.audio_session_id,
                "connection_id": request.connection_id, "response_id": request.response_id,
                "request_id": admission["request_id"], "turn_generation": admission["turn_generation"],
                "provider_call_id": call, "tool_event_id": str(tool.id), "tool_revision": tool.revision,
                "tool_payload_id": str(tool.payload_id), "request_event_id": canonical["event_id"],
                "request_revision": canonical["revision"]}
    if (not _matches(continuation, expected)
            or not isinstance(continuation.get("evidence"), EvidenceRef)):
        return result("unavailable", "explicit_tool_continuation_unavailable")
    return result("ready", "exact_remember_searchable_and_continued", response=response,
                  retained_source=pinned | {"retain_operation_id": str(retain["id"]), "status": "searchable"},
                  admission_evidence=asdict(admission["evidence"]),
                  continuation_evidence=asdict(continuation["evidence"]))
