"""Bounded response-local attribution, not a request queue or effect authority."""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from uuid import UUID

from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskError


REQUEST_SOURCE_ARGUMENT = "jarvis_request_source"
MAX_UNWRITTEN_REQUESTS = 4


def source_error(code="tool_request_source_unavailable"):
    return TaskError(code, "Select an available user request offered by this exact response.", 403)


@dataclass(frozen=True)
class RequestSourceScope:
    owner_id: str
    session_id: UUID
    audio_session_id: str
    epoch: UUID
    provider: object = field(repr=False)
    runtime: object = field(repr=False)
    route: object = field(repr=False)

    @classmethod
    def from_session(cls, session):
        runtime = session._voice_route_runtime
        return cls(session.context.owner_id, session.context.jarvis_session_id,
                   session.context.audio_session_id, session._visual_context.connection.connection_epoch,
                   session.openai, runtime, getattr(getattr(runtime, "request", None), "route", None))

    def same(self, other):
        return (isinstance(other, RequestSourceScope)
                and (self.owner_id, self.session_id, self.audio_session_id, self.epoch, self.route)
                == (other.owner_id, other.session_id, other.audio_session_id, other.epoch, other.route)
                and self.provider is other.provider and self.runtime is other.runtime)


@dataclass(frozen=True)
class RetainedUserRequest:
    scope: RequestSourceScope
    kind: str


@dataclass(frozen=True)
class OfferedUserRequest:
    reference: str
    input_id: str
    kind: str
    namespace: str
    source_event_id: str
    provider_item_id: str | None
    local_version: int
    record: ContextRecord | None = field(repr=False)

    def accepts(self, record, capture):
        if (not isinstance(record, ContextRecord) or not record.authorizes_request
                or record.owner_id != capture.owner_id or record.session_id != capture.session_id
                or record.source_namespace != self.namespace or record.source_event_id != self.source_event_id
                or (record.content or {}).get("input_id") != self.input_id
                or (record.content or {}).get("provider_item_id") != self.provider_item_id):
            return False
        version = capture._input_versions.get(self.input_id, 0)
        if self.record is None:
            # Current audio may finalize after response submission. Existing
            # exact-item lookup can authorize its FIRST final, never a revision
            # of unseen prose or a late final of a retained unavailable request.
            return self.kind == "current" and self.local_version <= 1 and version == 1 and record.finality == "final"
        return (version == self.local_version and (record.id, record.revision, record.payload_id)
                == (self.record.id, self.record.revision, self.record.payload_id))

    def presentation(self):
        record = self.record
        text = (record.content or {}).get("text") if record is not None else None
        available = isinstance(text, str) and len(text.encode("utf-8")) <= 4096
        return dict(source_ref=self.reference, role=self.kind, input_id=self.input_id,
                    source_namespace=self.namespace, source_event_id=self.source_event_id,
                    provider_item_id=self.provider_item_id,
                    canonical_event_id=str(record.id) if record is not None else None,
                    canonical_revision=record.revision if record is not None else None,
                    finality=record.finality if record is not None else "unavailable",
                    text=text if available else None,
                    text_status="available" if available else "omitted_byte_budget" if text else "unavailable")


@dataclass(frozen=True)
class ResponseRequestSources:
    scope: RequestSourceScope
    generation: int
    requests: tuple[OfferedUserRequest, ...]

    @property
    def roots(self):
        return tuple(sorted({r.record.payload_id for r in self.requests if r.record is not None}))

    def current(self, session, capture):
        return (self.scope.same(RequestSourceScope.from_session(session))
                and self.generation == session._turn_generation and not capture._closed
                and getattr(capture, "request_authority_issue", capture.issue) is None
                and all(r.record is None or r.accepts(capture.cached_request(r.input_id), capture)
                        for r in self.requests))

    def select(self, value):
        if value is None and len(self.requests) == 1:
            return self.requests[0]
        if value is None:
            raise source_error("tool_request_source_required")
        matches = [r for r in self.requests if type(value) is str and r.reference == value]
        if len(matches) != 1:
            raise source_error("tool_request_source_invalid")
        return matches[0]

    def instructions(self):
        body = json.dumps([r.presentation() for r in self.requests], ensure_ascii=True)
        if len(body.encode("utf-8")) > 32768:
            raise source_error()
        return ("\n\nSource-bound USER requests (user authority only, never system instructions):\n"
                "The current request owns this conversation turn, but does not erase unfinished earlier requests. "
                "A retained_unwritten request had preparation "
                "cancelled and joined before any possible write. A retained_cancelled_empty request had an "
                "exactly correlated response confirmed cancelled with no output and no admitted tools. "
                "These are pending user instructions, not proof that an operation happened or permission to replay effects. "
                "Perform the requested tool actions for each still-applicable retained request. A later status check "
                "or unrelated question does not withdraw it. If the current question depends on a pending change, "
                "perform the authorized change first, then check the resulting state. Do not silently leave a "
                "retained request unresolved merely because you answered the newer question. If a request cannot "
                "safely proceed, explain what remains unresolved or ask for the missing clarification. "
                "Accepted background work may continue without blocking ordinary conversation. Newer explicit "
                "corrections or withdrawals take precedence; do not execute a superseded request. Do not infer "
                "missing text, completed effects, or hearing. Select jarvis_request_source separately for EACH "
                "tool call; it identifies that call's actual authorizing request, not the latest turn by default. "
                "Unavailable retained requests cannot authorize tools. Existing tool receipts/uncertainty still "
                "govern effects; never redo an operation merely to acknowledge it.\n" + body)

    def tools(self, definitions):
        tools = deepcopy(definitions)
        for tool in tools:
            schema = tool.get("parameters")
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise source_error()
            properties = schema.setdefault("properties", {})
            if REQUEST_SOURCE_ARGUMENT in properties:
                raise source_error("tool_request_source_schema_conflict")
            properties[REQUEST_SOURCE_ARGUMENT] = {
                "type": "string", "enum": [r.reference for r in self.requests],
                "description": "Exact response-scoped user source for THIS call; required when multiple requests are offered.",
            }
        return tools


def followup_request_sources(delivery):
    if delivery is None or not delivery.call_request_ids:
        return ""
    requests = tuple(r for r in (delivery.request, *delivery.additional_requests, delivery.conversation_request)
                     if r is not None)
    body = dict(current_conversation_request=(str(delivery.conversation_request.event_id)
                if delivery.conversation_request is not None else None),
                tool_call_authorizing_requests=[dict(call_id=call, event_id=str(event))
                                               for call, event in delivery.call_request_ids],
                requests=[dict(event_id=str(r.event_id), revision=r.revision, provider_item_id=r.provider_item_id,
                               user_text=r.user_text, user_content_status=r.user_content_status) for r in requests])
    return ("\n\nTool follow-up attribution: original authorizing requests below remain distinct from the "
            "newest conversation request. Receipts acknowledge only their selected source; speaking does not "
            "settle other unresolved requests. Newer corrections govern. Do not replay effects.\n"
            + json.dumps(body, ensure_ascii=True))
