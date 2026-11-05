from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator
from uuid import UUID


@dataclass(frozen=True, slots=True)
class TaskToolInvocation:
    """Trusted admitted-worker identity, never populated from tool arguments."""

    task_id: UUID
    task_revision: int
    attempt_id: UUID
    action_id: UUID | None = None
    operation_key: str | None = None

    def __post_init__(self):
        if not isinstance(self.task_id, UUID) or not isinstance(self.attempt_id, UUID):
            raise ValueError("task and attempt identities must be UUIDs")
        if type(self.task_revision) is not int or self.task_revision < 1:
            raise ValueError("task revision must be a positive integer")
        if (self.action_id is None) != (self.operation_key is None):
            raise ValueError("action identity and operation key must be supplied together")
        if self.action_id is not None and (
            not isinstance(self.action_id, UUID) or not isinstance(self.operation_key, str)
            or not 1 <= len(self.operation_key.strip()) <= 200
        ):
            raise ValueError("invalid task action identity")


@dataclass(frozen=True, slots=True)
class CanonicalRequestIdentity:
    """Stable source names resolved by the trusted voice adapter, not tool JSON."""
    namespace: str
    event_id: str

    def __post_init__(self):
        if any(not isinstance(value, str) or not value.strip() or len(value) > 512
               for value in (self.namespace, self.event_id)):
            raise ValueError("invalid canonical source identity")

    @classmethod
    def from_record(cls, record):
        namespace = getattr(record, "source_namespace", None)
        event_id = getattr(record, "source_event_id", None)
        return cls(namespace, event_id) if isinstance(namespace, str) and isinstance(event_id, str) else None


@dataclass(frozen=True, slots=True)
class ToolInvocationContext:
    """Identity that belongs to one provider-initiated tool invocation."""

    provider_call_id: str | None = None
    # Supplied by the trusted voice/task adapter, never tool JSON arguments.
    request_event_id: UUID | None = None
    request_revision: int | None = None
    task: TaskToolInvocation | None = None
    request_identity: CanonicalRequestIdentity | None = None

    def __post_init__(self):
        if self.request_identity is not None and (
            not isinstance(self.request_identity, CanonicalRequestIdentity) or self.request_event_id is None
        ):
            raise ValueError("canonical identity requires trusted request evidence")
        if self.task is not None and not isinstance(self.task, TaskToolInvocation):
            raise ValueError("task invocation must be trusted typed context")
        if (self.request_event_id is None) != (self.request_revision is None):
            raise ValueError("request event identity and revision must be supplied together")
        if self.request_revision is not None and (type(self.request_revision) is not int or self.request_revision < 1):
            raise ValueError("request revision must be a positive integer")


_CURRENT_TOOL_INVOCATION: ContextVar[ToolInvocationContext] = ContextVar(
    "jarvis_tool_invocation",
    default=ToolInvocationContext(),
)


def current_tool_invocation() -> ToolInvocationContext:
    return _CURRENT_TOOL_INVOCATION.get()


@contextmanager
def bind_tool_invocation(
    *,
    provider_call_id: str | None,
    request_event_id: UUID | None = None,
    request_revision: int | None = None,
    task: TaskToolInvocation | None = None,
    request_identity: CanonicalRequestIdentity | None = None,
) -> Iterator[ToolInvocationContext]:
    normalized_call_id = provider_call_id.strip() if provider_call_id else None
    context = ToolInvocationContext(provider_call_id=normalized_call_id or None,
        request_event_id=request_event_id, request_revision=request_revision, task=task,
        request_identity=request_identity)
    token = _CURRENT_TOOL_INVOCATION.set(context)
    try:
        yield context
    finally:
        _CURRENT_TOOL_INVOCATION.reset(token)
