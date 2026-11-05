"""Lightweight admitted domain scope, also importable by the native plugin."""
from dataclasses import dataclass
from uuid import UUID

from app.hermes_sessions.bindings import bounded_id


@dataclass(frozen=True)
class DomainInputScope:
    """Server-resolved scope of ONE admitted input, not the master's first turn."""

    owner_id: str
    logical_session_id: str
    input_id: str
    jarvis_session_id: UUID
    allowed_tools: frozenset[str]
    request_event_id: UUID | None = None
    request_revision: int | None = None

    def __post_init__(self):
        for value in (self.owner_id, self.logical_session_id, self.input_id):
            bounded_id(value)
        if not isinstance(self.jarvis_session_id, UUID):
            raise ValueError("An owned Jarvis session UUID is required")
        if not isinstance(self.allowed_tools, frozenset) or len(self.allowed_tools) > 64:
            raise ValueError("Tool scope must be an immutable bounded allowlist")
        for name in self.allowed_tools:
            bounded_id(name)
        if (self.request_event_id is None) != (self.request_revision is None):
            raise ValueError("Request identity and revision belong together")
        if self.request_event_id is not None and (
            not isinstance(self.request_event_id, UUID)
            or type(self.request_revision) is not int or self.request_revision < 1
        ):
            raise ValueError("Invalid original request reference")
