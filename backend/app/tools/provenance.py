"""Trusted tool-result lineage carried beside, never inside, model-facing JSON."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from app.tasks.models import TaskError, digest

if TYPE_CHECKING:
    from app.tasks.context_capture import SessionContextCapture


@dataclass(frozen=True)
class ToolResultCapture:
    capture: SessionContextCapture
    call_id: str
    tool_name: str


class SourceBoundToolResult(dict):
    def __init__(self, payload, record):
        super().__init__(payload)
        self.payload_id = record.payload_id
        self.event_id = record.id

    def audit_receipt(self):
        return {"status":"success", "source_bound_tool_result":{
            "protocol_version":1, "context_event_id":str(self.event_id)}}


def source_read_arguments(arguments):
    return {"arguments_digest":digest(arguments)}


async def restore_source_result(capture, *, owner_id, session_id, call_id, name, saved):
    try:
        reference = saved["source_bound_tool_result"]
        if (type(reference["protocol_version"]) is not int or reference["protocol_version"] != 1
                or capture is None or capture.owner_id != owner_id or capture.session_id != session_id):
            raise ValueError("Invalid result scope")
        identity = UUID(reference["context_event_id"])
        record = await capture.repository.get(owner_id, identity, session_id=session_id)
        content = record.content or {}
        result = content.get("result")
        if (record.availability != "available" or not record.is_current or record.kind != "tool_outcome"
                or content.get("provider_call_id") != call_id or content.get("tool_name") != name
                or not isinstance(result, dict) or result.get("status") != "success"):
            raise ValueError("Result evidence is unavailable")
        return SourceBoundToolResult(result, record)
    except (KeyError, TypeError, ValueError, TaskError) as error:
        raise TaskError("tool_result_source_unavailable",
            "The original tool result is unavailable; do not reconstruct or repeat its contents.") from error

