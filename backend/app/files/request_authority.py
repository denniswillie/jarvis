"""Opt-in authority for conversational text files, not a worker/admission system.

Compose SourceFileRepository(..., request_authority=FileRequestAuthority(tasks))
with the existing request repository on the source pool. Only create_text_file
is supported. The publication manifest retains a small original-request reference;
recovery never consults the currently executing tool or a newer user request.
"""
from __future__ import annotations

from uuid import UUID

from app.knowledge.models import SourceError
from app.tasks.models import TaskError, TaskVoiceSource
from app.tools.invocation import current_tool_invocation


REQUEST_AUTHORITY_KEY = "file_request_authority"


class FileRequestAuthority:
    def __init__(self, tasks):
        self.tasks = tasks

    @staticmethod
    def _text_only(spec):
        if (spec.source != "assistant" or spec.content_type != "text/plain; charset=utf-8"
                or spec.source_visual_frame_id is not None or not isinstance(spec.session_id, UUID)):
            raise SourceError("file_request_mode_unsupported",
                              "Request-scoped publication supports conversational text files only.", 403)

    def bind(self, *, owner_id, spec):
        """Capture trusted invocation metadata, never model-supplied request IDs."""
        self._text_only(spec)
        if spec.metadata:
            raise SourceError("file_request_scope_override", "Request-scoped file metadata is server-owned.", 403)
        invocation = current_tool_invocation()
        if (invocation.task is not None or not invocation.provider_call_id
                or invocation.request_event_id is None or invocation.request_revision is None):
            raise SourceError("file_request_required", "An original conversational user request is required.", 403)
        reference = TaskVoiceSource(invocation.request_event_id, invocation.request_revision, spec.session_id)
        return spec.model_copy(update={"metadata": {REQUEST_AUTHORITY_KEY: {
            "version": 1, "owner_id": owner_id, **reference.parameters(),
        }}})

    async def validate_locked(self, connection, *, owner_id, spec):
        """Return the original payload under the caller's source/task-owner locks.

        FilePublicationService takes those existing locks before publication row
        locks. This method adds the existing exact-request/event/payload guards;
        it performs no external I/O and creates no Task, WorkerClaim or grant.
        """
        if not connection.is_in_transaction():
            raise RuntimeError("File request validation requires a publication transaction")
        self._text_only(spec)
        raw = spec.metadata.get(REQUEST_AUTHORITY_KEY)
        if (set(spec.metadata) != {REQUEST_AUTHORITY_KEY} or not isinstance(raw, dict)
                or set(raw) != {"version", "owner_id", "event_id", "revision", "session_id"}
                or type(raw.get("version")) is not int or raw["version"] != 1):
            raise SourceError("file_request_required", "The publication has no supported original request reference.", 403)
        try:
            reference = TaskVoiceSource(UUID(raw["event_id"]), raw["revision"], UUID(raw["session_id"]))
        except (ValueError, TypeError, AttributeError):
            raise SourceError("file_request_invalid", "The publication request reference is invalid.", 403) from None
        if raw["owner_id"] != owner_id or reference.session_id != spec.session_id:
            raise SourceError("file_request_scope_mismatch", "The publication request belongs to another owner/session.", 403)
        try:
            row = await self.tasks._voice_source(connection, owner_id, reference)
        except TaskError:
            raise SourceError("file_request_unavailable",
                              "The exact original user request changed or is unavailable; no new file effect was authorized.",
                              403) from None
        return row["payload_id"]
