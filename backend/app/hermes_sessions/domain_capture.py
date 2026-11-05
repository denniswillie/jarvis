"""Per-invocation source capture; no voice lifecycle or copied native transcript."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from app.experience.evidence import durable_tool_payload
from app.images import image_source_payload_ids
from app.tasks.context_events import ContextIngress, ContextRecord
from app.tasks.models import TaskError, digest


class NativeToolCapture:
    """The narrow capture interface consumed by the existing Jarvis executor."""

    def __init__(self, repository, *, scope, call_id, write_timeout=3.0):
        self.repository, self.scope, self.call_id = repository, scope, call_id
        self.owner_id, self.session_id = scope.owner_id, scope.jarvis_session_id
        self.namespace = "hermes:" + digest({"session": scope.logical_session_id})
        self.write_timeout = write_timeout
        self._request = None
        self._prepared = False
        self._pending = None

    async def prepare(self):
        if self._prepared:
            return
        if self.scope.request_event_id is not None:
            async with asyncio.timeout(self.write_timeout):
                record = await self.repository.get(self.owner_id, self.scope.request_event_id,
                                                   session_id=self.session_id)
            if (not isinstance(record, ContextRecord) or record.owner_id != self.owner_id
                    or record.session_id != self.session_id
                    or record.id != self.scope.request_event_id
                    or record.revision != self.scope.request_revision
                    or not record.authorizes_request):
                raise TaskError("hermes_request_unavailable",
                                "The original request was changed or is unavailable; do not substitute another turn.")
            self._request = record
        self._prepared = True

    async def request_for_call(self, call_id):
        if not self._prepared or call_id != self.call_id:
            raise TaskError("hermes_call_scope_mismatch", "The tool call is outside this admitted input.")
        return self._request

    def _event(self, call_id, name, result, sources, *, bound, mutates_state=False):
        if not self._prepared or call_id != self.call_id:
            raise TaskError("hermes_call_scope_mismatch", "The tool call is outside this admitted input.")
        dependencies = tuple(sources)
        if self._request is not None:
            dependencies += (self._request.payload_id,)
        payload = {"provider_call_id": call_id, "tool_name": name, "result": result,
                   "input_id": self.scope.input_id, "mutates_state": mutates_state}
        return ContextIngress(session_id=self.session_id, source_namespace=self.namespace,
            source_event_id=("bound-tool:" if bound else "tool:") + call_id,
            receipt_id=digest(payload), kind="tool_outcome", trust="tool_receipt",
            source_time=datetime.now(UTC), payload=payload, source_payload_ids=dependencies)

    async def bound_tool_outcome(self, connection, call_id, name, result, sources,
                                 *, publication_area_id=None):
        event = self._event(call_id, name, result, sources, bound=True)
        scope = {"publication_area_id": publication_area_id} if publication_area_id is not None else {}
        return await self.repository.append_in_transaction(connection, self.owner_id, event, **scope)

    def tool_outcome(self, call_id, name, result, *, mutates_state=False):
        sources = image_source_payload_ids(result, owner_id=self.owner_id)
        safe = durable_tool_payload(result, status=str(result.get("status", "failed")))
        self._pending = self._event(call_id, name, safe, sources, bound=False,
                                    mutates_state=mutates_state)

    async def flush(self):
        if self._pending is not None:
            async with asyncio.timeout(self.write_timeout):
                await self.repository.append(self.owner_id, self._pending)
            self._pending = None
