"""Optional foreground visibility into durable tasks and canonical evidence.

This is a bounded source index, not a second conversation or memory engine.
Native dialogue and existing assistant-context providers retain their current
semantics; original archived contents remain available to task-bound readers.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
from typing import ClassVar

from app.assistant.context_providers import ContextBuildRequest, ContextSection
from app.tasks.context import encoded_size, source_index
from app.tasks.context_repository import ContextRepository


class SessionTaskContextService:
    def __init__(self, repository: ContextRepository, *, max_tasks=20,
                 max_sources=16, max_characters=12000, timeout_seconds=1.5):
        if (type(max_tasks) is not int or not 1 <= max_tasks <= 100
                or type(max_sources) is not int or not 1 <= max_sources <= 128
                or type(max_characters) is not int or not 2048 <= max_characters <= 64000
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("invalid session task context bounds")
        self.repository = repository
        self.max_tasks, self.max_sources = max_tasks, max_sources
        self.max_characters, self.timeout_seconds = max_characters, timeout_seconds

    async def describe(self, request: ContextBuildRequest) -> dict:
        session_id = request.session.id if request.session is not None else None
        async with asyncio.timeout(self.timeout_seconds):
            if session_id is None:
                tasks = await self.repository.tasks.context_summaries(request.owner_id, limit=self.max_tasks)
                snapshot = None
            else:
                tasks, snapshot = await asyncio.gather(
                    self.repository.tasks.context_summaries(request.owner_id, limit=self.max_tasks),
                    self.repository.snapshot(
                        request.owner_id, session_id, max_messages=self.max_sources,
                        max_pins=min(8, self.max_sources),
                    ),
                )
        total_sources = snapshot["total"] if snapshot is not None else 0
        selected_tasks, terminal_tasks, references = [], [], []

        def assemble():
            return {
                "status": "available", "retrieved_at": request.requested_at_utc.isoformat(),
                "presentation": "silent_task_snapshot",
                "active_tasks": list(selected_tasks),
                "active_task_count": tasks["total"],
                "omitted_active_task_count": tasks["total"] - len(selected_tasks),
                "recent_terminal_tasks": list(terminal_tasks),
                "terminal_task_count": tasks["terminal_total"],
                "omitted_terminal_task_count": tasks["terminal_total"] - len(terminal_tasks),
                "canonical_history": {
                    "status": "available" if snapshot is not None else "no_session",
                    "session_id": str(session_id) if session_id is not None else None,
                    "watermark": snapshot["watermark"] if snapshot is not None else None,
                    "current_source_count": total_sources,
                    "source_references": list(references),
                    "unindexed_source_count": total_sources - len(references),
                    "content_included": False,
                    "archive_scope": "admitted_task_manifest",
                },
                "meaning": [
                    "Silent task snapshot, not an instruction to speak or a delivery receipt.",
                    "Do not append routine task progress to unrelated replies; use a user request or authorized attention event.",
                    "Use exact task_ref and revision with task tools; titles are not identifiers.",
                    "Current status supersedes older receipts and milestones; cancelled/failed does not mean a successful answer.",
                    "Recent terminal tasks are bounded, newest first. If a requested task is omitted, look it up; absence is not pending or nonexistent.",
                    "Task tools expose status, not expert prose. Results need authorized delivery by the selected voice route.",
                    "Concurrent task titles do not transfer their permissions to this conversation.",
                    "Canonical references are evidence indexes, not repeated user requests.",
                    "Unfinalized transcripts and unconfirmed playback remain unknown.",
                    "Archive contents and original images require authorized task-context readers.",
                ],
            }

        def foreground_summary(task):
            summary = dict(task)
            if "pending_question" in summary:
                question = summary.pop("pending_question")
                summary.setdefault("pending_question_available", bool(question))
            return summary

        for task in tasks["tasks"]:
            selected_tasks.append(foreground_summary(task))
            if encoded_size(assemble()) > self.max_characters:
                selected_tasks.pop()
        for task in tasks["recent_terminal_tasks"]:
            if len(selected_tasks) + len(terminal_tasks) >= self.max_tasks:
                break
            terminal_tasks.append(foreground_summary(task))
            if encoded_size(assemble()) > self.max_characters:
                terminal_tasks.pop()
        if snapshot is not None:
            pinned_ids = {record.id for record in snapshot["pinned"]}
            ordered = [*snapshot["pinned"], *(
                record for record in snapshot["recent"] if record.id not in pinned_ids
            )]
            for record in ordered:
                index = source_index(record)
                if record.kind == "assistant_output":
                    index["heard_state"] = "unknown"
                references.append(index)
                if encoded_size(assemble()) > self.max_characters:
                    references.pop()
        return assemble()


@dataclass(frozen=True)
class SessionTaskContextProvider:
    service: SessionTaskContextService
    key: ClassVar[str] = "second_brain"
    priority: ClassVar[int] = 100
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        payload = await self.service.describe(request)
        return ContextSection(self.key, payload, self.priority, request.requested_at_utc, encoded_size(payload))

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        payload = {
            "status": "unavailable", "reason": "task_context_unavailable",
            "meaning": "Task and canonical history state could not be read; do not infer there are no pending tasks.",
        }
        return ContextSection(self.key, payload, self.priority, request.requested_at_utc, encoded_size(payload))
