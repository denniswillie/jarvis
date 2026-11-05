"""Session-local MSSOI reads. No database, model, authorization grant or writer."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
import time
from uuid import UUID, uuid4

from app.live_state.models import LiveSnapshot
from app.live_state.projection import select_voice_context
from app.tools.registry import ToolDefinition


@dataclass(frozen=True)
class _Page:
    scope: object
    epoch: UUID
    focus: object
    entries: tuple
    dependencies: tuple
    deadline: float


class LiveBriefingReader:
    """An authenticated session's two-view, 30-second optional paging window."""

    def __init__(self, *, owner_id, session_id, snapshot,
                 clock=lambda: datetime.now(UTC), monotonic=time.monotonic):
        self.owner_id, self.session_id = owner_id, session_id
        self.snapshot, self.clock, self.monotonic = snapshot, clock, monotonic
        self._pages: OrderedDict[str, _Page] = OrderedDict()

    def tool(self):
        return ToolDefinition(
            name="live_briefing",
            description=("Read the useful current Live State briefing: observations, selected "
                         "memories and task facts with freshness and uncertainty. Optional; "
                         "do not call for ordinary conversation or to authorize actions."),
            input_schema={"type": "object", "properties": {
                "cursor": {"type": "string", "minLength": 1, "maxLength": 128}},
                "additionalProperties": False},
            handler=self.read, local_snapshot_read=True)

    @staticmethod
    def _unavailable(code, summary):
        return {"status": "success", "outcome": "unavailable", "effect_committed": False,
                "summary": summary, "error": {"code": code, "retryable": True}}

    async def read(self, arguments):
        if (not isinstance(arguments, dict) or set(arguments) - {"cursor"}
                or ("cursor" in arguments and (not isinstance(arguments["cursor"], str)
                    or not 1 <= len(arguments["cursor"]) <= 128))):
            return {"status": "failed", "outcome": "rejected", "effect_committed": False,
                    "summary": "Use only an optional briefing cursor.",
                    "error": {"code": "invalid_arguments", "retryable": False}}
        now, tick = self.clock(), self.monotonic()
        self._pages = OrderedDict((key, page) for key, page in self._pages.items()
                                  if page.deadline > tick)
        snapshot = self.snapshot()
        if (not isinstance(snapshot, LiveSnapshot)
                or (snapshot.scope.owner_id, snapshot.scope.session_id)
                    != (self.owner_id, self.session_id)):
            self._pages.clear()
            return self._unavailable("live_state_unavailable",
                "Live State is unavailable. Current visual evidence and background facts are unknown.")

        dependencies = ()
        deadline = tick + 30.0
        entries = snapshot.entries
        cursor = arguments.get("cursor")
        if cursor is not None:
            page = self._pages.get(cursor)
            if (page is None or page.scope != snapshot.scope
                    or page.epoch != snapshot.cursor.epoch
                    or page.focus != snapshot.conversation.focus):
                return self._unavailable("cursor_expired", "That briefing view expired; request a new first page if useful.")
            current = {entry.entry_id: entry for entry in snapshot.entries}
            # A page retains identities, not permission to resurrect old content.
            entries = tuple(entry for entry in page.entries if current.get(entry.entry_id) == entry)
            dependencies, deadline = page.dependencies, page.deadline

        def usable(entry):
            if (entry.validity != "active" or entry.guard_binding is None
                    or (entry.observed_at is not None and entry.observed_at > now)
                    or any(value is not None and value <= now
                           for value in (entry.fresh_until, entry.expires_at))):
                return False
            return all(not (old.source_namespace == source.source_namespace
                           and old.source_id == source.source_id and old != source)
                       for source in entry.sources for old in dependencies)

        entries = tuple(entry for entry in entries if usable(entry))
        view = snapshot.model_copy(update={"entries": entries})
        selection = select_voice_context(view, now=now, max_chars=4096,
                                         max_entries=8, max_encoded_bytes=32768)
        accepted = tuple(dict.fromkeys((*dependencies, *selection.dependencies)))
        remaining = tuple(entry for entry in entries
            if entry.entry_id not in selection.selected_entry_ids
            and all(not (old.source_namespace == source.source_namespace
                         and old.source_id == source.source_id and old != source)
                    for source in entry.sources for old in accepted))
        next_cursor = None
        # No endless cursor chain for a view in which nothing useful fits.
        if snapshot.readiness == "ready" and selection.selected_entry_ids and remaining:
            next_cursor = uuid4().hex
            self._pages[next_cursor] = _Page(snapshot.scope, snapshot.cursor.epoch,
                snapshot.conversation.focus, remaining, accepted, deadline)
            while len(self._pages) > 2:
                self._pages.popitem(last=False)
        briefing = {"text": selection.text, "generated_at": now.isoformat(),
                    "readiness": snapshot.readiness, "more_available": next_cursor is not None}
        if next_cursor is not None:
            briefing["next_cursor"] = next_cursor
        return {"status": "success",
                "outcome": "available" if snapshot.readiness == "ready" else "unavailable",
                "effect_committed": False, "summary": "A bounded, partial Live State view.",
                "briefing": briefing}
