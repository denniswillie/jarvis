from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any, ClassVar, Protocol
from uuid import UUID

from app.assistant.context_providers import ContextBuildRequest, ContextSection
from app.perception.models import OwnerWorkingMemoryState, SessionWorkingMemoryState


MAX_CONTEXT_DEPTH = 5
MAX_CONTEXT_MAPPING_ITEMS = 32
MAX_CONTEXT_SEQUENCE_ITEMS = 20
MAX_CONTEXT_STRING_CHARACTERS = 1_000
_BINARY_FIELD_MARKERS = ("base64", "image_bytes", "jpeg_bytes", "raw_image")
LOGGER = logging.getLogger(__name__)


def _emit_read_diagnostic(request, *, started, deadline_seconds, stages, outcome, reason):
    session_id = request.session.id if request.session is not None else None
    record = {
        "outcome": outcome, "reason": reason,
        "elapsed_ms": round((monotonic() - started) * 1000, 3),
        "deadline_ms": deadline_seconds * 1000,
        "session_id": str(session_id) if isinstance(session_id, UUID) else None,
        "stages": stages,
        "timing_scope": "service_calls_including_pool_wait",
        "pool_acquire_ms": None, "sql_statement_ms": None,
    }
    level = logging.WARNING if outcome == "unavailable" and stages else logging.INFO
    if LOGGER.isEnabledFor(level):
        LOGGER.log(level, "working_memory_context_diagnostic %s", json.dumps(record, separators=(",", ":")))


class WorkingMemoryReader(Protocol):
    async def get_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> SessionWorkingMemoryState | None: ...

    async def get_owner_working_memory(
        self,
        *,
        owner_id: str,
    ) -> OwnerWorkingMemoryState | None: ...


@dataclass(frozen=True)
class WorkingMemoryContextProvider:
    """Expose current structured state without placing frame pixels in the prompt."""

    service: WorkingMemoryReader | None
    read_timeout_seconds: float = 0.35
    key: ClassVar[str] = "working_memory"
    priority: ClassVar[int] = 65
    critical: ClassVar[bool] = False

    async def build_section(self, request: ContextBuildRequest) -> ContextSection:
        started, stages = monotonic(), []
        outcome, reason = "unavailable", "read_failed"

        async def read(stage_name, method, **kwargs):
            read_started = monotonic()
            stage = {"stage": stage_name, "outcome": "failed"}
            stages.append(stage)
            try:
                state = await method(**kwargs)
                stage["outcome"] = "empty" if state is None else "available"
                revision = getattr(state, "revision", None)
                if type(revision) is int and revision >= 0:
                    stage["revision"] = revision
                return state
            except asyncio.CancelledError:
                stage["outcome"] = "cancelled"
                raise
            finally:
                stage["elapsed_ms"] = round((monotonic() - read_started) * 1000, 3)

        # The foreground already holds a bounded per-session context snapshot.
        # Refresh it without serial database waits or unbounded optional reads.
        # Do not substitute a stale cache after deletion/expiry or storage failure.
        try:
            if self.service is None or request.session is None:
                reason = "service_unconfigured" if self.service is None else "session_missing"
                return self.fallback_section(request)
            deadline = asyncio.timeout(self.read_timeout_seconds)
            try:
                async with deadline:
                    async with asyncio.TaskGroup() as reads:
                        session_read = reads.create_task(read("session_read", self.service.get_session_working_memory,
                            owner_id=request.owner_id, session_id=request.session.id))
                        owner_read = reads.create_task(read("owner_read", self.service.get_owner_working_memory,
                            owner_id=request.owner_id))
                session_state, owner_state = session_read.result(), owner_read.result()
            except TimeoutError:
                reason = "read_deadline_exceeded" if deadline.expired() else "operation_timeout"
                return self.fallback_section(request)
            payload = build_working_memory_payload(session_state, owner_state)
            freshness = _latest_datetime(
                request.requested_at_utc,
                session_state.updated_at if session_state is not None else None,
                owner_state.updated_at if owner_state is not None else None,
            )
            section = ContextSection(
                key=self.key,
                payload=payload,
                priority=self.priority,
                freshness=freshness,
                size=len(json.dumps(payload, ensure_ascii=True, separators=(",", ":"))),
            )
            outcome, reason = "available", "complete"
            return section
        except asyncio.CancelledError:
            outcome, reason = "cancelled", "caller_cancelled"
            raise
        finally:
            _emit_read_diagnostic(request, started=started, deadline_seconds=self.read_timeout_seconds,
                                  stages=stages, outcome=outcome, reason=reason)

    def fallback_section(self, request: ContextBuildRequest) -> ContextSection:
        return ContextSection(
            key=self.key,
            payload={"available": False, "session": None, "owner": None},
            priority=self.priority,
            freshness=request.requested_at_utc,
            size=0,
        )


def build_working_memory_payload(
    session_state: SessionWorkingMemoryState | None,
    owner_state: OwnerWorkingMemoryState | None,
) -> dict[str, Any]:
    return {
        "available": True,
        "session": (
            {
                "revision": session_state.revision,
                "observed_at": _iso_or_none(session_state.last_observation_ended_at),
                "updated_at": session_state.updated_at.isoformat(),
                "expires_at": session_state.expires_at.isoformat(),
                "state": _bounded_value(session_state.state),
            }
            if session_state is not None
            else None
        ),
        "owner": (
            {
                "revision": owner_state.revision,
                "updated_at": owner_state.updated_at.isoformat(),
                "expires_at": _iso_or_none(owner_state.expires_at),
                "state": _bounded_value(owner_state.state),
            }
            if owner_state is not None
            else None
        ),
    }


def session_working_memory_revision(payload: Mapping[str, Any] | None) -> int | None:
    session = payload.get("session") if isinstance(payload, Mapping) else None
    revision = session.get("revision") if isinstance(session, Mapping) else None
    return revision if isinstance(revision, int) and revision >= 0 else None


def owner_working_memory_revision(payload: Mapping[str, Any] | None) -> int | None:
    owner = payload.get("owner") if isinstance(payload, Mapping) else None
    revision = owner.get("revision") if isinstance(owner, Mapping) else None
    return revision if isinstance(revision, int) and revision >= 0 else None


def working_memory_context_available(payload: Mapping[str, Any] | None) -> bool:
    return isinstance(payload, Mapping) and payload.get("available") is True


def working_memory_next_expires_at(
    payload: Mapping[str, Any] | None,
) -> datetime | None:
    expirations: list[datetime] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key == "expires_at" and isinstance(item, str):
                    try:
                        parsed = datetime.fromisoformat(item.replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if parsed.tzinfo is not None:
                        expirations.append(parsed)
                else:
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(payload)
    return min(expirations) if expirations else None


def working_memory_app_event_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    session = payload.get("session")
    session_payload = session if isinstance(session, Mapping) else {}
    owner = payload.get("owner")
    state = session_payload.get("state")
    state_payload = state if isinstance(state, Mapping) else {}
    return {
        "revision": session_working_memory_revision(payload),
        "owner_revision": owner_working_memory_revision(payload),
        "session_available": isinstance(session, Mapping),
        "owner_available": isinstance(owner, Mapping),
        "observed_at": session_payload.get("observed_at"),
        "expires_at": session_payload.get("expires_at"),
        "scene_summary": _state_label(state_payload.get("scene")),
        "activity_summary": _state_label(state_payload.get("activity")),
    }


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_CONTEXT_DEPTH:
        return "[bounded]"
    if isinstance(value, Mapping):
        bounded: dict[str, Any] = {}
        for key, item in list(value.items())[:MAX_CONTEXT_MAPPING_ITEMS]:
            key_text = str(key)
            normalized_key = key_text.lower()
            if any(marker in normalized_key for marker in _BINARY_FIELD_MARKERS):
                continue
            bounded[key_text] = _bounded_value(item, depth=depth + 1)
        return bounded
    if isinstance(value, (list, tuple)):
        return [
            _bounded_value(item, depth=depth + 1)
            for item in value[:MAX_CONTEXT_SEQUENCE_ITEMS]
        ]
    if isinstance(value, str):
        return value[:MAX_CONTEXT_STRING_CHARACTERS]
    if isinstance(value, bytes):
        return "[binary omitted]"
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:MAX_CONTEXT_STRING_CHARACTERS]


def _state_label(value: Any) -> str | None:
    if isinstance(value, Mapping):
        label = value.get("label")
        return label if isinstance(label, str) and label.strip() else None
    return value if isinstance(value, str) and value.strip() else None


def _latest_datetime(fallback: datetime, *values: datetime | None) -> datetime:
    available = [value for value in values if value is not None]
    return max(available) if available else fallback


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
