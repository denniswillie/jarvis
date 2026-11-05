from __future__ import annotations

import base64
import binascii
import json
from datetime import UTC, datetime
from typing import Annotated, cast
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from app.knowledge.browser import SourceMemoryBrowser
from app.runtime.feature_config import MemoryAuthority

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.memory.models import MEMORY_TYPES, MemoryItem, MemoryType
from app.memory.presentation import (
    evidence_payload,
    explanation_payload,
    memory_item_payload,
    working_state_payload,
)
from app.memory.projection import MemoryProjectionService
from app.memory.repository import MemoryMissingError
from app.memory.service import MemoryApplicationService


router = APIRouter(prefix="/memory", tags=["memory"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]


@router.get("/recent")
async def get_recent_memory(
    request: Request,
    principal: Principal,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
    memory_types: Annotated[list[str] | None, Query()] = None,
    source_types: Annotated[list[str] | None, Query()] = None,
    session_id: UUID | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    cursor: str | None = None,
) -> dict[str, object]:
    if _uses_source_memory(request):
        if memory_types:
            raise JarvisHTTPError(400, "invalid_memory_filter",
                "Source timelines use source_types, not legacy inferred memory classifications.")
        try:
            payload = await _source_browser(request, principal.owner_id).recent(
                owner_id=principal.owner_id, limit=limit, source_types=source_types or (),
                session_id=session_id, start_time=_utc_or_none(start_time, "start_time"),
                end_time=_utc_or_none(end_time, "end_time"), cursor=cursor,
            )
        except ValueError as error:
            raise _validation_error(error) from error
        return _source_response(payload)
    if source_types:
        raise JarvisHTTPError(400, "invalid_memory_filter", "Source filters require source-backed memory.")
    service = _memory_service(request)
    normalized_types = _memory_types(memory_types or [])
    cursor_time, cursor_id = _decode_cursor(cursor)
    try:
        items = await service.list_recent(
            owner_id=principal.owner_id,
            limit=limit + 1,
            memory_types=normalized_types,
            session_id=session_id,
            start_time=_utc_or_none(start_time, "start_time"),
            end_time=_utc_or_none(end_time, "end_time"),
            cursor_time=cursor_time,
            cursor_id=cursor_id,
        )
    except ValueError as error:
        raise _validation_error(error) from error
    page = items[:limit]
    return {
        "items": [memory_item_payload(item) for item in page],
        "next_cursor": (
            _encode_memory_cursor(page[-1])
            if len(items) > limit and page
            else None
        ),
    }


@router.get("/working-state")
async def get_working_memory_state(
    request: Request,
    principal: Principal,
    session_id: UUID | None = None,
) -> dict[str, object]:
    if _uses_source_memory(request):
        runtime = _selected_runtime(request, principal.owner_id)
        service = runtime.working_memory_service
        if service is None:
            raise JarvisHTTPError(503, "working_memory_not_configured", "Current sensory state is unavailable.")
        session_state = (await service.get_session_working_memory(
            owner_id=principal.owner_id, session_id=session_id)) if session_id is not None else None
        owner_state = await service.get_owner_working_memory(owner_id=principal.owner_id)
        return _source_response(working_state_payload(session_state, owner_state))
    session_state, owner_state = await _memory_service(request).working_state(
        owner_id=principal.owner_id,
        session_id=session_id,
    )
    return working_state_payload(session_state, owner_state)


@router.get("/profile")
async def get_memory_profile(
    request: Request,
    principal: Principal,
) -> dict[str, object]:
    if _uses_source_memory(request):
        _selected_runtime(request, principal.owner_id)
        return _source_response({
            "profile": None, "authority": "hindsight", "status": "not_materialized",
            "meaning": "No generated profile is maintained. Use source-backed recall and originals; null does not mean nothing is known.",
            "provider_contacted": False,
        })
    profile = await _memory_projection_service(request).get_profile(
        owner_id=principal.owner_id
    )
    return {"profile": profile.as_payload()}


@router.get("/{memory_id}/evidence")
async def get_memory_evidence(
    memory_id: UUID,
    request: Request,
    principal: Principal,
    limit: Annotated[int, Query(ge=1, le=100)] = 100,
    revision: Annotated[int | None, Query(ge=1)] = None,
) -> dict[str, object]:
    if _uses_source_memory(request):
        detail = await _source_browser(request, principal.owner_id).detail(
            owner_id=principal.owner_id, memory_id=memory_id, revision=revision)
        return _source_response({"items": detail["evidence"][:limit], "next_cursor": None,
            "authority": "hindsight", "meaning": detail["uncertainty"]})
    if revision is not None:
        raise JarvisHTTPError(400, "invalid_memory_request", "Source revisions require source-backed memory.")
    try:
        evidence = await _memory_service(request).evidence(
            owner_id=principal.owner_id,
            memory_id=memory_id,
            limit=limit,
        )
    except MemoryMissingError as error:
        raise _not_found() from error
    return {
        "items": [evidence_payload(item) for item in evidence],
        "next_cursor": None,
    }


@router.get("/{memory_id}")
async def get_memory_detail(
    memory_id: UUID,
    request: Request,
    principal: Principal,
    revision: Annotated[int | None, Query(ge=1)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    expected_digest: Annotated[str | None, Query(max_length=64)] = None,
) -> dict[str, object]:
    if _uses_source_memory(request):
        try:
            detail = await _source_browser(request, principal.owner_id).detail(
                owner_id=principal.owner_id, memory_id=memory_id, revision=revision,
                offset=offset, expected_digest=expected_digest)
        except ValueError as error:
            raise _validation_error(error) from error
        return _source_response(detail)
    if revision is not None or offset or expected_digest is not None:
        raise JarvisHTTPError(400, "invalid_memory_request", "Source pagination requires source-backed memory.")
    try:
        explanation = await _memory_service(request).explain(
            owner_id=principal.owner_id,
            memory_id=memory_id,
        )
    except MemoryMissingError as error:
        raise _not_found() from error
    return explanation_payload(explanation)


def _uses_source_memory(request: Request) -> bool:
    runtime = getattr(request.app.state, "runtime_services", None)
    authority = getattr(runtime, "memory_authority", MemoryAuthority.LEGACY)
    if authority not in {MemoryAuthority.LEGACY, MemoryAuthority.HINDSIGHT}:
        raise JarvisHTTPError(503, "memory_authority_unavailable", "Memory authority is invalid.")
    return authority == MemoryAuthority.HINDSIGHT


def _selected_runtime(request: Request, owner_id: str):
    runtime = request.app.state.runtime_services
    if runtime.settings.JARVIS_OWNER_ID != owner_id:
        raise JarvisHTTPError(403, "knowledge_owner_mismatch", "Memory is outside this runtime owner scope.")
    return runtime


def _source_browser(request: Request, owner_id: str) -> SourceMemoryBrowser:
    services = _selected_runtime(request, owner_id)
    runtime = services.knowledge_runtime
    if runtime is None:
        raise JarvisHTTPError(503, "selected_memory_unavailable",
            "Selected source-backed memory is unavailable; legacy memory is not a fallback.")
    return SourceMemoryBrowser(
        runtime.sources, runtime.memory_read_backend_for(owner_id), owner_id=owner_id,
        area_id=runtime.configuration.area_id, access_check=runtime.context_for,
    )


def _source_response(payload):
    return JSONResponse(payload, headers={"Cache-Control": "no-store", "Vary": "Authorization"})


def _memory_service(request: Request) -> MemoryApplicationService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "memory_service", None)
    if service is None:
        raise JarvisHTTPError(
            503,
            "memory_not_configured",
            "Jarvis memory is not configured.",
        )
    return cast(MemoryApplicationService, service)


def _memory_projection_service(request: Request) -> MemoryProjectionService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "memory_projection_service", None)
    if service is None:
        raise JarvisHTTPError(
            503,
            "memory_profile_not_configured",
            "Jarvis memory profile projection is not configured.",
        )
    return cast(MemoryProjectionService, service)


def _memory_types(values: list[str]) -> tuple[MemoryType, ...]:
    normalized = tuple(dict.fromkeys(value.strip() for value in values if value.strip()))
    unsupported = set(normalized) - MEMORY_TYPES
    if unsupported:
        raise JarvisHTTPError(
            400,
            "invalid_memory_filter",
            "Unsupported memory type filter.",
            details={"unsupported": sorted(unsupported)},
        )
    return cast(tuple[MemoryType, ...], normalized)


def _utc_or_none(value: datetime | None, label: str) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise JarvisHTTPError(
            400,
            "invalid_memory_time",
            f"{label} must include a timezone.",
        )
    return value.astimezone(UTC)


def _encode_memory_cursor(item: MemoryItem) -> str:
    timestamp = item.occurred_at or item.created_at
    raw = json.dumps(
        {"time": timestamp.isoformat(), "id": str(item.id)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> tuple[datetime | None, UUID | None]:
    if value is None:
        return None, None
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        timestamp = datetime.fromisoformat(str(payload["time"]).replace("Z", "+00:00"))
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError
        return timestamp.astimezone(UTC), UUID(str(payload["id"]))
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        raise JarvisHTTPError(
            400,
            "invalid_memory_cursor",
            "Memory cursor is invalid.",
        ) from error


def _not_found() -> JarvisHTTPError:
    return JarvisHTTPError(404, "memory_not_found", "Memory was not found.")


def _validation_error(error: ValueError) -> JarvisHTTPError:
    return JarvisHTTPError(400, "invalid_memory_request", str(error))
