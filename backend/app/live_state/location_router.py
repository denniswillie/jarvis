"""Authenticated companion device location ingress into pull-only Live State RAM."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.live_state.companion import CompanionState


router = APIRouter(prefix="/live-state", tags=["live-state"])
PRIVATE_HEADERS = {
    "Cache-Control": "private, no-store",
    "Vary": "Authorization",
    "X-Content-Type-Options": "nosniff",
}


class LocationUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    horizontal_accuracy_m: float | None = Field(default=None, ge=0.0, le=100_000.0)
    place_name: str | None = Field(default=None, max_length=160)
    locality: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=120)
    timezone: str | None = Field(default=None, max_length=80)
    captured_at: AwareDatetime

    @field_validator("place_name", "locality", "country", "timezone")
    @classmethod
    def clean_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed if trimmed else None


def _get_companion_state(request: Request, principal: JarvisPrincipal) -> CompanionState:
    runtime = getattr(request.app.state, "runtime_services", None)
    state = getattr(runtime, "companion_state", None)
    if not isinstance(state, CompanionState):
        raise JarvisHTTPError(503, "live_state_unavailable", "Companion Live State is not active right now.")
    if state.owner_id != principal.owner_id:
        raise JarvisHTTPError(403, "live_state_owner_mismatch", "Location belongs to another account.")
    return state


@router.get("/location")
async def get_location(
    request: Request,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
) -> JSONResponse:
    state = _get_companion_state(request, principal)
    snapshot = state.read()
    return JSONResponse({"location": snapshot.get("location")}, headers=PRIVATE_HEADERS)


@router.put("/location")
async def put_location(
    body: LocationUpdateRequest,
    request: Request,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
) -> JSONResponse:
    state = _get_companion_state(request, principal)
    payload = {
        "latitude": round(body.latitude, 6),
        "longitude": round(body.longitude, 6),
    }
    if body.horizontal_accuracy_m is not None:
        payload["horizontal_accuracy_m"] = round(body.horizontal_accuracy_m, 1)
    for field_name in ("place_name", "locality", "country", "timezone"):
        value = getattr(body, field_name)
        if value is not None:
            payload[field_name] = value
    applied = await state.update_location(payload, captured_at=body.captured_at)
    snapshot = state.read()
    return JSONResponse(
        {
            "status": "applied" if applied else "superseded",
            "location": snapshot.get("location"),
        },
        headers=PRIVATE_HEADERS,
    )
