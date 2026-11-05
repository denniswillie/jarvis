"""Authenticated Preferences API. Runtime composition is owned by Core."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.assistant.models import BehaviorProfileRevisionConflict, BehaviorProfileUpdate
from app.assistant.profile_service import BehaviorProfileService
from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError


router = APIRouter(prefix="/assistant", tags=["preferences"])
PRIVATE_HEADERS = {
    "Cache-Control": "private, no-store",
    "Vary": "Authorization",
    "X-Content-Type-Options": "nosniff",
}


def get_behavior_profile_service(request: Request) -> BehaviorProfileService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "behavior_profile_service", None)
    if not isinstance(service, BehaviorProfileService):
        raise JarvisHTTPError(
            503, "behavior_profile_unavailable", "Persistent preferences are not ready."
        )
    return service


@router.get("/profile")
async def get_profile(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[BehaviorProfileService, Depends(get_behavior_profile_service)],
) -> JSONResponse:
    return JSONResponse(await service.get(owner_id=principal.owner_id), headers=PRIVATE_HEADERS)


@router.put("/profile")
async def put_profile(
    update: BehaviorProfileUpdate,
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    service: Annotated[BehaviorProfileService, Depends(get_behavior_profile_service)],
) -> JSONResponse:
    try:
        profile = await service.update(owner_id=principal.owner_id, update=update)
    except BehaviorProfileRevisionConflict as error:
        return JSONResponse({"error": error.error_payload()}, status_code=409, headers=PRIVATE_HEADERS)
    return JSONResponse(profile, headers=PRIVATE_HEADERS)
