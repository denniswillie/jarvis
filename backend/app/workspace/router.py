"""Authenticated Workspace reads and phone acknowledgements only.

Retained files and sandbox presentations use their independently owned services.
Core installs the services in runtime_services when their dependencies are ready.
"""
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.workspace.models import WorkspaceContext
from app.workspace.presentation import PresentationAck


router = APIRouter(prefix="/workspace", tags=["workspace"])
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
PRIVATE_HEADERS = {"Cache-Control": "private, no-store", "Vary": "Authorization",
                   "X-Content-Type-Options": "nosniff"}


def _service(request, principal, name):
    runtime = getattr(request.app.state, "runtime_services", None)
    settings = getattr(runtime, "settings", None)
    service = getattr(runtime, name, None)
    if service is None or getattr(settings, "JARVIS_OWNER_ID", None) != principal.owner_id:
        raise JarvisHTTPError(503, "workspace_unavailable", "Workspace is not ready in this backend.")
    return service


def _response(value):
    return JSONResponse(value, headers=PRIVATE_HEADERS)


@router.get("/requests")
async def presentation_requests(request: Request, principal: Principal, session_id: UUID):
    return _response(await _service(request, principal, "workspace_presentations").list_requests(
        WorkspaceContext(principal.owner_id, session_id)))


@router.post("/requests/{request_id}/ack")
async def acknowledge_presentation(request_id: UUID, body: PresentationAck, request: Request, principal: Principal):
    return _response(await _service(request, principal, "workspace_presentations").acknowledge(
        WorkspaceContext(principal.owner_id, body.session_id), request_id, body))
