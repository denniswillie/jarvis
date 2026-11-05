"""Owner enrollment and scoped outbound connector endpoints."""
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.knowledge.models import SourceError
from app.projects.models import Completion, Enrollment, Inspection, MAX_BODY_BYTES, Poll


router = APIRouter(prefix="/projects", tags=["projects"])


def get_project_service(request: Request):
    runtime = getattr(request.app.state, "runtime_services", None)
    knowledge = getattr(runtime, "knowledge_runtime", None)
    if knowledge is None:
        raise SourceError("project_service_unavailable", "Source-backed project connections are disabled.", 503)
    return knowledge.projects


Service = Annotated[object, Depends(get_project_service)]
Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]


def connector_token(request: Request):
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer ") or len(header) > 200:
        raise SourceError("project_auth_invalid", "A scoped connector credential is required.", 401)
    return header[7:]


async def body(request, model):
    parts, size = [], 0
    async for part in request.stream():
        size += len(part)
        if size > MAX_BODY_BYTES:
            raise SourceError("project_body_too_large", "The connector response exceeds its bounded envelope.", 413)
        parts.append(part)
    try:
        return model.model_validate_json(b"".join(parts))
    except (ValidationError, ValueError, TypeError):
        raise SourceError("project_request_invalid", "Invalid bounded connector envelope.", 422) from None


@router.post("/connectors")
async def enroll(request: Request, principal: Principal, service: Service):
    return await service.enroll(principal.owner_id, await body(request, Enrollment))


@router.delete("/connectors/{connector_id}")
async def revoke(connector_id: UUID, principal: Principal, service: Service):
    return await service.revoke(principal.owner_id, connector_id)


@router.get("/areas/{area_id}/roots")
async def roots(area_id: UUID, principal: Principal, service: Service):
    return await service.roots(principal.owner_id, area_id)


@router.post("/areas/{area_id}/requests")
async def submit(area_id: UUID, request: Request, principal: Principal, service: Service):
    return await service.submit(principal.owner_id, area_id, await body(request, Inspection))


@router.get("/areas/{area_id}/requests/{request_id}")
async def result(area_id: UUID, request_id: UUID, principal: Principal, service: Service):
    return await service.result(principal.owner_id, area_id, request_id)


@router.post("/connector/poll")
async def poll(request: Request, service: Service):
    return await service.poll(connector_token(request), await body(request, Poll))


@router.post("/connector/complete")
async def complete(request: Request, service: Service):
    return await service.complete(connector_token(request), await body(request, Completion))
