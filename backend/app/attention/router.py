from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from app.attention.models import AttentionError, InboxMutation
from app.attention.push_models import DeviceMutation, DeviceRegistration
from app.attention.service import AttentionService
from app.auth.static_bearer import JarvisPrincipal, require_static_bearer


router = APIRouter(prefix="/attention", tags=["attention"])


def get_attention_service(request: Request):
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "attention_service", None)
    if service is None:
        raise AttentionError(getattr(runtime, "attention_issue", None) or "attention_disabled",
                             "The durable inbox is unavailable in this backend process.", 503)
    return service


Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
Service = Annotated[AttentionService, Depends(get_attention_service)]


@router.get("")
async def inbox(principal: Principal, service: Service,
                limit: Annotated[int, Query(ge=1, le=100)] = 20,
                offset: Annotated[int, Query(ge=0, le=100000)] = 0,
                include_dismissed: bool = False, task_id: UUID | None = None):
    return await service.repository.list(principal.owner_id, limit=limit, offset=offset,
                                         include_dismissed=include_dismissed, task_id=task_id)


@router.get("/push-status")
async def phone_status(principal: Principal, service: Service):
    return service.phone_status()


def push_service(service):
    if service.push is None:
        raise AttentionError(service.push_issue or "push_unavailable",
                             "Phone notification registration is unavailable in this backend.", 503)
    return service.push


@router.get("/devices")
async def devices(principal: Principal, service: Service):
    return await push_service(service).repository.devices(principal.owner_id)


@router.put("/devices/{device_id}")
async def register_device(device_id: UUID, body: DeviceRegistration, principal: Principal, service: Service):
    return await push_service(service).repository.change_device(principal.owner_id, device_id, body)


@router.post("/devices/{device_id}/revoke")
async def revoke_device(device_id: UUID, body: DeviceMutation, principal: Principal, service: Service):
    return await push_service(service).repository.change_device(principal.owner_id, device_id, body, revoke=True)


@router.get("/{intent_id}")
async def detail(intent_id: UUID, principal: Principal, service: Service):
    return await service.repository.detail(principal.owner_id, intent_id)


@router.post("/{intent_id}/dismiss")
async def dismiss(intent_id: UUID, body: InboxMutation, principal: Principal, service: Service):
    return await service.repository.mutate(principal.owner_id, intent_id, "dismiss", body)


@router.post("/{intent_id}/replay")
async def replay(intent_id: UUID, body: InboxMutation, principal: Principal, service: Service):
    return await service.repository.mutate(principal.owner_id, intent_id, "replay", body)
