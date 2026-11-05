from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.tasks.models import TaskCreate, TaskError, TaskMutation, TaskView
from app.tasks.service import TaskService


router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_task_service(request: Request) -> TaskService:
    runtime = getattr(request.app.state, "runtime_services", None)
    service = getattr(runtime, "background_task_service", None)
    if service is None:
        issue = getattr(runtime, "background_task_issue", None) or "tasks_disabled"
        raise TaskError(issue, "Background tasks are unavailable in this backend process.", 503)
    return service


Principal = Annotated[JarvisPrincipal, Depends(require_static_bearer)]
Service = Annotated[TaskService, Depends(get_task_service)]


@router.post("", response_model=TaskView)
async def create_task(body: TaskCreate, principal: Principal, service: Service):
    return await service.create(principal.owner_id, body)


@router.get("", response_model=list[TaskView])
async def list_tasks(principal: Principal, service: Service, limit: Annotated[int, Query(ge=1, le=100)] = 20,
                     offset: Annotated[int, Query(ge=0, le=100_000)] = 0, active_only: bool = False):
    return await service.repository.list(principal.owner_id, limit=limit, offset=offset, active_only=active_only)


@router.get("/providers")
async def provider_status(principal: Principal, service: Service):
    return await service.provider_status(principal.owner_id)


@router.get("/{task_id}", response_model=TaskView)
async def get_task(task_id: UUID, principal: Principal, service: Service):
    return await service.repository.get(principal.owner_id, task_id)


@router.get("/{task_id}/events")
async def get_events(task_id: UUID, principal: Principal, service: Service,
                     after: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 100):
    return await service.repository.events(principal.owner_id, task_id, after=after, limit=limit)


@router.post("/{task_id}/updates", response_model=TaskView)
async def update_task(task_id: UUID, body: TaskMutation, principal: Principal, service: Service):
    return await service.mutate(principal.owner_id, task_id, "update", body)


@router.post("/{task_id}/cancel", response_model=TaskView)
async def cancel_task(task_id: UUID, body: TaskMutation, principal: Principal, service: Service):
    return await service.mutate(principal.owner_id, task_id, "cancel", body)


@router.post("/{task_id}/resume", response_model=TaskView)
async def resume_task(task_id: UUID, body: TaskMutation, principal: Principal, service: Service):
    return await service.mutate(principal.owner_id, task_id, "resume", body)
