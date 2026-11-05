from __future__ import annotations

from typing import Annotated

import asyncpg
from fastapi import APIRouter, Depends, Query

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.db import get_pool
from app.reminders.models import (
    ActiveReminderResponse,
    ActiveRemindersResponse,
    ReminderEventResponse,
    ReminderEventsResponse,
)
from app.reminders.repository import ReminderRepository


router = APIRouter(prefix="/reminders", tags=["reminders"])


def get_reminder_repository(pool: Annotated[asyncpg.Pool, Depends(get_pool)]) -> ReminderRepository:
    return ReminderRepository(pool)


@router.get("/active", response_model=ActiveRemindersResponse)
async def get_active_reminders(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    repository: Annotated[ReminderRepository, Depends(get_reminder_repository)],
) -> ActiveRemindersResponse:
    reminders = await repository.list_active_reminders(owner_id=principal.owner_id)
    return ActiveRemindersResponse(
        reminders=[
            ActiveReminderResponse(
                id=reminder.id,
                title=reminder.title,
                details=reminder.details,
                remind_at=reminder.remind_at,
                condition=reminder.condition,
                status=reminder.status,
                updated_at=reminder.updated_at,
            )
            for reminder in reminders
        ]
    )


@router.get("/events", response_model=ReminderEventsResponse)
async def get_reminder_events(
    principal: Annotated[JarvisPrincipal, Depends(require_static_bearer)],
    repository: Annotated[ReminderRepository, Depends(get_reminder_repository)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> ReminderEventsResponse:
    events = await repository.list_events(owner_id=principal.owner_id, limit=limit)
    return ReminderEventsResponse(
        events=[
            ReminderEventResponse(
                id=event.id,
                reminder_id=event.reminder_id,
                session_id=event.session_id,
                event_type=event.event_type,
                message=event.message,
                provider_response_id=event.provider_response_id,
                created_at=event.created_at,
            )
            for event in events
        ]
    )
