from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


LOGGER = logging.getLogger(__name__)


class DomainEventName(str, Enum):
    VISUAL_FRAME_ACCEPTED = "visual_frame_accepted"
    PERCEPTION_OBSERVATION_COMPLETED = "perception_observation_completed"
    WORKING_MEMORY_UPDATED = "working_memory_updated"
    OWNER_WORKING_MEMORY_UPDATED = "owner_working_memory_updated"
    EXPERIENCE_EVENT_CLOSED = "experience_event_closed"
    MEMORY_ITEM_CHANGED = "memory_item_changed"


@dataclass(frozen=True)
class VisualFrameAcceptedPayload:
    owner_id: str
    session_id: UUID
    frame_id: UUID
    frame_sequence: int


@dataclass(frozen=True)
class PerceptionObservationCompletedPayload:
    owner_id: str
    session_id: UUID
    observation_id: UUID
    frame_id: UUID | None


@dataclass(frozen=True)
class WorkingMemoryUpdatedPayload:
    owner_id: str
    session_id: UUID
    revision: int


@dataclass(frozen=True)
class OwnerWorkingMemoryUpdatedPayload:
    owner_id: str
    revision: int


@dataclass(frozen=True)
class ExperienceEventClosedPayload:
    owner_id: str
    session_id: UUID
    experience_event_id: UUID
    started_at: datetime | None = None
    ended_at: datetime | None = None
    boundary_reason: str | None = None


@dataclass(frozen=True)
class MemoryItemChangedPayload:
    owner_id: str
    memory_item_id: UUID
    change_type: str
    session_id: UUID | None = None
    memory_type: str | None = None
    reason: str | None = None


DomainEventPayload = (
    VisualFrameAcceptedPayload
    | PerceptionObservationCompletedPayload
    | WorkingMemoryUpdatedPayload
    | OwnerWorkingMemoryUpdatedPayload
    | ExperienceEventClosedPayload
    | MemoryItemChangedPayload
)


EXPECTED_PAYLOAD_TYPES: Mapping[DomainEventName, type[Any]] = {
    DomainEventName.VISUAL_FRAME_ACCEPTED: VisualFrameAcceptedPayload,
    DomainEventName.PERCEPTION_OBSERVATION_COMPLETED: PerceptionObservationCompletedPayload,
    DomainEventName.WORKING_MEMORY_UPDATED: WorkingMemoryUpdatedPayload,
    DomainEventName.OWNER_WORKING_MEMORY_UPDATED: OwnerWorkingMemoryUpdatedPayload,
    DomainEventName.EXPERIENCE_EVENT_CLOSED: ExperienceEventClosedPayload,
    DomainEventName.MEMORY_ITEM_CHANGED: MemoryItemChangedPayload,
}


@dataclass(frozen=True)
class DomainEvent:
    name: DomainEventName
    payload: DomainEventPayload
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        expected_type = EXPECTED_PAYLOAD_TYPES[self.name]
        if not isinstance(self.payload, expected_type):
            raise TypeError(
                f"{self.name.value} requires payload type {expected_type.__name__}"
            )


DomainEventHandler = Callable[[DomainEvent], Awaitable[None]]


@dataclass(frozen=True)
class DomainEventDispatchResult:
    delivered_count: int
    failed_count: int


class DomainEventBus:
    """Best-effort, process-local event delivery for non-critical notifications.

    Events are not persisted and do not cross backend processes. Durable work must
    use Postgres-backed state or jobs rather than relying on this bus.
    """

    def __init__(self) -> None:
        self._handlers: dict[DomainEventName, list[DomainEventHandler]] = defaultdict(list)

    def subscribe(self, name: DomainEventName, handler: DomainEventHandler) -> None:
        handlers = self._handlers[name]
        if any(_same_handler(existing, handler) for existing in handlers):
            raise ValueError(f"Handler is already subscribed to {name.value}")
        handlers.append(handler)

    async def publish(self, event: DomainEvent) -> DomainEventDispatchResult:
        delivered_count = 0
        failed_count = 0
        for handler in tuple(self._handlers.get(event.name, ())):
            try:
                await handler(event)
            except Exception:
                failed_count += 1
                LOGGER.exception(
                    "domain event handler failed",
                    extra={
                        "event": "domain_event_handler_failed",
                        "domain_event_name": event.name.value,
                        "domain_event_id": str(event.event_id),
                    },
                )
                continue
            delivered_count += 1
        return DomainEventDispatchResult(
            delivered_count=delivered_count,
            failed_count=failed_count,
        )

    def unsubscribe(self, name: DomainEventName, handler: DomainEventHandler) -> bool:
        """Remove an owned lifecycle subscription without affecting other users."""
        handlers = self._handlers.get(name, ())
        for index, existing in enumerate(handlers):
            if _same_handler(existing, handler):
                del handlers[index]
                if not handlers:
                    self._handlers.pop(name, None)
                return True
        return False


def _same_handler(left: DomainEventHandler, right: DomainEventHandler) -> bool:
    left_instance = getattr(left, "__self__", None)
    right_instance = getattr(right, "__self__", None)
    left_function = getattr(left, "__func__", None)
    right_function = getattr(right, "__func__", None)
    if left_function is not None or right_function is not None:
        return left_instance is right_instance and left_function is right_function
    return left is right
