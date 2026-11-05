from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol
from uuid import UUID

from app.perception.models import (
    OwnerWorkingMemoryState,
    SessionWorkingMemoryState,
    VisualObservation,
    WorkingMemoryWriteResult,
)
from app.perception.repository import WorkingMemoryTargetMissingError
from app.perception.working_memory import (
    WorkingMemoryPolicy,
    normalize_owner_working_memory,
    owner_working_memory_expires_at,
    prune_session_working_memory,
    reduce_session_working_memory,
)
from app.runtime.events import (
    DomainEvent,
    DomainEventBus,
    DomainEventName,
    OwnerWorkingMemoryUpdatedPayload,
    WorkingMemoryUpdatedPayload,
)


class WorkingMemoryRepository(Protocol):
    async def get_session_working_memory(
        self, *, owner_id: str, session_id: UUID
    ) -> SessionWorkingMemoryState | None: ...

    async def compare_and_swap_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        expected_revision: int,
        observation_id: UUID,
        state: dict[str, object],
        expires_at: datetime,
        contract_version: int = 1,
        effect_guard=None,
    ) -> WorkingMemoryWriteResult: ...

    async def set_latest_experience_event_id(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
        expires_at: datetime,
    ) -> SessionWorkingMemoryState | None: ...

    async def get_owner_working_memory(
        self, *, owner_id: str
    ) -> OwnerWorkingMemoryState | None: ...

    async def compare_and_swap_owner_working_memory(
        self,
        *,
        owner_id: str,
        expected_revision: int,
        state: Mapping[str, Any],
        expires_at: datetime | None = None,
        contract_version: int = 1,
    ) -> WorkingMemoryWriteResult: ...


class SessionWorkingMemoryReader(Protocol):
    async def get_session_working_memory(
        self, *, owner_id: str, session_id: UUID
    ) -> SessionWorkingMemoryState | None: ...


@dataclass(frozen=True)
class WorkingMemoryApplyResult:
    applied: bool
    state: SessionWorkingMemoryState | None
    reason: str


@dataclass(frozen=True)
class OwnerWorkingMemoryApplyResult:
    applied: bool
    state: OwnerWorkingMemoryState | None
    reason: str


class WorkingMemoryService:
    def __init__(
        self,
        *,
        repository: WorkingMemoryRepository,
        policy: WorkingMemoryPolicy,
        domain_event_bus: DomainEventBus | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._policy = policy
        self._domain_event_bus = domain_event_bus
        self._now = now

    async def apply_observation(
        self, observation: VisualObservation, *, effect_guard=None,
    ) -> WorkingMemoryApplyResult:
        current = await self._repository.get_session_working_memory(
            owner_id=observation.owner_id,
            session_id=observation.session_id,
        )
        for attempt in range(2):
            if _is_stale(observation, current):
                return WorkingMemoryApplyResult(False, current, "stale_observation")
            base_state = (
                current.state
                if current is not None and current.expires_at > observation.ended_at
                else {}
            )
            reduced = reduce_session_working_memory(
                base_state,
                observation,
                policy=self._policy,
            )
            expires_at = max(self._now(), observation.ended_at) + timedelta(
                seconds=self._policy.ttl_seconds
            )
            try:
                write = await self._repository.compare_and_swap_session_working_memory(
                    owner_id=observation.owner_id,
                    session_id=observation.session_id,
                    expected_revision=current.revision if current is not None else 0,
                    observation_id=observation.id,
                    state=reduced,
                    expires_at=expires_at,
                    **({"effect_guard": effect_guard} if effect_guard is not None else {}),
                )
            except WorkingMemoryTargetMissingError:
                if attempt == 0:
                    current = await self._repository.get_session_working_memory(
                        owner_id=observation.owner_id,
                        session_id=observation.session_id,
                    )
                    continue
                raise
            if write.applied:
                state = _session_state(write.state)
                await self._publish_updated(state)
                return WorkingMemoryApplyResult(True, state, "applied")
            current = _session_state(write.state)
        return WorkingMemoryApplyResult(False, current, "concurrent_update")

    async def get_session_working_memory(
        self,
        *,
        owner_id: str,
        session_id: UUID,
    ) -> SessionWorkingMemoryState | None:
        state = await self._repository.get_session_working_memory(
            owner_id=owner_id,
            session_id=session_id,
        )
        now = self._now()
        if state is None or state.expires_at <= now:
            return None
        return replace(
            state,
            state=prune_session_working_memory(
                state.state,
                now=now,
                policy=self._policy,
            ),
        )

    async def set_latest_experience_event_id(
        self,
        *,
        owner_id: str,
        session_id: UUID,
        event_id: UUID,
    ) -> SessionWorkingMemoryState | None:
        state = await self._repository.set_latest_experience_event_id(
            owner_id=owner_id,
            session_id=session_id,
            event_id=event_id,
            expires_at=self._now()
            + timedelta(seconds=self._policy.ttl_seconds),
        )
        if state is not None:
            await self._publish_updated(state)
        return state

    async def get_owner_working_memory(
        self,
        *,
        owner_id: str,
    ) -> OwnerWorkingMemoryState | None:
        state = await self._repository.get_owner_working_memory(owner_id=owner_id)
        now = self._now()
        if state is not None and state.expires_at is not None and state.expires_at <= now:
            return None
        if state is None:
            return None
        return replace(
            state,
            state=normalize_owner_working_memory(state.state, now=now),
        )

    async def update_owner_working_memory(
        self,
        *,
        owner_id: str,
        updates: Mapping[str, Any],
    ) -> OwnerWorkingMemoryApplyResult:
        if not updates:
            raise ValueError("owner working-memory updates cannot be empty")
        now = self._now()
        current = await self._repository.get_owner_working_memory(owner_id=owner_id)
        for attempt in range(2):
            base = (
                normalize_owner_working_memory(current.state, now=now)
                if current is not None
                and (current.expires_at is None or current.expires_at > now)
                else {}
            )
            next_state = dict(base)
            for field, value in updates.items():
                if value is None:
                    if field not in {
                        "active_goal_ids",
                        "active_visual_focus",
                        "pending_commitments",
                        "carried_context",
                    }:
                        raise ValueError(
                            f"unsupported owner working-memory field: {field}"
                        )
                    next_state.pop(field, None)
                    continue
                normalized_update = normalize_owner_working_memory(
                    {field: value},
                    now=now,
                )
                if field in normalized_update:
                    next_state[field] = normalized_update[field]
                else:
                    next_state.pop(field, None)
            persisted_state = dict(next_state)
            expires_at = owner_working_memory_expires_at(next_state)
            if current is not None and "companion_live_state" in current.state:
                # Preserve the independent owner's section, including when our
                # expiring fields have all disappeared. Rebuild on CAS retry.
                persisted_state["companion_live_state"] = deepcopy(current.state["companion_live_state"])
                expires_at = None
            try:
                write = await self._repository.compare_and_swap_owner_working_memory(
                    owner_id=owner_id,
                    expected_revision=current.revision if current is not None else 0,
                    state=persisted_state,
                    expires_at=expires_at,
                )
            except WorkingMemoryTargetMissingError:
                if attempt == 0:
                    current = await self._repository.get_owner_working_memory(
                        owner_id=owner_id
                    )
                    continue
                raise
            if write.applied:
                state = _owner_state(write.state)
                await self._publish_owner_updated(state)
                return OwnerWorkingMemoryApplyResult(
                    True, state, "applied"
                )
            current = _owner_state(write.state)
        return OwnerWorkingMemoryApplyResult(False, current, "concurrent_update")

    async def _publish_updated(self, state: SessionWorkingMemoryState) -> None:
        if self._domain_event_bus is None:
            return
        await self._domain_event_bus.publish(
            DomainEvent(
                name=DomainEventName.WORKING_MEMORY_UPDATED,
                payload=WorkingMemoryUpdatedPayload(
                    owner_id=state.owner_id,
                    session_id=state.session_id,
                    revision=state.revision,
                ),
            )
        )

    async def _publish_owner_updated(self, state: OwnerWorkingMemoryState) -> None:
        if self._domain_event_bus is None:
            return
        await self._domain_event_bus.publish(
            DomainEvent(
                name=DomainEventName.OWNER_WORKING_MEMORY_UPDATED,
                payload=OwnerWorkingMemoryUpdatedPayload(
                    owner_id=state.owner_id,
                    revision=state.revision,
                ),
            )
        )


def _is_stale(
    observation: VisualObservation,
    current: SessionWorkingMemoryState | None,
) -> bool:
    return (
        current is not None
        and current.last_observation_ended_at is not None
        and observation.ended_at <= current.last_observation_ended_at
    )


def _session_state(
    value: SessionWorkingMemoryState | OwnerWorkingMemoryState,
) -> SessionWorkingMemoryState:
    if not isinstance(value, SessionWorkingMemoryState):
        raise TypeError("session working-memory write returned owner state")
    return value


def _owner_state(
    value: SessionWorkingMemoryState | OwnerWorkingMemoryState,
) -> OwnerWorkingMemoryState:
    if not isinstance(value, OwnerWorkingMemoryState):
        raise TypeError("owner working-memory write returned session state")
    return value
