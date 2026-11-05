from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.memory.router import router

from app.perception.models import OwnerWorkingMemoryState, WorkingMemoryWriteResult
from app.perception.services import WorkingMemoryService
from app.perception.working_memory import WorkingMemoryPolicy, normalize_owner_working_memory
from app.runtime.events import (
    DomainEventName,
    DomainEventBus,
    OwnerWorkingMemoryUpdatedPayload,
)


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class _Repository:
    def __init__(self) -> None:
        self.state: OwnerWorkingMemoryState | None = None

    async def get_owner_working_memory(self, *, owner_id):
        return self.state

    async def compare_and_swap_owner_working_memory(
        self, *, owner_id, expected_revision, state, expires_at, contract_version=1
    ):
        current_revision = self.state.revision if self.state else 0
        if current_revision != expected_revision:
            assert self.state is not None
            return WorkingMemoryWriteResult(False, self.state)
        self.state = OwnerWorkingMemoryState(
            owner_id=owner_id,
            revision=current_revision + 1,
            state=state,
            expires_at=expires_at,
            created_at=self.state.created_at if self.state else NOW,
            updated_at=NOW,
            contract_version=contract_version,
        )
        return WorkingMemoryWriteResult(True, self.state)


class OwnerWorkingMemoryServiceTests(unittest.IsolatedAsyncioTestCase):
    def metadata_repository(self):
        repository = _Repository()
        repository.state = OwnerWorkingMemoryState(owner_id="owner", revision=1,
            state={"companion_live_state": {"human": {"content": {"setting": "office"}}},
                   "active_goal_ids": _entry(["goal-1"], NOW + timedelta(minutes=10)),
                   "carried_context": _entry(["expired"], NOW - timedelta(seconds=1))},
            expires_at=None, created_at=NOW, updated_at=NOW, contract_version=1)
        return repository

    async def test_read_ignores_companion_metadata_without_mutating_storage(self):
        repository = self.metadata_repository()
        original = deepcopy(repository.state.state)
        result = await _service(repository).get_owner_working_memory(owner_id="owner")
        self.assertEqual(set(result.state), {"active_goal_ids"})
        self.assertEqual(repository.state.state, original)
        self.assertEqual(repository.state.revision, 1)
        self.assertEqual(normalize_owner_working_memory({"companion_live_state": None}, now=NOW), {})
        with self.assertRaisesRegex(ValueError, "unsupported owner working-memory fields"):
            normalize_owner_working_memory({"companion_live_state": {}, "unrecognized": {}}, now=NOW)

    async def test_update_preserves_independent_companion_section(self):
        repository = self.metadata_repository()
        companion = deepcopy(repository.state.state["companion_live_state"])
        result = await _service(repository).update_owner_working_memory(owner_id="owner",
            updates={"active_goal_ids": None})
        self.assertTrue(result.applied)
        self.assertEqual(repository.state.state, {"companion_live_state": companion})
        self.assertIsNone(repository.state.expires_at)

    async def test_http_working_state_accepts_companion_metadata(self):
        repository = self.metadata_repository()
        original = deepcopy(repository.state.state)
        service = _service(repository)

        async def working_state(*, owner_id, session_id):
            return None, await service.get_owner_working_memory(owner_id=owner_id)

        app = FastAPI()
        app.include_router(router)
        app.state.runtime_services = SimpleNamespace(memory_service=SimpleNamespace(working_state=working_state))
        app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(owner_id="owner")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            response = await client.get("/memory/working-state")
        self.assertEqual(response.status_code, 200)
        self.assertIn("goal-1", response.text)
        self.assertNotIn("companion_live_state", response.text)
        self.assertEqual(repository.state.state, original)

    async def test_update_validates_and_expires_individual_fields(self) -> None:
        service = _service(_Repository())
        result = await service.update_owner_working_memory(
            owner_id="owner",
            updates={
                "active_goal_ids": _entry(["goal-1"], NOW + timedelta(minutes=10)),
                "carried_context": _entry(["expired"], NOW - timedelta(seconds=1)),
            },
        )

        self.assertTrue(result.applied)
        assert result.state is not None
        self.assertEqual(result.state.state["active_goal_ids"]["value"], ["goal-1"])
        self.assertNotIn("carried_context", result.state.state)

    async def test_update_rejects_non_working_memory_fields(self) -> None:
        service = _service(_Repository())
        with self.assertRaisesRegex(ValueError, "unsupported owner working-memory field"):
            await service.update_owner_working_memory(
                owner_id="owner",
                updates={"preferences": _entry(["tea"], NOW + timedelta(minutes=10))},
            )

    async def test_applied_update_publishes_owner_refresh_event(self) -> None:
        bus = DomainEventBus()
        received = []

        async def receive(event):
            received.append(event)

        bus.subscribe(DomainEventName.OWNER_WORKING_MEMORY_UPDATED, receive)
        service = _service(_Repository(), domain_event_bus=bus)

        result = await service.update_owner_working_memory(
            owner_id="owner",
            updates={
                "active_goal_ids": _entry(["goal-1"], NOW + timedelta(minutes=10))
            },
        )

        self.assertTrue(result.applied)
        self.assertEqual(len(received), 1)
        payload = received[0].payload
        self.assertIsInstance(payload, OwnerWorkingMemoryUpdatedPayload)
        self.assertEqual(payload.owner_id, "owner")
        self.assertEqual(payload.revision, 1)


def _entry(value, expires_at):
    return {
        "value": value,
        "expires_at": expires_at.isoformat(),
        "provenance": {"source": "test"},
    }


def _service(repository, *, domain_event_bus=None):
    return WorkingMemoryService(
        repository=repository,
        policy=WorkingMemoryPolicy(120, 30, 60, 30, 12),
        domain_event_bus=domain_event_bus,
        now=lambda: NOW,
    )
