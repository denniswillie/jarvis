from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.assistant.memory_context import working_memory_app_event_payload
from app.realtime.gemini.client import GeminiTransportError
from app.realtime.gemini.context_updates import GeminiContextUpdates
from app.runtime.events import DomainEvent, DomainEventName, MemoryItemChangedPayload
from test_gemini_domain_integration import context


def payload(*, revision=3, owner_revision=7, expires=None):
    return {"working_memory": {"available": True, "session": {
        "revision": revision, "observed_at": datetime.now(UTC).isoformat(),
        "expires_at": expires.isoformat() if expires else None,
        "state": {"scene": {"label": "desk"}, "activity": {"label": "coding"}},
    }, "owner": {"revision": owner_revision, "expires_at": None, "state": {}}}}


class GeminiContextUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.context = context()
        self.closed = False
        self.build = AsyncMock(return_value=payload())
        self.events = []
        self.updates = GeminiContextUpdates(
            context=self.context, build=self.build, emit=AsyncMock(side_effect=self.events.append),
            closed=lambda: self.closed, timeout_seconds=.1, minimum_interval_seconds=0,
        )

    async def asyncTearDown(self):
        await self.updates.close()

    async def test_coalesces_working_revisions_and_reuses_existing_phone_payload(self):
        for revision in (1, 3, 2):
            await self.updates.notify_working_memory_updated(revision)
        await self.updates.coordinator.wait_idle()
        self.build.assert_awaited_once()
        self.assertEqual(self.events, [{"type": "working_memory.updated",
                                       "session_id": str(self.context.jarvis_session_id),
                                       **working_memory_app_event_payload(self.build.return_value["working_memory"])}])
        self.assertEqual(self.updates.coordinator.applied_revision, 3)

    async def test_owner_revision_is_independent_from_session_revision(self):
        self.updates.initialize(payload(revision=20, owner_revision=2))
        self.build.return_value = payload(revision=20, owner_revision=3)
        await self.updates.notify_owner_working_memory_updated(3)
        await self.updates.coordinator.wait_idle()
        self.assertEqual(self.events[0]["owner_revision"], 3)
        self.assertEqual(self.events[0]["revision"], 20)

    async def test_durable_memory_hint_is_owner_and_session_scoped(self):
        memory_id = uuid4()
        valid = DomainEvent(DomainEventName.MEMORY_ITEM_CHANGED, MemoryItemChangedPayload(
            owner_id=self.context.owner_id, session_id=self.context.jarvis_session_id,
            memory_item_id=memory_id, change_type="updated", memory_type="semantic_fact", reason="user_correction",
        ))
        for owner, session_id in (("foreign", self.context.jarvis_session_id), (self.context.owner_id, uuid4())):
            await self.updates.notify_memory_item_changed(DomainEvent(
                DomainEventName.MEMORY_ITEM_CHANGED, MemoryItemChangedPayload(
                    owner_id=owner, session_id=session_id, memory_item_id=uuid4(), change_type="updated",
                ),
            ))
        self.assertEqual(self.events, [])
        await self.updates.notify_memory_item_changed(valid)
        await self.updates.coordinator.wait_idle()
        self.assertEqual(self.events[0], {
            "type": "memory.updated", "change_id": str(valid.event_id), "changed_at": valid.occurred_at.isoformat(),
            "affected_memory_ids": [str(memory_id)], "affected_memory_types": ["semantic_fact"],
            "reason": "user_correction", "change_type": "updated",
        })

    async def test_expiry_refreshes_empty_memory_without_a_new_revision(self):
        self.updates.initialize(payload(expires=datetime.now(UTC) + timedelta(seconds=.01)))
        self.build.return_value = {"working_memory": {"available": True, "session": None, "owner": None}}
        expiry = self.updates._expiry_task
        await expiry
        await self.updates.coordinator.wait_idle()
        self.assertFalse(self.events[-1]["session_available"])
        self.assertFalse(self.events[-1]["owner_available"])
        self.assertFalse(self.updates.diagnostics()["expiry_pending"])

    async def test_unavailable_context_does_not_replace_existing_revision_or_expiry(self):
        self.updates.initialize(payload(expires=datetime.now(UTC) + timedelta(seconds=30)))
        expiry = self.updates._expiry_task
        self.build.return_value = {"working_memory": {"available": False}}
        with self.assertRaises(GeminiTransportError):
            await self.updates._refresh(None)
        self.assertIs(self.updates._expiry_task, expiry)
        self.assertEqual(self.events, [])
        self.assertEqual(self.updates.diagnostics()["owner_revision"], 7)

    async def test_a_late_older_build_does_not_roll_back_memory_revision(self):
        self.updates.initialize(payload(revision=5, owner_revision=9))
        with self.assertRaises(GeminiTransportError):
            await self.updates._refresh(None)
        self.assertEqual(self.events, [])
        self.assertEqual(self.updates.diagnostics()["session_revision"], 5)

    async def test_stop_cancels_owned_refresh_and_expiry_and_ignores_new_updates(self):
        entered = asyncio.Event()

        async def blocked():
            entered.set()
            await asyncio.Event().wait()

        self.updates.initialize(payload(revision=1, expires=datetime.now(UTC) + timedelta(seconds=30)))
        self.build.side_effect = blocked
        await self.updates.notify_working_memory_updated(2)
        await entered.wait()
        await self.updates.close()
        await self.updates.notify_working_memory_updated(3)
        await self.updates.notify_owner_working_memory_updated(10)
        self.assertEqual(self.events, [])
        self.assertTrue(self.updates.diagnostics()["stopped"])
        self.assertFalse(self.updates.diagnostics()["expiry_pending"])
        self.assertIsNone(self.updates.coordinator._task)

    async def test_closed_during_build_cannot_send_a_late_hint(self):
        async def departed():
            self.closed = True
            return payload()

        self.build.side_effect = departed
        self.assertIsNone(await self.updates._refresh(3))
        self.assertEqual(self.events, [])
