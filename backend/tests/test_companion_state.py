import asyncio
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.live_state.companion import CompanionState, STATE_KEY
from app.attention.quiet_delivery import QuietDeliverySuppressed


class Memory:
    def __init__(self):
        self.row = NS(revision=1, state={"unrelated": "preserved"})

    async def get_owner_working_memory(self, **kwargs):
        return self.row

    async def compare_and_swap_owner_working_memory(self, **kwargs):
        if kwargs["expected_revision"] != self.row.revision:
            return NS(applied=False, state=self.row)
        self.row = NS(revision=self.row.revision + 1, state=deepcopy(kwargs["state"]))
        return NS(applied=True, state=self.row)


class Reminders:
    def __init__(self):
        self.rows = {}

    def row(self, data):
        return NS(id=data["id"], model_dump=lambda **kw: json.loads(json.dumps(data, default=str)))

    async def list_reminders(self, **kwargs):
        return [self.row(row) for row in self.rows.values() if row["status"] == "active"]

    async def create_reminder(self, **kwargs):
        data = {**kwargs, "id": str(uuid4()), "status": "active"}
        self.rows[data["id"]] = data
        return self.row(data)

    async def complete_reminder(self, **kwargs):
        return self.change(kwargs["reminder_id"], "completed")

    async def delete_reminder(self, **kwargs):
        return self.change(kwargs["reminder_id"], "deleted")

    def change(self, identity, status):
        row = self.rows.get(str(identity))
        if row:
            row["status"] = status
            return self.row(row)


class CompanionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.now = datetime(2026, 11, 4, tzinfo=timezone.utc)
        self.memory, self.reminders = Memory(), Reminders()
        self.state = CompanionState(owner_id="fixture", working_memory=self.memory,
            reminders=self.reminders, clock=lambda: self.now)
        await self.state.initialize()

    async def test_read_is_ram_only_and_returns_a_copy(self):
        self.memory.get_owner_working_memory = AsyncMock(side_effect=AssertionError("I/O"))
        self.reminders.list_reminders = AsyncMock(side_effect=AssertionError("I/O"))
        value = self.state.read()
        value["system"]["reminders"].append({"id": "caller-only"})
        self.assertEqual(self.state.read()["system"]["reminders"], [])
        self.assertNotIn("quiet_mode", value["system"])
        self.assertTrue(value["ready"])

    async def test_legacy_stored_quiet_is_preserved_but_not_projected_or_enforced(self):
        legacy = {"active": True, "updated_at": self.now.isoformat(),
                  "expires_at": (self.now + timedelta(seconds=60)).isoformat()}
        self.memory.row.state[STATE_KEY] = {"quiet": legacy}
        restored = CompanionState(owner_id="fixture", working_memory=self.memory,
            reminders=self.reminders, clock=lambda: self.now)
        await restored.initialize()
        self.assertTrue(restored.notice_allowed())
        async with restored.delivery_guard():
            pass
        self.assertNotIn("quiet_mode", restored.read()["system"])
        self.assertFalse(hasattr(restored, "quiet"))
        self.assertFalse(hasattr(restored, "set_quiet"))
        self.now += timedelta(seconds=61)
        self.assertTrue(restored.notice_allowed(observed_at=self.now))
        self.assertTrue(restored.notice_allowed(observed_at=self.now - timedelta(seconds=30)))
        self.assertEqual(self.memory.row.state[STATE_KEY]["quiet"], legacy)
        self.assertEqual(self.memory.row.state["unrelated"], "preserved")

    async def test_notice_admission_requires_initialized_state(self):
        state = CompanionState(owner_id="fixture", working_memory=Memory(),
                               reminders=Reminders(), clock=lambda: self.now)
        self.assertFalse(state.notice_allowed())
        with self.assertRaises(QuietDeliverySuppressed):
            async with state.delivery_guard():
                self.fail("uninitialized state admitted delivery")
        await state.initialize()
        self.assertTrue(state.notice_allowed())
        async with state.delivery_guard():
            pass

    async def test_human_availability_remains_advisory(self):
        await self.state.update_human({"availability": "do_not_disturb"}, observed_at=self.now)
        self.assertTrue(self.state.notice_allowed())
        self.now += timedelta(days=1)
        human = self.state.read()["human"]
        self.assertTrue(human["stale"])
        self.assertNotEqual(human["last_observed_at"], self.now.isoformat())

    async def test_human_update_cannot_rewind_evidence(self):
        self.assertTrue(await self.state.update_human({"setting": "office"}, observed_at=self.now))
        self.assertFalse(await self.state.update_human({"setting": "kitchen"},
            observed_at=self.now - timedelta(minutes=1)))
        self.assertEqual(self.state.read()["human"]["content"]["setting"], "office")

    async def test_concurrent_human_and_reminder_writes_preserve_both_projections(self):
        await asyncio.gather(self.state.update_human({"setting": "office"}, observed_at=self.now),
                             self.state.manage_reminder(action="create", title="Keys", condition="Leaving"))
        self.assertEqual(self.state.read()["system"]["reminders"][0]["title"], "Keys")
        self.assertEqual(self.state.read()["human"]["content"]["setting"], "office")

    async def test_forget_invalidates_derived_state_and_matching_reminders(self):
        first = await self.state.manage_reminder(action="create", title="Umbrella", condition="Leaving",
            source_document_ids=("doc-a",))
        second = await self.state.manage_reminder(action="create", title="Keys", condition="Leaving",
            source_document_ids=("doc-b",))
        generation = self.state.read()["human_generation"]
        await self.state.update_human({"setting": "office"}, observed_at=self.now, generation=generation)
        await self.state.invalidate_memory(NS(owner_id="fixture", document_ids=("doc-a",)))
        self.assertIsNone(self.state.read()["human"])
        self.assertEqual(self.reminders.rows[first["id"]]["status"], "deleted")
        self.assertEqual(self.reminders.rows[second["id"]]["status"], "active")
        self.assertFalse(await self.state.update_human({"setting": "old"}, observed_at=self.now,
                                                     generation=generation))

    async def test_foreign_forget_is_rejected(self):
        with self.assertRaises(ValueError):
            await self.state.invalidate_memory(NS(owner_id="other", document_ids=("doc-a",)))

    async def test_complete_only_targets_one_occurrence(self):
        first = await self.state.manage_reminder(action="create", title="Check", condition="Leaving")
        second = await self.state.manage_reminder(action="create", title="Other", condition="Arriving")
        await self.state.manage_reminder(action="complete", reminder_id=first["id"])
        self.assertEqual(self.reminders.rows[first["id"]]["status"], "completed")
        self.assertEqual(self.reminders.rows[second["id"]]["status"], "active")

