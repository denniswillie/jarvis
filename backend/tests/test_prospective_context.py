from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

from app.memory.prospective.context import ProspectiveContextBuilder
from app.memory.prospective.models import ProspectiveContext
from app.visual_memory.context import _resolve_active_focus


class _FocusReader:
    async def get_active_visual_focus(self, **kwargs):
        return None


class _ReminderReader:
    async def list_active_reminders(self, **kwargs):
        return [SimpleNamespace(status="active", condition="when brushing")]


class _GoalReader:
    async def list_active_goals(self, **kwargs):
        return []


class _WorkingMemoryReader:
    async def get_owner_working_memory(self, **kwargs):
        return SimpleNamespace(
            state={
                "pending_commitments": {
                    "value": ["call Mum"],
                    "expires_at": datetime.now(UTC).isoformat(),
                    "provenance": {"source": "test"},
                }
            }
        )


class ProspectiveContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_keeps_sources_distinct(self) -> None:
        builder = ProspectiveContextBuilder(
            focus_reader=_FocusReader(), reminder_reader=_ReminderReader(),
            goal_reader=_GoalReader(), owner_working_memory_reader=_WorkingMemoryReader(),
            max_conditional_reminders=5, max_goals=5, max_commitments=5,
        )
        context = await builder.build(owner_id="owner")
        self.assertEqual(len(context.conditional_reminders), 1)
        self.assertEqual(context.commitments, ("call Mum",))

    def test_durable_empty_focus_does_not_restore_legacy_focus(self) -> None:
        prospective = ProspectiveContext(
            visual_focus=None,
            conditional_reminders=(),
            goals=(),
            commitments=(),
            generated_at=datetime.now(UTC),
        )

        resolved = _resolve_active_focus(
            prospective=prospective,
            legacy_focus={"id": "stale-focus"},
        )

        self.assertIsNone(resolved)
