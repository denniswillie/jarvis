import json
import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.live_state.briefing import LiveBriefingReader
from app.live_state.models import (
    AggregateRef, ConversationView, GuardedPayloadBinding, LiveSnapshot,
    ObservationEntry, SessionScope, SourceDependency, StateCursor,
)


class BriefingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 1, tzinfo=UTC)
        self.tick = 0.0
        self.scope = SessionScope(owner_id="briefing-test", session_id=UUID(int=1))
        self.snapshot = LiveSnapshot(scope=self.scope,
            cursor=StateCursor(epoch=UUID(int=2), revision=1), as_of=self.now,
            readiness="ready", entries=tuple(self.entry(index) for index in range(9)))
        self.reader = LiveBriefingReader(owner_id=self.scope.owner_id,
            session_id=self.scope.session_id, snapshot=lambda: self.snapshot,
            clock=lambda: self.now, monotonic=lambda: self.tick)

    def entry(self, index, **changes):
        source = SourceDependency(source_namespace="knowledge_source", source_id=UUID(int=100 + index),
                                  revision=1, generation=1)
        fields = dict(entry_id=f"observation:{index:02}", scope=self.scope,
            received_at=self.now, observed_at=self.now - timedelta(seconds=1),
            fresh_until=self.now + timedelta(seconds=60), expires_at=self.now + timedelta(seconds=60),
            evidence_kind="observed", sources=(source,), summary=f"Visible value {index}.",
            guard_binding=GuardedPayloadBinding(scope=self.scope, dependencies=(source,),
                                               payload_root_ids=(UUID(int=200 + index),)))
        return ObservationEntry(**(fields | changes))

    async def test_read_is_bounded_and_does_not_expose_internal_lineage(self):
        result = await self.reader.read({})
        self.assertEqual(result["outcome"], "available")
        self.assertFalse(result["effect_committed"])
        self.assertLessEqual(len(result["briefing"]["text"]), 4096)
        self.assertEqual(result["briefing"]["text"].count("Untrusted visual"), 8)
        for field in ("payload_root_ids", "source_id", "owner_id", "dependencies", "epoch"):
            self.assertNotIn(field, json.dumps(result))

    async def test_second_page_uses_remaining_entries(self):
        first = await self.reader.read({})
        second = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
        self.assertIn("Visible value 8.", second["briefing"]["text"])
        self.assertNotIn("Visible value 0.", second["briefing"]["text"])
        self.assertFalse(second["briefing"]["more_available"])

    async def test_unrelated_snapshot_revision_does_not_expire_view(self):
        first = await self.reader.read({})
        self.snapshot = self.snapshot.model_copy(update={"cursor": self.snapshot.cursor.model_copy(update={"revision": 2})})
        second = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
        self.assertEqual(second["outcome"], "available")

    async def test_changed_epoch_or_focus_expires_cursor(self):
        for changes in ({"cursor": StateCursor(epoch=uuid4(), revision=1)},
                        {"conversation": ConversationView(focus=AggregateRef(namespace="focus", aggregate_id=uuid4(), revision=1))}):
            first = await self.reader.read({})
            self.snapshot = self.snapshot.model_copy(update=changes)
            result = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
            self.assertEqual(result["error"]["code"], "cursor_expired")

    async def test_expired_view_is_not_renewed_by_read(self):
        first = await self.reader.read({})
        self.tick = 30.0
        result = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
        self.assertEqual(result["error"]["code"], "cursor_expired")

    async def test_only_two_views_are_retained(self):
        first = await self.reader.read({})
        await self.reader.read({})
        await self.reader.read({})
        result = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
        self.assertEqual(result["error"]["code"], "cursor_expired")

    async def test_retraction_or_replacement_is_not_resurrected_by_cursor(self):
        first = await self.reader.read({})
        replacement = self.entry(8, summary="Corrected new observation.")
        self.snapshot = self.snapshot.model_copy(update={"entries": (*self.snapshot.entries[:8], replacement)})
        second = await self.reader.read({"cursor": first["briefing"]["next_cursor"]})
        self.assertNotIn("Visible value 8.", second["briefing"]["text"])
        self.assertNotIn("Corrected new observation.", second["briefing"]["text"])

    async def test_stale_visuals_remain_unknown(self):
        self.now += timedelta(seconds=61)
        result = await self.reader.read({})
        self.assertIn("Current visual view unknown", result["briefing"]["text"])
        self.assertNotIn("Visible value", result["briefing"]["text"])

    async def test_missing_or_other_owner_snapshot_is_unavailable(self):
        for value in (None, self.snapshot.model_copy(update={"scope": SessionScope(owner_id="other", session_id=UUID(int=1))})):
            self.snapshot = value
            self.assertEqual((await self.reader.read({}))["error"]["code"], "live_state_unavailable")

    async def test_warming_does_not_wait_or_issue_continuation(self):
        self.snapshot = self.snapshot.model_copy(update={"readiness": "warming"})
        result = await self.reader.read({})
        self.assertEqual(result["outcome"], "unavailable")
        self.assertFalse(result["briefing"]["more_available"])
        self.assertNotIn("Visible value", result["briefing"]["text"])

    async def test_invalid_argument_shapes_are_rejected(self):
        for arguments in (None, [], {"owner_id": "other"}, {"cursor": None}, {"cursor": True}, {"cursor": ""}):
            with self.subTest(arguments=arguments):
                self.assertEqual((await self.reader.read(arguments))["outcome"], "rejected")

    def test_tool_is_explicitly_local_read_only(self):
        tool = self.reader.tool()
        self.assertTrue(tool.local_snapshot_read)
        self.assertFalse(tool.mutates_state)
        self.assertIsNone(tool.result_capture)


if __name__ == "__main__":
    unittest.main()
