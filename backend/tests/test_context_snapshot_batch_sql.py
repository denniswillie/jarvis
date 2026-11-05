"""Manifested SQL acceptance for bounded current/pinned projection contents."""
import os
import unittest
from uuid import uuid4

from app.tasks.models import TaskError
import test_second_brain_context_integration as fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL")
                     and os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested SQL fixture required")
class ContextSnapshotBatchSQLTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.CanonicalContextDatabaseTests.asyncSetUp
    ingress = fixture.CanonicalContextDatabaseTests.ingress

    async def test_revisions_pins_windows_foreign_scope_and_empty_scope(self):
        records = [await self.context.append(self.owner, self.ingress(identity=name, finality="final"))
                   for name in "abcdef"]
        for index in (0, 2, 4):
            await self.context.pin(self.owner, self.session, records[index].id, reason="Synthetic pinned context")
        revised = await self.context.append(self.owner, self.ingress(identity="a", finality="revised",
            expected_revision=1, text="Correction does not make this the newest utterance"))
        result = await self.context.snapshot(self.owner, self.session, max_messages=4, max_pins=1)
        self.assertEqual([r.id for r in result["recent"]], [r.id for r in reversed(records[2:])])
        self.assertEqual([r.id for r in result["pinned"]], [records[4].id])
        self.assertEqual(result["selected_pinned_ids"], {records[2].id, records[4].id})
        self.assertEqual((result["watermark"], result["total"], result["pinned_total"]), (revised.sequence, 6, 3))
        wider = await self.context.snapshot(self.owner, self.session, max_messages=1, max_pins=3)
        self.assertEqual([r.id for r in wider["pinned"]], [records[4].id, records[2].id, revised.id])
        self.assertEqual(wider["pinned"][-1].revision, 2)
        self.assertEqual(wider["pinned"][-1].origin_sequence, records[0].sequence)
        for owner, session in ((self.other, self.session), (self.owner, uuid4())):
            with self.assertRaises(TaskError) as error:
                await self.context.snapshot(owner, session, max_messages=1, max_pins=1)
            self.assertEqual(error.exception.code, "context_session_not_found")
        empty = await self.context.snapshot(self.owner, None, max_messages=1, max_pins=1)
        self.assertEqual((empty["total"], empty["pinned_total"], empty["scope_session_count"]), (0, 0, 0))
        self.assertEqual(empty["watermark"], revised.sequence)
        self.assertEqual((empty["recent"], empty["pinned"], empty["selected_pinned_ids"]), ([], [], set()))
