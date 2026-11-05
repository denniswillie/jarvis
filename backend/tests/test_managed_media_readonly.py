"""Media source validation preserves authority in both transaction modes."""
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from app.tasks.media_editing import MEDIA_SOURCE_SQL, MediaInput, validate_media_input
from app.tasks.models import TaskError


class MediaReadOnlyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pin = MediaInput(owner_id="fixture-owner", session_id=uuid4(), recording_id=uuid4(),
            area_id=uuid4(), file_id=uuid4(), source_id=uuid4(), version_id="source_exact",
            receipt_event_id=uuid4(), source_payload_id=uuid4(), receipt_payload_id=uuid4(),
            admission_payload_id=uuid4(), content_sha256="a" * 64, size_bytes=100)
        row = {**self.pin.model_dump(), "completion_event_id": self.pin.receipt_event_id,
               "completion_payload_id": self.pin.receipt_payload_id, "original_sha256": self.pin.content_sha256}
        self.connection = SimpleNamespace(fetchrow=AsyncMock(return_value=row))

    async def test_mutation_and_execution_keep_row_locks_by_default(self):
        actual = await validate_media_input(self.connection, self.pin.owner_id, self.pin.session_id, self.pin)
        self.assertEqual(actual, self.pin)
        query, *args = self.connection.fetchrow.await_args.args
        self.assertEqual(query, MEDIA_SOURCE_SQL + " for share of r,a,s,v,p,c,ap,f,fp,ce,ae")
        self.assertEqual(args, [self.pin.owner_id, self.pin.session_id, self.pin.recording_id])

    async def test_readonly_mode_uses_identical_source_predicates_without_locks(self):
        actual = await validate_media_input(self.connection, self.pin.owner_id, self.pin.session_id,
                                            self.pin, lock=False)
        self.assertEqual(actual, self.pin)
        query, *args = self.connection.fetchrow.await_args.args
        self.assertEqual(query, MEDIA_SOURCE_SQL)
        self.assertNotIn("for share", query.lower())
        self.assertNotIn("for update", query.lower())
        self.assertIn("not r.revoked", query)
        self.assertIn("v.revision=s.current_revision", query)
        self.assertEqual(args, [self.pin.owner_id, self.pin.session_id, self.pin.recording_id])

    async def test_readonly_mode_does_not_bypass_withdrawal(self):
        self.connection.fetchrow.return_value = None
        with self.assertRaises(TaskError):
            await validate_media_input(self.connection, self.pin.owner_id, self.pin.session_id,
                                       self.pin, lock=False)


if __name__ == "__main__":
    unittest.main()
