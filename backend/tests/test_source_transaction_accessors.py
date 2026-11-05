"""Offline guards for Workspace's shared source-reader integration seams."""
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID

from app.knowledge.models import SourceError
from app.knowledge.notes import SourceNoteRepository
from app.knowledge.sources import SourceRepository
from app.notes.models import JarvisNote
from app.tasks.models import TaskError


OWNER = "workspace-fixture@example.test"
AREA = UUID(int=1)
SOURCE = UUID(int=2)


class SourceTransactionAccessorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Any accidental pool acquisition fails rather than contacting a DB.
        self.repository = SourceRepository(object())
        self.connection = SimpleNamespace(fetchrow=AsyncMock(), fetch=AsyncMock())

    async def test_document_resolution_reuses_connection_and_history_validator(self):
        row, version = object(), object()
        self.connection.fetchrow.return_value = row
        self.repository._version = Mock(return_value=version)
        result = await self.repository.resolve_document_in_transaction(
            self.connection, OWNER, AREA, "source_fixture", historical=True,
        )
        self.assertIs(result, version)
        query, *arguments = self.connection.fetchrow.call_args.args
        self.assertEqual(arguments, [OWNER, AREA, "source_fixture"])
        self.assertIn("v.owner_id=$1 and v.area_id=$2", query)
        self.assertIn("a.status='active'", query)
        self.repository._version.assert_called_once_with(row, historical=True)

    async def test_historical_document_resolution_still_rejects_deleted_source(self):
        self.connection.fetchrow.return_value = {
            "revision": 1, "current_revision": 2, "context_is_current": True,
            "source_status": "deleted", "availability": "available",
        }
        with self.assertRaises(SourceError) as raised:
            await self.repository.resolve_document_in_transaction(
                self.connection, OWNER, AREA, "source_fixture", historical=True,
            )
        self.assertEqual(raised.exception.code, "knowledge_source_unavailable")

    async def test_versions_keep_unavailable_count_and_bounded_continuation(self):
        self.repository._read = AsyncMock(return_value=SimpleNamespace(generation=3, revision=4))
        self.connection.fetch.return_value = [{"revision": 4}, {"revision": 3}, {"revision": 2}]

        def version(row, *, historical):
            self.assertTrue(historical)
            if row["revision"] == 3:
                raise SourceError("knowledge_source_unavailable", "Removed original")
            return SimpleNamespace(original_evidence=lambda: {
                "revision": row["revision"], "content": {"private": "not returned"},
            })

        self.repository._version = Mock(side_effect=version)
        result = await self.repository.list_versions_in_transaction(
            self.connection, OWNER, AREA, SOURCE, limit=2, expected_generation=3,
        )
        self.assertEqual(result["versions"], [{"revision": 4}])
        self.assertEqual(result["unavailable_revisions"], 1)
        self.assertEqual(result["next_before_revision"], 3)
        self.assertEqual(result["expected_generation"], 3)
        self.assertEqual(self.connection.fetch.call_args.args[1:], (OWNER, AREA, SOURCE, None, 3))
        self.repository._read.assert_awaited_once_with(self.connection, OWNER, AREA, SOURCE)

    async def test_changed_generation_rejects_before_history_fetch(self):
        self.repository._read = AsyncMock(return_value=SimpleNamespace(generation=2, revision=4))
        with self.assertRaises(SourceError) as raised:
            await self.repository.list_versions_in_transaction(
                self.connection, OWNER, AREA, SOURCE, before_revision=4, expected_generation=1,
            )
        self.assertEqual(raised.exception.code, "knowledge_source_changed")
        self.connection.fetch.assert_not_awaited()

    async def test_invalid_windows_fail_without_pool_or_connection_io(self):
        for arguments in ({"limit": 0}, {"limit": True}, {"before_revision": 2},
                          {"expected_generation": False}, {"before_revision": -1}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    await self.repository.list_versions(OWNER, AREA, SOURCE, **arguments)
                with self.assertRaises(ValueError):
                    await self.repository.list_versions_in_transaction(
                        self.connection, OWNER, AREA, SOURCE, **arguments,
                    )
        self.connection.fetch.assert_not_awaited()

    async def test_public_methods_retain_one_transaction_and_delegate(self):
        entered = []

        @asynccontextmanager
        async def transaction(owner):
            entered.append(owner)
            yield self.connection

        self.repository.transaction = transaction
        self.repository.resolve_document_in_transaction = AsyncMock(return_value="version")
        self.repository.list_versions_in_transaction = AsyncMock(return_value={"versions": []})
        self.assertEqual(await self.repository.resolve_document(
            OWNER, AREA, "source_fixture", historical=True), "version")
        await self.repository.list_versions(OWNER, AREA, SOURCE, limit=2)
        self.assertEqual(entered, [OWNER, OWNER])
        self.repository.resolve_document_in_transaction.assert_awaited_once_with(
            self.connection, OWNER, AREA, "source_fixture", historical=True)
        self.repository.list_versions_in_transaction.assert_awaited_once_with(
            self.connection, OWNER, AREA, SOURCE, limit=2,
            before_revision=None, expected_generation=None)


class StructuredNoteLegacyGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.current = JarvisNote(
            id=UUID(int=3), session_id=UUID(int=4), title="Training", body="Reps: 10",
            source="assistant", created_at=datetime(2026, 11, 1, tzinfo=UTC),
            updated_at=datetime(2026, 11, 1, tzinfo=UTC),
            metadata={"workspace_v1": {"schema_version": 1, "blocks": []}},
        )
        self.connection = SimpleNamespace(fetchrow=AsyncMock(return_value={}))
        self.entered = []

        @asynccontextmanager
        async def transaction(owner):
            self.entered.append(owner)
            yield self.connection

        self.repository = object.__new__(SourceNoteRepository)
        self.repository.owner_id = OWNER
        self.repository.access_check = None
        self.repository.task_evidence = None
        self.repository.sources = SimpleNamespace(transaction=transaction)
        self.repository._area_for = AsyncMock(return_value=AREA)
        self.repository._expose = AsyncMock(return_value=self.current)
        self.repository._mapping = AsyncMock(return_value={"id": SOURCE})
        self.repository._publish = AsyncMock()

    async def update(self, *, title=None, body=None):
        with patch("app.knowledge.notes._note_from_record", return_value=self.current):
            return await self.repository.update_note(
                owner_id=OWNER, note_id=self.current.id, title=title, body=body)

    async def test_raw_body_edit_rejected_before_write_or_publication(self):
        with patch("app.knowledge.notes.NoteRepository.update_note", new_callable=AsyncMock) as write:
            with self.assertRaises(SourceError) as raised:
                await self.update(title="Different", body="Reps: 8")
            self.assertEqual(raised.exception.code, "note_structured_update_required")
            write.assert_not_awaited()
        self.repository._publish.assert_not_awaited()

    async def test_identical_body_is_still_a_noop(self):
        result = await self.update(body=self.current.body)
        self.assertIs(result, self.current)
        self.repository._publish.assert_not_awaited()

    async def test_title_only_edit_retains_existing_publication_path(self):
        updated = self.current.model_copy(update={"title": "Next session"})
        self.repository._publish.return_value = updated
        with patch("app.knowledge.notes.NoteRepository.update_note", new_callable=AsyncMock) as write:
            write.return_value = updated
            self.assertIs(await self.update(title="Next session"), updated)
            write.assert_awaited_once_with(owner_id=OWNER, note_id=self.current.id,
                                          title="Next session", body=None)
        self.repository._publish.assert_awaited_once()

    async def test_legacy_plain_text_body_remains_editable(self):
        self.current = self.current.model_copy(update={"metadata": {"retained": True}})
        self.repository._expose.return_value = self.current
        updated = self.current.model_copy(update={"body": "Reps: 8"})
        self.repository._publish.return_value = updated
        with patch("app.knowledge.notes.NoteRepository.update_note", new_callable=AsyncMock) as write:
            write.return_value = updated
            self.assertIs(await self.update(body="Reps: 8"), updated)
            write.assert_awaited_once()

    async def test_worker_edit_prohibition_is_unchanged(self):
        self.repository.task_evidence = object()
        with self.assertRaises(TaskError) as raised:
            await self.update(body="Reps: 8")
        self.assertEqual(raised.exception.code, "note_update_not_granted")
        self.assertEqual(self.entered, [])


if __name__ == "__main__":
    unittest.main()
