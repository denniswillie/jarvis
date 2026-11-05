"""Core-owned lane: real SQL acceptance for the C0 note slice.

This file is authored by Workspace but executed only by Core with its manifested
fixture lane. A skipped test is not persistence evidence. No provider is called.
"""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.knowledge.models import SourceError
from app.knowledge.notes import SourceNoteRepository
from app.notes.workspace_repository import WorkspaceNoteRepository
from app.workspace.models import WorkspaceContext
from legacy_workspace_notes_fixture import WorkspaceNoteService
from evals.second_brain.source_fixture import source_fixture


def activity_content():
    return {"schema_version": 1, "blocks": [{"type": "table", "id": "sets",
        "columns": [{"id": "reps", "label": "Reps"}],
        "rows": [{"id": "set_1", "cells": {"reps": None}}]}]}


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Core's explicit manifested database lane is required")
class WorkspaceNotesDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_lifecycle_history_replay_and_rollback(self):
        f = await self.enterAsyncContext(source_fixture(os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        notes = SourceNoteRepository(f.sources, owner_id=f.owner, area_id=f.area)
        service = WorkspaceNoteService(WorkspaceNoteRepository(notes))
        async with f.pool.acquire() as conn:
            session = await conn.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
            later_session = await conn.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.owner)
            foreign_session = await conn.fetchval("insert into jarvis_sessions(owner_id) values($1) returning id", f.other)
        def ctx(operation=None):
            return WorkspaceContext(f.owner, session, operation)
        async def edit(base, value, operation=None):
            return await service.patch_note(ctx(operation or uuid4()), note_id, base,
                [{"op": "set_cell", "block_id": "sets", "row_id": "set_1", "column_id": "reps", "value": value}])

        create_operation = uuid4()
        created = await service.create_note(ctx(create_operation), "Workspace SQL fixture", activity_content())
        note_id = UUID(created["artifact"]["artifact_id"])
        initial = created["artifact"]["version_id"]
        correction_operation = uuid4()
        corrected = await edit(initial, 8, correction_operation)
        corrected_version = corrected["artifact"]["version_id"]
        newest = await edit(corrected_version, 9)
        replay = await edit(initial, 8, correction_operation)
        self.assertEqual(replay["artifact"]["version_id"], corrected_version)
        self.assertEqual((await edit(initial, 40))["outcome"], "conflict")
        current = await service.read_note(WorkspaceContext(f.owner, later_session), str(note_id))
        self.assertEqual(current["artifact"]["version_id"], newest["artifact"]["version_id"])
        self.assertEqual(current["content"]["blocks"][0]["rows"][0]["cells"]["reps"], 9)
        historical = await service.read_note(ctx(), note_id, initial)
        self.assertFalse(historical["artifact"]["is_current"])
        self.assertIsNone(historical["content"]["blocks"][0]["rows"][0]["cells"]["reps"])
        receipt = await service.operation_receipt(ctx(), correction_operation)
        self.assertTrue(receipt["effect_committed"])
        self.assertEqual(receipt["artifact"]["version_id"], corrected_version)

        # Extracted history/read seams must compose inside a shared guard; a
        # nested public exclusive read would deadlock this same owner.
        history = await service.list_versions(ctx(), note_id, limit=1)
        self.assertEqual(history["items"][0]["version_id"], newest["artifact"]["version_id"])
        second_page = await service.list_versions(ctx(), note_id, limit=1, cursor=history["next_cursor"])
        self.assertEqual(second_page["items"][0]["version_id"], corrected_version)
        with self.assertRaises(SourceError) as error:
            await notes.update_note(owner_id=f.owner, note_id=note_id, title=None, body="Raw overwrite")
        self.assertEqual(error.exception.code, "note_structured_update_required")

        restored = await service.restore_note(ctx(uuid4()), note_id, newest["artifact"]["version_id"], initial)
        restored_version = restored["artifact"]["version_id"]
        self.assertNotIn(restored_version, (initial, corrected_version, newest["artifact"]["version_id"]))
        with self.assertRaises(SourceError) as error:
            await service.list_versions(ctx(), note_id, limit=1, cursor=history["next_cursor"])
        self.assertEqual(error.exception.code, "knowledge_source_changed")
        noop = await edit(restored_version, None)
        self.assertEqual(noop["artifact"]["version_id"], restored_version)
        before = await service.read_snapshot(ctx(), note_id)
        # Null and literal 'unknown' have identical body text but distinct data.
        metadata_change = await edit(restored_version, "unknown")
        after = await service.read_snapshot(ctx(), note_id)
        self.assertEqual(before.note.body, after.note.body)
        self.assertNotEqual(before.source.document_id, after.source.document_id)
        self.assertEqual(after.view().content.blocks[0].rows[0].cells["reps"], "unknown")

        second = await service.create_note(ctx(uuid4()), "Workspace SQL fixture other", activity_content())
        with self.assertRaises(SourceError) as error:
            await service.read_note(ctx(), note_id, second["artifact"]["version_id"])
        self.assertEqual(error.exception.code, "note_version_mismatch")
        first_page = await service.list_notes(ctx(), query="Workspace SQL fixture", limit=1)
        next_page = await service.list_notes(ctx(), query="Workspace SQL fixture", limit=1, cursor=first_page["next_cursor"])
        self.assertEqual(len({first_page["items"][0]["artifact_id"], next_page["items"][0]["artifact_id"]}), 2)
        with self.assertRaises(SourceError) as error:
            await service.list_notes(ctx(), query="changed query", limit=1, cursor=first_page["next_cursor"])
        self.assertEqual(error.exception.code, "invalid_cursor")
        with self.assertRaises(SourceError):
            await service.read_note(WorkspaceContext(f.owner, foreign_session), note_id)

        # Inject an application failure after actual note/source SQL but before
        # receipt insertion. The original owner transaction must roll it back.
        original_transaction = f.sources.transaction
        class FailReceiptConnection:
            def __init__(self, connection): self.connection = connection
            def __getattr__(self, name): return getattr(self.connection, name)
            async def execute(self, sql, *args):
                if "insert into jarvis_note_operations" in sql:
                    raise RuntimeError("fixture receipt failure")
                return await self.connection.execute(sql, *args)
        @asynccontextmanager
        async def fail_transaction(owner, **kwargs):
            async with original_transaction(owner, **kwargs) as connection:
                yield FailReceiptConnection(connection)
        with patch.object(f.sources, "transaction", fail_transaction):
            with self.assertRaisesRegex(RuntimeError, "fixture receipt failure"):
                await edit(metadata_change["artifact"]["version_id"], 999)
        after_failure = await service.read_note(ctx(), note_id)
        self.assertEqual(after_failure["artifact"]["version_id"], metadata_change["artifact"]["version_id"])
        self.assertEqual(after_failure["content"]["blocks"][0]["rows"][0]["cells"]["reps"], "unknown")

        await service.delete_note(ctx(uuid4()), note_id, metadata_change["artifact"]["version_id"])
        replay_deleted = await service.create_note(ctx(create_operation), "Workspace SQL fixture", activity_content())
        self.assertTrue(replay_deleted["effect_committed"])
        self.assertEqual(replay_deleted["error"]["code"], "note_result_unavailable")
        with self.assertRaises(SourceError):
            await service.read_note(ctx(), note_id, initial)
        async with f.pool.acquire() as conn:
            self.assertFalse(await conn.fetchval("select exists(select 1 from jarvis_notes where owner_id=$1 and id=$2)", f.owner, note_id))
            self.assertTrue(await conn.fetchval("select exists(select 1 from jarvis_note_operations where owner_id=$1 and operation_id=$2)", f.owner, create_operation))


if __name__ == "__main__":
    unittest.main()
