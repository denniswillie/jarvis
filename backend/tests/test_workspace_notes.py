"""Domain behavior against a transactional in-memory source adapter.

These tests exercise operation/patch/read orchestration and rollback, not SQL
locking or actual persistence qualification. Core owns the separate SQL lane.
"""
import copy
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from app.knowledge.models import SourceError, SourceVersion
from app.knowledge.notes import SourceNoteRepository
from app.notes.models import JarvisNote
from app.notes.workspace_repository import WorkspaceNoteRepository, _decode_cursor, _encode_cursor
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.workspace.models import WorkspaceContext
from legacy_workspace_notes_fixture import WorkspaceNoteService


OWNER = "workspace-test"
AREA = uuid4()
SESSION = uuid4()


def content(text="Set one: observation uncertain."):
    return {"schema_version": 1, "blocks": [{"type": "text", "id": "text", "text": text}]}


class MemoryConnection:
    def __init__(self, sources, *, transaction_active=False):
        self.sources = sources
        self.transaction_active = transaction_active

    def is_in_transaction(self):
        return self.transaction_active

    async def fetchval(self, sql, *args):
        if "from jarvis_sessions" in sql:
            return args[0] == OWNER and args[1] in self.sources.sessions
        raise AssertionError(sql)

    async def fetchrow(self, sql, *args):
        s = self.sources
        if sql.startswith("select * from jarvis_note_operations"):
            return s.receipts.get(args[1])
        if sql.startswith("select * from jarvis_notes"):
            row = s.notes.get(args[1])
            return copy.deepcopy(row) if row else None
        if "insert into jarvis_notes" in sql:
            owner, session, title, body, metadata, note_id = args
            assert owner == OWNER and note_id not in s.notes
            now = datetime.now(timezone.utc)
            row = dict(id=note_id, session_id=session, title=title, body=body, metadata=json.loads(metadata),
                source="jarvis", created_at=now, updated_at=now)
            s.notes[note_id] = row
            return copy.deepcopy(row)
        if sql.startswith("update jarvis_notes"):
            _, note_id, title, body, metadata = args
            s.notes[note_id].update(title=title, body=body, metadata=json.loads(metadata), updated_at=datetime.now(timezone.utc))
            return copy.deepcopy(s.notes[note_id])
        raise AssertionError(sql)

    async def execute(self, sql, *args):
        if "insert into jarvis_note_operations" in sql:
            if self.sources.fail_receipt:
                raise RuntimeError("Injected receipt-write failure")
            keys = ("owner_id", "operation_id", "session_id", "area_id", "operation_kind", "parameters_digest", "artifact_id", "result_version_id")
            self.sources.receipts[args[1]] = dict(zip(keys, args)) | {"outcome": "completed"}
            return "INSERT 0 1"
        raise AssertionError(sql)


class MemorySources:
    def __init__(self):
        self.pool = object()
        self.notes, self.receipts, self.mappings, self.versions = {}, {}, {}, {}
        self.sessions = {SESSION}
        self.fail_receipt = False
        self.read_transactions, self.write_transactions = 0, 0

    @asynccontextmanager
    async def transaction(self, owner):
        self.write_transactions += 1
        before = copy.deepcopy((self.notes, self.receipts, self.mappings, self.versions))
        connection = MemoryConnection(self, transaction_active=True)
        try:
            yield connection
        except BaseException:
            self.notes, self.receipts, self.mappings, self.versions = before
            raise
        finally:
            connection.transaction_active = False

    @asynccontextmanager
    async def read_guard_transaction(self, owner):
        self.read_transactions += 1
        connection = MemoryConnection(self, transaction_active=True)
        try:
            yield connection
        finally:
            connection.transaction_active = False

    async def _area(self, conn, owner, area):
        assert owner == OWNER and area == AREA

    async def _read(self, conn, owner, area, source_id):
        mapping = next((v for v in self.mappings.values() if v["id"] == source_id), None)
        if mapping is None or mapping["status"] != "active":
            raise SourceError("knowledge_source_unavailable", "Unavailable")
        return self.versions[mapping["version_id"]].model_copy(update={"is_current": True})

    async def resolve_document_in_transaction(self, conn, owner, area, document_id, historical=False):
        source = self.versions.get(document_id)
        if source is None:
            raise SourceError("knowledge_source_not_found", "Unavailable")
        current = await self._read(conn, owner, area, source.source_id)
        return source.model_copy(update={"is_current": source.document_id == current.document_id})

    async def delete_source_in_transaction(self, conn, owner, area, source_id, expected_revision):
        note_id, mapping = next((key, value) for key, value in self.mappings.items() if value["id"] == source_id)
        self.notes.pop(note_id)
        mapping["status"] = "deleted"


class MemoryNotes(SourceNoteRepository):
    def __init__(self, sources):
        super().__init__(sources, owner_id=OWNER, area_id=AREA)

    async def _area_for(self, conn, note_id=None):
        return AREA

    async def _mapping(self, conn, note_id, required=True, area_id=None):
        mapping = self.sources.mappings.get(note_id)
        if mapping is None and required:
            raise SourceError("note_source_unavailable", "Unavailable")
        return mapping

    async def _expose(self, conn, note, area_id=None):
        mapping = await self._mapping(conn, note.id)
        current = await self.sources._read(conn, OWNER, AREA, mapping["id"])
        if current.content["record"] != note.model_dump(mode="json"):
            raise SourceError("note_original_changed", "Mismatch")
        metadata = current.original_evidence(); metadata.pop("content")
        return note.model_copy(update={"memory_source": metadata})

    async def _publish(self, conn, note, mapping, area_id=None):
        version_id = "source_" + uuid4().hex
        revision = mapping["current_revision"] + 1 if mapping else 1
        source_id = mapping["id"] if mapping else uuid4()
        source = SourceVersion(owner_id=OWNER, area_id=AREA, source_id=source_id, revision=revision,
            generation=1, payload_id=uuid4(), document_id=version_id, bank_name="test", content_digest="0" * 64,
            source_type="note", trust="external_content", source_time=note.updated_at, time_basis="test",
            recorded_at=datetime.now(timezone.utc), content={"record": note.model_dump(mode="json")},
            is_current=True, ingestion_status="not_queued")
        self.sources.versions[version_id] = source
        self.sources.mappings[note.id] = {"id": source_id, "current_revision": revision, "version_id": version_id, "status": "active"}
        return await self._expose(conn, note)


class WorkspaceNoteOperationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sources = MemorySources()
        self.notes = MemoryNotes(self.sources)
        self.service = WorkspaceNoteService(WorkspaceNoteRepository(self.notes))
        self.read_ctx = WorkspaceContext(OWNER, SESSION)

    def command_ctx(self, operation=None):
        return WorkspaceContext(OWNER, SESSION, operation or uuid4())

    async def create(self):
        return await self.service.create_note(self.command_ctx(), "Back day", content())

    async def patch(self, ref, text, ctx=None):
        return await self.service.patch_note(ctx or self.command_ctx(), ref["artifact_id"], ref["version_id"],
            [{"op": "set_text", "block_id": "text", "text": text}])

    async def test_create_correct_reopen_and_later_session_uses_current(self):
        first = await self.create()
        edited = await self.patch(first["artifact"], "Set one: eight reps, corrected by user.")
        self.assertEqual(first["artifact"]["artifact_id"], edited["artifact"]["artifact_id"])
        self.assertNotEqual(first["artifact"]["version_id"], edited["artifact"]["version_id"])
        later = uuid4(); self.sources.sessions.add(later)
        read = await self.service.read_note(WorkspaceContext(OWNER, later), UUID(edited["artifact"]["artifact_id"]))
        self.assertEqual(read["artifact"]["version_id"], edited["artifact"]["version_id"])
        self.assertEqual(read["content"]["blocks"][0]["text"], "Set one: eight reps, corrected by user.")
        self.assertEqual(self.sources.read_transactions, 1)

    async def test_lost_response_replay_returns_original_after_newer_edit(self):
        first = await self.create()
        ctx = self.command_ctx()
        corrected = await self.patch(first["artifact"], "Eight reps", ctx)
        newest = await self.patch(corrected["artifact"], "Nine reps")
        replay = await self.patch(first["artifact"], "Eight reps", ctx)
        self.assertEqual(replay["artifact"]["version_id"], corrected["artifact"]["version_id"])
        read = await self.service.read_note(self.read_ctx, UUID(first["artifact"]["artifact_id"]))
        self.assertEqual(read["artifact"]["version_id"], newest["artifact"]["version_id"])
        self.assertEqual(len(self.sources.versions), 3)

    async def test_stale_base_conflicts_without_mutation(self):
        first = await self.create()
        await self.patch(first["artifact"], "Eight reps")
        result = await self.patch(first["artifact"], "Twenty reps")
        self.assertEqual(result["outcome"], "conflict")
        self.assertFalse(result["effect_committed"])
        self.assertEqual(len(self.sources.versions), 2)
        self.assertEqual(len(self.sources.receipts), 2)

    async def test_same_operation_different_arguments_conflict(self):
        ctx = self.command_ctx()
        await self.service.create_note(ctx, "Back day", content())
        with self.assertRaises(SourceError) as caught:
            await self.service.create_note(ctx, "Different", content())
        self.assertEqual(caught.exception.code, "operation_conflict")
        self.assertEqual(len(self.sources.notes), 1)

    async def test_replay_after_delete_reports_prior_commit_without_recreation(self):
        ctx = self.command_ctx()
        first = await self.service.create_note(ctx, "Back day", content())
        ref = first["artifact"]
        await self.service.delete_note(self.command_ctx(), ref["artifact_id"], ref["version_id"])
        replay = await self.service.create_note(ctx, "Back day", content())
        self.assertTrue(replay["effect_committed"])
        self.assertEqual(replay["error"]["code"], "note_result_unavailable")
        self.assertEqual(len(self.sources.notes), 0)
        self.assertEqual(len(self.sources.receipts), 2)

    async def test_noop_receipt_does_not_make_a_version(self):
        first = await self.create()
        noop = await self.patch(first["artifact"], content()["blocks"][0]["text"])
        self.assertEqual(noop["artifact"]["version_id"], first["artifact"]["version_id"])
        self.assertEqual(len(self.sources.versions), 1)
        self.assertEqual(len(self.sources.receipts), 2)

    async def test_restore_publishes_new_head_preserving_identity_and_original_session(self):
        first = await self.create()
        edited = await self.patch(first["artifact"], "Nine reps")
        restored = await self.service.restore_note(self.command_ctx(), first["artifact"]["artifact_id"],
            edited["artifact"]["version_id"], first["artifact"]["version_id"])
        self.assertEqual(restored["artifact"]["artifact_id"], first["artifact"]["artifact_id"])
        self.assertNotIn(restored["artifact"]["version_id"], [first["artifact"]["version_id"], edited["artifact"]["version_id"]])
        note = self.sources.notes[UUID(first["artifact"]["artifact_id"])]
        self.assertEqual(note["session_id"], SESSION)
        self.assertEqual(note["body"], content()["blocks"][0]["text"])

    async def test_version_of_another_owned_note_is_rejected(self):
        first, second = await self.create(), await self.create()
        with self.assertRaises(SourceError) as caught:
            await self.service.read_note(self.read_ctx, UUID(first["artifact"]["artifact_id"]), second["artifact"]["version_id"])
        self.assertEqual(caught.exception.code, "note_version_mismatch")

    async def test_historical_read_keeps_historical_content_and_current_reference(self):
        first = await self.create(); edited = await self.patch(first["artifact"], "Eight reps")
        read = await self.service.read_note(self.read_ctx, UUID(first["artifact"]["artifact_id"]), first["artifact"]["version_id"])
        self.assertFalse(read["artifact"]["is_current"])
        self.assertEqual(read["artifact"]["current_version_id"], edited["artifact"]["version_id"])
        self.assertEqual(read["content"], content())

    async def test_wire_uuid_strings_work_for_exact_historical_read(self):
        first = await self.create(); await self.patch(first["artifact"], "Eight reps")
        read = await self.service.read_note(self.read_ctx, first["artifact"]["artifact_id"], first["artifact"]["version_id"])
        self.assertFalse(read["artifact"]["is_current"])

    async def test_historical_payload_cannot_impersonate_another_note(self):
        first = await self.create(); await self.patch(first["artifact"], "Eight reps")
        source = self.sources.versions[first["artifact"]["version_id"]]
        tampered = copy.deepcopy(source.content)
        tampered["record"]["id"] = str(uuid4())
        self.sources.versions[source.document_id] = source.model_copy(update={"content": tampered})
        with self.assertRaises(SourceError) as caught:
            await self.service.read_note(self.read_ctx, first["artifact"]["artifact_id"], source.document_id)
        self.assertEqual(caught.exception.code, "note_version_mismatch")

    async def test_invalid_read_identity_rejected_before_guard(self):
        for identity in (None, 123, "not-a-uuid"):
            with self.subTest(identity=identity), self.assertRaises(SourceError) as caught:
                await self.service.read_note(self.read_ctx, identity)
            self.assertEqual(caught.exception.code, "invalid_workspace_identity")
        self.assertEqual(self.sources.read_transactions, 0)

    async def test_snapshot_in_transaction_reuses_caller_guard_and_exact_history(self):
        first = await self.create()
        edited = await self.patch(first["artifact"], "Eight reps")
        async with self.sources.transaction(OWNER) as connection:
            counts = self.sources.read_transactions, self.sources.write_transactions
            historical = await self.service.read_snapshot_in_transaction(
                connection, self.read_ctx, first["artifact"]["artifact_id"], first["artifact"]["version_id"])
            current = await self.service.read_snapshot_in_transaction(
                connection, self.read_ctx, first["artifact"]["artifact_id"])
            self.assertEqual(historical.source.document_id, first["artifact"]["version_id"])
            self.assertEqual(historical.current.document_id, edited["artifact"]["version_id"])
            self.assertFalse(historical.source.is_current)
            self.assertEqual(historical.view().content.model_dump(mode="json"), content())
            self.assertEqual(current.source.document_id, edited["artifact"]["version_id"])
            self.assertTrue(current.source.is_current)
            self.assertEqual((self.sources.read_transactions, self.sources.write_transactions), counts)
            self.assertTrue(connection.is_in_transaction())
        self.assertFalse(connection.is_in_transaction())

    async def test_snapshot_in_transaction_requires_active_transaction(self):
        connection = MemoryConnection(self.sources)
        with self.assertRaisesRegex(RuntimeError, "requires an existing transaction"):
            await self.service.read_snapshot_in_transaction(connection, self.read_ctx, uuid4())
        self.assertEqual((self.sources.read_transactions, self.sources.write_transactions), (0, 0))

    async def test_snapshot_in_transaction_requires_trusted_owned_session(self):
        first = await self.create()
        async with self.sources.transaction(OWNER) as connection:
            with self.assertRaises(TypeError):
                await self.service.read_snapshot_in_transaction(
                    connection, {"owner_id": OWNER, "session_id": SESSION}, first["artifact"]["artifact_id"])
            for ctx, code in ((WorkspaceContext("another-owner", SESSION), "knowledge_owner_mismatch"),
                              (WorkspaceContext(OWNER, uuid4()), "workspace_session_unavailable")):
                with self.subTest(code=code), self.assertRaises(SourceError) as caught:
                    await self.service.read_snapshot_in_transaction(connection, ctx, first["artifact"]["artifact_id"])
                self.assertEqual(caught.exception.code, code)

    async def test_snapshot_in_transaction_validates_uuid_and_version(self):
        async with self.sources.transaction(OWNER) as connection:
            for artifact, version in (("invalid", None), (None, None), (uuid4(), ""),
                                      (uuid4(), 1), (uuid4(), "x" * 201), (uuid4(), "bad\x00version")):
                with self.subTest(artifact=artifact, version=version), self.assertRaises(SourceError) as caught:
                    await self.service.read_snapshot_in_transaction(connection, self.read_ctx, artifact, version)
                self.assertEqual(caught.exception.code, "invalid_workspace_identity")

    async def test_snapshot_in_transaction_rejects_wrong_note_and_missing_exact_version(self):
        first, second = await self.create(), await self.create()
        async with self.sources.transaction(OWNER) as connection:
            for version, code in ((second["artifact"]["version_id"], "note_version_mismatch"),
                                  ("missing-version", "knowledge_source_not_found")):
                with self.subTest(code=code), self.assertRaises(SourceError) as caught:
                    await self.service.read_snapshot_in_transaction(
                        connection, self.read_ctx, first["artifact"]["artifact_id"], version)
                self.assertEqual(caught.exception.code, code)

    async def test_snapshot_in_transaction_preserves_original_and_source_guards(self):
        first = await self.create()
        artifact = UUID(first["artifact"]["artifact_id"])
        async with self.sources.transaction(OWNER) as connection:
            original = copy.deepcopy(self.sources.notes[artifact])
            self.sources.notes[artifact]["body"] = "Untracked replacement"
            with self.assertRaises(SourceError) as caught:
                await self.service.read_snapshot_in_transaction(connection, self.read_ctx, artifact)
            self.assertEqual(caught.exception.code, "note_original_changed")
            self.sources.notes[artifact] = original
            self.sources.mappings[artifact]["status"] = "deleted"
            with self.assertRaises(SourceError) as caught:
                await self.service.read_snapshot_in_transaction(
                    connection, self.read_ctx, artifact, first["artifact"]["version_id"])
            self.assertEqual(caught.exception.code, "knowledge_source_unavailable")

    async def test_receipt_failure_rolls_back_original_and_published_version(self):
        self.sources.fail_receipt = True
        with self.assertRaisesRegex(RuntimeError, "receipt-write"):
            await self.create()
        self.assertEqual((self.sources.notes, self.sources.versions, self.sources.receipts), ({}, {}, {}))

    async def test_operation_read_can_reconcile_without_generic_executor(self):
        ctx = self.command_ctx()
        first = await self.service.create_note(ctx, "Back day", content())
        result = await self.service.operation_receipt(self.read_ctx, ctx.operation_id)
        self.assertTrue(result["effect_committed"])
        self.assertEqual(result["artifact"], first["artifact"])

    async def test_missing_admission_and_worker_invocation_rejected(self):
        with self.assertRaises(SourceError) as caught:
            await self.service.create_note(self.read_ctx, "No admission", content())
        self.assertEqual(caught.exception.code, "command_admission_required")
        with bind_tool_invocation(provider_call_id="worker-call", task=TaskToolInvocation(uuid4(), 1, uuid4())):
            with self.assertRaises(SourceError) as caught:
                await self.create()
        self.assertEqual(caught.exception.code, "structured_worker_not_granted")
        self.assertFalse(self.sources.notes)

    async def test_foreign_session_denied_before_read_or_mutation(self):
        ctx = WorkspaceContext(OWNER, uuid4(), uuid4())
        with self.assertRaises(SourceError) as caught:
            await self.service.create_note(ctx, "Foreign", content())
        self.assertEqual(caught.exception.code, "workspace_session_unavailable")
        self.assertFalse(self.sources.notes)


class WorkspaceCursorTests(unittest.TestCase):
    def test_cursor_binds_exact_scope_and_types(self):
        scope = {"owner": "owner", "session": str(uuid4()), "query": None, "limit": 1}
        cursor = _encode_cursor(scope | {"position": 3})
        self.assertEqual(_decode_cursor(cursor, scope, ("position",))["position"], 3)
        for changed in ({"owner": "other"}, {"query": "new"}, {"limit": True}):
            with self.subTest(changed=changed), self.assertRaises(SourceError):
                _decode_cursor(_encode_cursor(scope | changed | {"position": 3}), scope, ("position",))

    def test_malformed_and_oversized_cursor_is_domain_error(self):
        for value in ("!", "x" * 4097, None, _encode_cursor([]), _encode_cursor({"extra": "field"})):
            with self.subTest(value=str(value)[:40]), self.assertRaises(SourceError) as caught:
                _decode_cursor(value, {}, ())
            self.assertEqual(caught.exception.code, "invalid_cursor")


if __name__ == "__main__":
    unittest.main()
