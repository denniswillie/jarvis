"""Account-note domain/HTTP composition with isolated storage; no real services."""
import asyncio
import copy
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace
import unittest
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import httpx

from app.auth.static_bearer import JarvisPrincipal, require_static_bearer
from app.errors import JarvisHTTPError
from app.knowledge.models import SourceError
from app.knowledge.product_notes import NOTE_NAMESPACE
from app.notes.workspace_account_note import (
    ACCOUNT_NOTE_MARKER, account_note_edit_parameters, account_note_id,
)
from app.notes.workspace_content import parameters_digest
from app.notes.workspace_repository import WorkspaceNoteRepository
from app.tools.invocation import TaskToolInvocation, bind_tool_invocation
from app.workspace.models import WorkspaceContext
from app.workspace.router import router
from legacy_workspace_notes_fixture import WorkspaceNoteService

from test_workspace_notes import AREA, OWNER, SESSION, MemoryConnection, MemoryNotes, MemorySources, content


def append_text(identity="intro", text="Requested account information"):
    return [{"op": "append_block", "block": {"type": "text", "id": identity, "text": text}}]


class AccountConnection(MemoryConnection):
    async def fetchval(self, sql, *args):
        if "lock_jarvis_copy_sources" in sql:
            owner, identities = args
            assert owner == OWNER
            known = self.sources.originals | {source.payload_id for source in self.sources.versions.values()}
            return set(identities) <= known and not (set(identities) & self.sources.revoked)
        return await super().fetchval(sql, *args)

    async def executemany(self, sql, values):
        if "insert into jarvis_context_payload_dependencies" not in sql:
            raise AssertionError(sql)
        if self.sources.fail_dependencies:
            raise RuntimeError("Injected dependency failure")
        self.sources.dependencies.update(values)


class AccountSources(MemorySources):
    def __init__(self):
        super().__init__()
        self.owner_lock = asyncio.Lock()
        self.instruction = uuid4()
        self.originals = {self.instruction}
        self.dependencies, self.revoked = set(), set()
        self.fail_dependencies = False
        self.selected_area = AREA
        self.area_failure = None

    @asynccontextmanager
    async def transaction(self, owner):
        async with self.owner_lock:
            self.write_transactions += 1
            before = copy.deepcopy((self.notes, self.receipts, self.mappings, self.versions, self.dependencies))
            connection = AccountConnection(self, transaction_active=True)
            try:
                yield connection
            except BaseException:
                self.notes, self.receipts, self.mappings, self.versions, self.dependencies = before
                raise
            finally:
                connection.transaction_active = False

    @asynccontextmanager
    async def read_guard_transaction(self, owner):
        async with self.owner_lock:
            self.read_transactions += 1
            connection = AccountConnection(self, transaction_active=True)
            try:
                yield connection
            finally:
                connection.transaction_active = False

    async def _area(self, conn, owner, area):
        if self.area_failure:
            raise self.area_failure
        assert owner == OWNER and area == AREA

    async def _read(self, conn, owner, area, source_id):
        result = await super()._read(conn, owner, area, source_id)
        if result.payload_id in self.revoked:
            raise SourceError("knowledge_payload_unavailable", "Unavailable")
        return result


class AccountNotes(MemoryNotes):
    async def _area_for(self, conn, note_id=None):
        mapping = self.sources.mappings.get(note_id)
        return mapping["area_id"] if mapping else self.sources.selected_area

    async def _mapping(self, conn, note_id, required=True, area_id=None):
        mapping = await super()._mapping(conn, note_id, required, area_id)
        if mapping is not None and mapping["area_id"] != area_id:
            raise SourceError("note_source_scope", "Wrong area")
        return mapping

    async def _expose(self, conn, note, area_id=None):
        # The reused memory adapter calls _mapping without its area argument.
        mapping = self.sources.mappings[note.id]
        current = await self.sources._read(conn, OWNER, AREA, mapping["id"])
        if current.content["record"] != note.model_dump(mode="json"):
            raise SourceError("note_original_changed", "Mismatch")
        metadata = current.original_evidence(); metadata.pop("content")
        return note.model_copy(update={"memory_source": metadata})

    async def _publish(self, conn, note, mapping, area_id=None):
        assert area_id == AREA
        result = await super()._publish(conn, note, mapping, area_id)
        self.sources.mappings[note.id].update(namespace=NOTE_NAMESPACE, source_key=str(note.id), area_id=AREA)
        return result


class AccountNoteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sources = AccountSources()
        self.notes = AccountNotes(self.sources)
        self.service = WorkspaceNoteService(WorkspaceNoteRepository(self.notes))
        self.ctx = WorkspaceContext(OWNER, SESSION)
        self.guarded = []

    def edit_ctx(self, operation=None, session=SESSION):
        return WorkspaceContext(OWNER, session, operation or uuid4())

    async def guard(self, connection, current, candidate):
        self.assertTrue(connection.is_in_transaction())
        self.assertEqual(candidate.artifact_id, account_note_id(OWNER))
        self.assertEqual(candidate.parameters_digest, parameters_digest(account_note_edit_parameters(
            OWNER, candidate.base_version_id, candidate.changes)))
        # Read accessors are copies: changing a guard's view cannot change commit.
        changes = candidate.changes
        changes.clear()
        self.guarded.append((current, candidate))
        return (self.sources.instruction,)

    async def first(self, ctx=None, changes=None):
        return await self.service.publish_account_edit(ctx or self.edit_ctx(), None,
                                                      changes or append_text(), self.guard)

    async def patch(self, version, text="Corrected", ctx=None, guard=None):
        return await self.service.publish_account_edit(ctx or self.edit_ctx(), version,
            [{"op": "set_text", "block_id": "intro", "text": text}], guard or self.guard)

    async def test_absent_read_is_null_and_never_initializes(self):
        self.assertIsNone(await self.service.read_account_note(self.ctx))
        self.assertEqual(self.sources.write_transactions, 0)
        self.assertFalse(self.sources.notes or self.sources.versions or self.sources.receipts)
        with self.assertRaises(SourceError):
            await self.service.read_account_note(self.ctx, "source_missing")

    async def test_owner_current_read_needs_no_session_and_does_not_bootstrap(self):
        self.sources.sessions.clear()
        self.assertIsNone(await self.service.read_current_account_note(OWNER))
        self.assertEqual(self.sources.write_transactions, 0)
        self.assertFalse(self.sources.notes or self.sources.versions or self.sources.receipts or self.sources.sessions)
        with self.assertRaises(SourceError):
            await self.service.read_current_account_note("foreign")
        # This does not make a missing session usable for the existing APIs.
        with self.assertRaises(SourceError):
            await self.service.read_account_note(self.ctx)
        with self.assertRaises(SourceError):
            await self.service.publish_account_edit(self.edit_ctx(), None, append_text(), self.guard)

    async def test_owner_current_read_preserves_current_version_and_source_guards(self):
        first = await self.first()
        corrected = await self.patch(first["artifact"]["version_id"])
        self.sources.sessions.clear()
        writes = self.sources.write_transactions
        view = await self.service.read_current_account_note(OWNER)
        self.assertEqual(view["artifact"]["version_id"], corrected["artifact"]["version_id"])
        self.assertEqual(view["artifact"]["version_id"], view["artifact"]["current_version_id"])
        self.assertTrue(view["artifact"]["is_current"])
        self.assertEqual(view["content"]["blocks"][0]["text"], "Corrected")
        self.assertEqual(self.sources.write_transactions, writes)
        self.sources.mappings[account_note_id(OWNER)]["status"] = "forgotten"
        with self.assertRaises(SourceError):
            await self.service.read_current_account_note(OWNER)

    async def test_first_edit_uses_default_area_and_one_account_identity_across_sessions(self):
        self.sources.selected_area = uuid4()
        first = await self.first()
        identity = account_note_id(OWNER)
        self.assertNotEqual(identity, account_note_id("another-owner"))
        self.assertEqual(first["artifact"]["artifact_id"], str(identity))
        self.assertIsNone(self.sources.notes[identity]["session_id"])
        self.assertEqual(self.sources.notes[identity]["metadata"][ACCOUNT_NOTE_MARKER], "v1")
        self.assertEqual(len(self.sources.versions), 1)  # No separately visible empty version.
        later = uuid4(); self.sources.sessions.add(later)
        view = await self.service.read_account_note(WorkspaceContext(OWNER, later))
        self.assertEqual(view["artifact"]["version_id"], first["artifact"]["version_id"])
        self.assertEqual(view["artifact"]["version_id"], view["artifact"]["current_version_id"])
        self.assertEqual(view["artifact"]["revision"], view["artifact"]["current_revision"])
        self.assertTrue(view["artifact"]["is_current"])

    async def test_existing_unrelated_notes_are_preserved_not_adopted(self):
        legacy = await self.service.create_note(self.edit_ctx(), "Old saved note", content("Leave untouched"))
        before = copy.deepcopy(self.sources.notes[UUID(legacy["artifact"]["artifact_id"])])
        self.assertIsNone(await self.service.read_account_note(self.ctx))
        await self.first()
        self.assertEqual(self.sources.notes[before["id"]], before)
        self.assertEqual(len(self.sources.notes), 2)

    async def test_duplicate_concurrent_first_publication_reconciles_without_second_guard(self):
        ctx = self.edit_ctx()
        first, replay = await asyncio.gather(self.first(ctx), self.first(ctx))
        self.assertEqual(first["artifact"]["version_id"], replay["artifact"]["version_id"])
        self.assertEqual((len(self.sources.notes), len(self.sources.versions), len(self.sources.receipts), len(self.guarded)),
                         (1, 1, 1, 1))

    async def test_concurrent_distinct_first_edits_cannot_silently_overwrite(self):
        results = await asyncio.gather(self.first(changes=append_text("a", "A")),
                                       self.first(changes=append_text("b", "B")))
        self.assertEqual(sorted(result["outcome"] for result in results), ["completed", "conflict"])
        self.assertEqual((len(self.sources.versions), len(self.sources.receipts), len(self.guarded)), (1, 1, 1))

    async def test_exact_correction_history_and_unaddressed_blocks(self):
        first = await self.first(changes=append_text() + append_text("other", "Keep this information"))
        second = await self.patch(first["artifact"]["version_id"])
        current = await self.service.read_account_note(self.ctx)
        old = await self.service.read_account_note(self.ctx, first["artifact"]["version_id"])
        self.assertEqual(current["content"]["blocks"][0]["text"], "Corrected")
        self.assertEqual(current["content"]["blocks"][1], old["content"]["blocks"][1])
        self.assertFalse(old["artifact"]["is_current"])
        self.assertEqual(old["artifact"]["current_version_id"], second["artifact"]["version_id"])
        self.assertNotEqual(current["content"], old["content"])

    async def test_stale_base_is_conflict_before_guard_and_missing_base_is_not_bootstrap_again(self):
        first = await self.first()
        await self.patch(first["artifact"]["version_id"])
        for base in (first["artifact"]["version_id"], None):
            count = len(self.guarded)
            result = await self.service.publish_account_edit(self.edit_ctx(), base, append_text("late"), self.guard)
            self.assertEqual(result["outcome"], "conflict")
            self.assertFalse(result["effect_committed"])
            self.assertEqual(len(self.guarded), count)
        self.assertEqual(len(self.sources.versions), 2)

    async def test_receipt_replay_precedes_new_base_and_guard_checks(self):
        ctx = self.edit_ctx(); first = await self.first(ctx)
        await self.patch(first["artifact"]["version_id"])
        async def deny(*_):
            self.fail("An existing receipt must not reissue or reauthorize a new effect")
        replay = await self.service.publish_account_edit(ctx, None, append_text(), deny)
        self.assertEqual(replay["artifact"]["version_id"], first["artifact"]["version_id"])
        self.assertEqual(len(self.sources.versions), 2)
        with self.assertRaises(SourceError) as caught:
            await self.service.publish_account_edit(ctx, None, append_text(text="Changed candidate"), deny)
        self.assertEqual(caught.exception.code, "operation_conflict")

    async def test_receipt_survives_unavailable_original_without_resurrection(self):
        ctx = self.edit_ctx(); await self.first(ctx)
        identity = account_note_id(OWNER)
        self.sources.mappings[identity]["status"] = "deleted"
        self.sources.notes.pop(identity)
        replay = await self.service.publish_account_edit(ctx, None, append_text(), None)
        self.assertTrue(replay["effect_committed"])
        self.assertEqual(replay["error"]["code"], "note_result_unavailable")
        self.assertFalse(self.sources.notes)
        with self.assertRaises(SourceError):
            await self.first()

    async def test_tombstone_forget_and_missing_original_are_errors_not_empty(self):
        await self.first(); identity = account_note_id(OWNER)
        for state in ("deleted", "forgotten"):
            self.sources.mappings[identity]["status"] = state
            with self.subTest(state=state), self.assertRaises(SourceError):
                await self.service.read_account_note(self.ctx)
        self.sources.mappings[identity]["status"] = "active"
        self.sources.notes.pop(identity)
        with self.assertRaises(SourceError):
            await self.service.read_account_note(self.ctx)

    async def test_conflicting_marker_mapping_and_untracked_collision_are_rejected(self):
        await self.first(); identity = account_note_id(OWNER)
        self.sources.notes[identity]["metadata"].pop(ACCOUNT_NOTE_MARKER)
        with self.assertRaisesRegex(SourceError, "adopted"):
            await self.service.read_account_note(self.ctx)
        self.sources.notes[identity]["metadata"][ACCOUNT_NOTE_MARKER] = "v1"
        self.sources.mappings[identity]["namespace"] = "legacy:m03:v1"
        with self.assertRaises(SourceError):
            await self.service.read_account_note(self.ctx)
        self.sources.mappings.pop(identity)
        with self.assertRaises(SourceError):
            await self.first()
        self.assertEqual(len(self.sources.notes), 1)

    async def test_owner_session_and_original_authorization_still_apply(self):
        for ctx in (WorkspaceContext("foreign", SESSION), WorkspaceContext(OWNER, uuid4())):
            with self.subTest(ctx=ctx), self.assertRaises(SourceError):
                await self.service.read_account_note(ctx)
        ctx = self.edit_ctx(); result = await self.first(ctx)
        self.sources.revoked.add(self.sources.versions[result["artifact"]["version_id"]].payload_id)
        with self.assertRaises(SourceError):
            await self.service.read_account_note(self.ctx)
        def denied(_):
            raise SourceError("access_revoked", "Access revoked", 403)
        self.notes.access_check = denied
        with self.assertRaises(SourceError):
            await self.service.publish_account_edit(ctx, None, append_text(), self.guard)

    async def test_guard_denial_or_invalid_evidence_rolls_back_first_initialization(self):
        async def denied(*_):
            raise SourceError("task_claim_lost", "Not admitted", 403)
        async def no_evidence(*_):
            return ()
        for guard in (None, denied, no_evidence):
            with self.subTest(guard=guard), self.assertRaises(SourceError):
                await self.service.publish_account_edit(self.edit_ctx(), None, append_text(), guard)
            self.assertFalse(self.sources.notes or self.sources.versions or self.sources.receipts)

    async def test_dependencies_bind_instruction_and_previous_note_with_the_commit(self):
        first = await self.first()
        original = self.sources.versions[first["artifact"]["version_id"]].payload_id
        self.assertIn((OWNER, self.sources.instruction, original), self.sources.dependencies)
        second = await self.patch(first["artifact"]["version_id"])
        latest = self.sources.versions[second["artifact"]["version_id"]].payload_id
        self.assertTrue({(OWNER, self.sources.instruction, latest), (OWNER, original, latest)} <= self.sources.dependencies)

    async def test_source_withdrawal_during_guard_prevents_commit(self):
        async def revoked(connection, current, candidate):
            self.sources.revoked.add(self.sources.instruction)
            return (self.sources.instruction,)
        with self.assertRaises(SourceError) as caught:
            await self.service.publish_account_edit(self.edit_ctx(), None, append_text(), revoked)
        self.assertEqual(caught.exception.code, "note_publication_evidence_unavailable")
        self.assertFalse(self.sources.notes or self.sources.versions or self.sources.receipts)

    async def test_dependency_and_receipt_failures_roll_back_note_and_source_versions(self):
        for flag in ("fail_dependencies", "fail_receipt"):
            setattr(self.sources, flag, True)
            with self.subTest(flag=flag), self.assertRaises(RuntimeError):
                await self.first()
            setattr(self.sources, flag, False)
            self.assertFalse(self.sources.notes or self.sources.versions or self.sources.receipts or self.sources.dependencies)

    async def test_noop_records_receipt_without_mutating_old_version_provenance(self):
        first = await self.first()
        before = set(self.sources.dependencies)
        result = await self.patch(first["artifact"]["version_id"], text="Requested account information")
        self.assertEqual(result["artifact"]["version_id"], first["artifact"]["version_id"])
        self.assertEqual(self.sources.dependencies, before)
        self.assertEqual((len(self.sources.versions), len(self.sources.receipts)), (1, 2))

    async def test_in_transaction_read_never_starts_another_transaction(self):
        await self.first()
        before = (self.sources.read_transactions, self.sources.write_transactions)
        connection = AccountConnection(self.sources, transaction_active=True)
        snapshot = await self.service.read_account_snapshot_in_transaction(connection, self.ctx)
        self.assertEqual(snapshot.note.id, account_note_id(OWNER))
        self.assertEqual(before, (self.sources.read_transactions, self.sources.write_transactions))
        connection.transaction_active = False
        with self.assertRaises(RuntimeError):
            await self.service.read_account_snapshot_in_transaction(connection, self.ctx)

    async def test_foreground_mutate_still_denies_workers_and_publisher_requires_its_guard(self):
        invocation = TaskToolInvocation(task_id=uuid4(), task_revision=1, attempt_id=uuid4())
        with bind_tool_invocation(provider_call_id="account-note-worker-test", task=invocation):
            with self.assertRaises(SourceError) as caught:
                await self.service.create_note(self.edit_ctx(), "Bypass", content())
            self.assertEqual(caught.exception.code, "structured_worker_not_granted")
            with self.assertRaises(SourceError):
                await self.service.publish_account_edit(self.edit_ctx(), None, append_text(), None)
            result = await self.first()
            self.assertTrue(result["effect_committed"])
        self.assertEqual(len(self.guarded), 1)

    def app(self):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[require_static_bearer] = lambda: JarvisPrincipal(OWNER)
        app.state.runtime_services = SimpleNamespace(settings=SimpleNamespace(JARVIS_OWNER_ID=OWNER),
                                                    workspace_notes=self.service)
        @app.exception_handler(JarvisHTTPError)
        async def expected_error(request, error):
            return JSONResponse({"error": {"code": error.code}}, status_code=error.status_code)
        return app


if __name__ == "__main__":
    unittest.main()
