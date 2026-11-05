"""Receipt-backed command/note projection with the existing offline SQL fixture.

Real command decoding, task receipt checks, note-result validation, source-root
guards and snapshot projection run. Only the note grant owner and saved-note
commit-proof reads are stubbed; this is not SQL locking/grant qualification.
"""
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

from app.knowledge.sources import SourceRepository
from app.live_state.admission import CommandAdmission, command_parameters, task_command_source
from app.live_state.canonical import CanonicalContextBinder
from app.live_state.models import SessionScope
from app.live_state.task_context import TaskContextProjector
from app.notes.workspace_account_note import account_note_id
from app.runtime.feature_config import TaskLimits
from app.tasks.context_events import ContextRecord
from app.tasks.models import TaskCreate, TaskError, digest
from app.tasks.note_result import NoteResult
from app.tasks.repository import TaskRepository

from test_live_state_task_context import Connection, Database, NOW


class CommandDatabase(Database):
    def __init__(self, scope):
        super().__init__(scope)
        self.command_receipts = {}

    @asynccontextmanager
    async def acquire(self):
        yield CommandConnection(self)

    def command_task(self, *, status="running", result=False):
        identity = self.task(status=status, area=self.scope.area_id,
                             answer="Unused fixture placeholder" if result else None)
        task = self.tasks[identity]
        instruction = "Append the observed results to my note."
        command = CommandAdmission(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            provider_session_key="fixture-provider", provider_call_id=str(uuid4()),
            tool_name="live_note", capability="tasks.create",
            normalized_arguments={"instruction": instruction})
        create = TaskCreate(request_id=command.operation_id, session_id=self.scope.session_id,
            knowledge_area_id=self.scope.area_id, title="Edit account note",
            requested_outcome=instruction, edit_note=True)
        body = {"format": "jarvis.task_command.v1", "command": command.parameters(),
            "operation": "create", "task_id": str(identity), "base_revision": 0,
            "area_id": str(self.scope.area_id), "request": create.model_dump(mode="json"),
            "mode": "background", "parameters_digest": digest(command_parameters(command))}
        event = self.events[task["request_event_id"]]
        event.update(source_namespace="voice_command", source_event_id=str(command.operation_id),
                     kind="command_request", trust="derived", content=body, content_digest=digest(body))
        self.payloads[event["payload_id"]]["content"] = body
        self.payloads[task["description_payload_id"]]["content"].update(
            title="Edit account note", requested_outcome=instruction)
        task.update(revision=1, title="Edit account note", requested_outcome=instruction)
        self.command_receipts[command.operation_id] = {"task_id": identity, "operation": "create",
            "parameters_digest": body["parameters_digest"], "accepted_revision": 1}
        if result:
            self.receipts[identity]["revision"] = 1
        return identity, command


class CommandConnection(Connection):
    async def fetchrow(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("select * from jarvis_task_requests"):
            self.query(sql, args)
            if args[0] != self.db.scope.owner_id:
                return None
            return self.db.command_receipts.get(args[1])
        return await super().fetchrow(sql, *args)

    async def fetchval(self, sql, *args):
        normalized = " ".join(sql.lower().split())
        if normalized.startswith("select count(distinct d.source_payload_id)"):
            self.query(sql, args)
            owner, child, roots = args
            return len({root for root in roots if root in self.db.parents.get(child, ())
                        and self.db.available(owner, root)})
        return await super().fetchval(sql, *args)


class NoteTaskContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.scope = SessionScope(owner_id="note-task-context-fixture", session_id=uuid4(), area_id=uuid4())
        self.db = CommandDatabase(self.scope)
        self.repository = TaskRepository(self.db, TaskLimits())
        self.binder = CanonicalContextBinder(SourceRepository(self.db), self.scope, clock=lambda: NOW)
        self.projector = TaskContextProjector(self.binder, self.repository)
        # Grant-family SQL belongs to its existing owner. Keep real command
        # representation and receipt validation; do not mock task_command_source.
        target = patch("app.live_state.admission.require_command_target", new_callable=AsyncMock)
        self.target = target.start()
        self.addCleanup(target.stop)

    async def project(self, identity):
        return await self.projector.project((identity,), received_at=NOW,
                                            expires_at=NOW + timedelta(seconds=10))

    def note_result(self, *, committed=True, outcome="completed"):
        status = "succeeded" if committed else {
            "failed": "failed", "clarification_required": "waiting_for_input",
            "paused": "paused", "outcome_unknown": "outcome_unknown",
        }[outcome]
        identity, command = self.db.command_task(status=status, result=True)
        task = self.db.tasks[identity]
        request = self.db.events[task["request_event_id"]]
        source = SimpleNamespace(document_id="saved-v1", source_id=uuid4(), revision=1,
                                 payload_id=self.db.payload({"note": "Saved version"}), is_current=True)
        receipt = self.db.payload({"receipt": "Saved note action"})
        fields = dict(publication_operation_id=uuid4(), version_id=source.document_id,
            source_id=source.source_id, source_revision=source.revision,
            source_payload_id=source.payload_id, action_receipt_payload_id=receipt) if committed else {}
        result = NoteResult(owner_id=self.scope.owner_id, task_id=identity, task_revision=1,
            attempt_id=self.db.receipts[identity]["result_attempt_id"], operation_id=command.operation_id,
            request_payload_id=request["payload_id"], artifact_id=account_note_id(self.scope.owner_id),
            area_id=self.scope.area_id, effect_committed=committed, outcome=outcome, **fields)
        self.db.payloads[task["result_payload_id"]]["content"] = result.model_dump(mode="json")
        self.db.parents[task["result_payload_id"]] = {request["payload_id"]}
        if committed:
            self.db.parents[task["result_payload_id"]].update((source.payload_id, receipt))
            self.db.parents[receipt] = {source.payload_id}
        return identity, result, source, {"receipt_payload_id": receipt}

    async def test_exact_command_receipt_projects_without_reclassifying_derived_input(self):
        identity, command = self.db.command_task()
        request = self.db.events[self.db.tasks[identity]["request_event_id"]]
        self.assertFalse(ContextRecord.model_validate(request).authorizes_request)
        with patch("app.live_state.task_context.task_command_source", wraps=task_command_source) as proof:
            entry, = await self.project(identity)
            self.assertEqual(entry.status, "running")
            self.assertEqual(entry.task.revision, 1)
            self.assertTrue(await self.binder.available(entry.guard_binding))
        self.assertTrue(proof.await_args_list)
        self.assertTrue(all(call.kwargs == {"lock": False} for call in proof.await_args_list))
        self.assertFalse(ContextRecord.model_validate(request).authorizes_request)
        self.assertTrue(self.target.await_args_list)
        self.assertTrue(any(call.kwargs == {"lock": False} for call in self.target.await_args_list))
        command_reads = [sql for sql, _ in self.db.queries if sql.startswith("select e.*,p.availability")]
        self.assertTrue(command_reads)
        self.assertTrue(any("for share of e,p" not in sql for sql in command_reads))
        root = entry.guard_binding.payload_root_ids[0]
        self.assertEqual(self.db.parents[root], {request["payload_id"], self.db.tasks[identity]["description_payload_id"]})
        self.assertEqual(self.db.event_dependencies[root], [(request["id"], request["revision"])])
        self.assertIn(command.operation_id, self.db.command_receipts)

    async def test_missing_mismatched_and_stale_command_receipts_are_not_authority(self):
        for change in ("missing", "task_id", "operation", "parameters_digest", "accepted_revision"):
            with self.subTest(change=change):
                identity, command = self.db.command_task()
                receipt = self.db.command_receipts[command.operation_id]
                if change == "missing":
                    del self.db.command_receipts[command.operation_id]
                else:
                    receipt[change] = {"task_id": uuid4(), "operation": "cancel",
                        "parameters_digest": "b" * 64, "accepted_revision": 2}[change]
                self.assertEqual(await self.project(identity), ())

    async def test_arbitrary_derived_relabelled_and_withdrawn_requests_stay_rejected(self):
        for changes in ({"source_namespace": "observer", "kind": "observation"},
                        {"is_current": False}, {"trust": "user_input"},
                        {"finality": "provisional"}, {"revision": 2},
                        {"owner_id": "other-owner"}, {"session_id": uuid4()}):
            with self.subTest(changes=changes):
                identity, _ = self.db.command_task()
                self.db.events[self.db.tasks[identity]["request_event_id"]].update(changes)
                self.assertEqual(await self.project(identity), ())
        identity, _ = self.db.command_task()
        self.db.remove(self.db.events[self.db.tasks[identity]["request_event_id"]]["payload_id"])
        self.assertEqual(await self.project(identity), ())

    async def test_changed_task_scope_revision_or_target_authority_is_rejected(self):
        for changes in ({"owner_id": "other-owner"}, {"session_id": uuid4()},
                        {"knowledge_area_id": uuid4()}, {"request_revision": 2}, {"revision": 2}):
            with self.subTest(changes=changes):
                identity, _ = self.db.command_task()
                self.db.tasks[identity].update(changes)
                self.assertEqual(await self.project(identity), ())
        identity, _ = self.db.command_task()
        self.target.side_effect = TaskError("task_command_scope_invalid", "Grant family revoked", 403)
        self.assertEqual(await self.project(identity), ())

    async def test_withdrawn_receipt_invalidates_an_already_published_snapshot(self):
        identity, command = self.db.command_task()
        entry, = await self.project(identity)
        del self.db.command_receipts[command.operation_id]
        self.assertFalse(await self.binder.available(entry.guard_binding))
        self.assertEqual(await self.project(identity), ())

    async def test_runtime_receipt_is_not_worker_prose_and_tracks_historical_currentness(self):
        identity, result, source, saved = self.note_result()
        with patch("app.tasks.note_result.commit_proof", AsyncMock(return_value=(saved, source))):
            first, = await self.project(identity)
            self.assertIn("Runtime receipt result", first.summary)
            self.assertIn("Your note edit was saved.", first.summary)
            self.assertNotIn("Worker-authored", first.summary)
            self.assertIn("delivery/heard state is unknown", first.summary)
            self.assertTrue(await self.binder.available(first.guard_binding))
            source.is_current = False
            self.assertFalse(await self.binder.available(first.guard_binding))
            later, = await self.project(identity)
            self.assertIn("since been updated", later.summary)
            self.assertNotEqual(first.guard_binding, later.guard_binding)
            self.assertTrue(await self.binder.available(later.guard_binding))
        self.assertEqual(self.db.payloads[self.db.tasks[identity]["result_payload_id"]]["content"],
                         result.model_dump(mode="json"))

    async def test_uncommitted_results_report_runtime_status_without_claiming_a_save(self):
        for outcome in ("failed", "clarification_required", "paused", "outcome_unknown"):
            with self.subTest(outcome=outcome):
                identity, _, _, _ = self.note_result(committed=False, outcome=outcome)
                with patch("app.tasks.note_result.commit_proof", new_callable=AsyncMock) as proof:
                    entry, = await self.project(identity)
                proof.assert_not_awaited()
                self.assertIn("Runtime receipt result", entry.summary)
                self.assertNotIn("Worker-authored", entry.summary)
                self.assertNotIn("Your note edit was saved.", entry.summary)
                self.assertTrue(await self.binder.available(entry.guard_binding))

    async def test_bad_result_identity_authorship_and_attempt_never_claim_a_save(self):
        for key, value in (("owner_id", "other-owner"), ("task_id", str(uuid4())),
                           ("task_revision", 9), ("attempt_id", str(uuid4())),
                           ("area_id", str(uuid4())), ("artifact_id", str(uuid4())),
                           ("authorship", {"kind": "expert"}), ("authored_by", "worker"),
                           ("answer", "The worker says it saved the note.")):
            with self.subTest(key=key):
                identity, _, source, saved = self.note_result()
                payload = self.db.payloads[self.db.tasks[identity]["result_payload_id"]]["content"]
                payload[key] = value
                with patch("app.tasks.note_result.commit_proof", AsyncMock(return_value=(saved, source))):
                    entry, = await self.project(identity)
                self.assertIn("not voice-ready", entry.summary)
                self.assertNotIn("was saved", entry.summary)
                self.assertNotIn("worker says", entry.summary)

    async def test_receipt_root_withdrawal_and_missing_lineage_revoke_saved_claims(self):
        for original in ("request", "description", "result", "manifest", "saved_note", "action_receipt"):
            with self.subTest(original=original):
                identity, result, source, saved = self.note_result()
                task = self.db.tasks[identity]
                root = {"request": result.request_payload_id, "description": task["description_payload_id"],
                    "result": task["result_payload_id"], "manifest": self.db.receipts[identity]["manifest_payload_id"],
                    "saved_note": result.source_payload_id, "action_receipt": result.action_receipt_payload_id}[original]
                with patch("app.tasks.note_result.commit_proof", AsyncMock(return_value=(saved, source))):
                    entry, = await self.project(identity)
                    self.db.remove(root)
                    self.assertFalse(await self.binder.available(entry.guard_binding))
                    entries = await self.project(identity)
                self.assertFalse(any("was saved" in item.summary for item in entries))
        identity, result, source, saved = self.note_result()
        self.db.parents[self.db.tasks[identity]["result_payload_id"]].remove(result.action_receipt_payload_id)
        with patch("app.tasks.note_result.commit_proof", AsyncMock(return_value=(saved, source))):
            entry, = await self.project(identity)
        self.assertIn("not voice-ready", entry.summary)
        self.assertNotIn("was saved", entry.summary)

    async def test_canonical_proof_failure_and_missing_task_receipt_never_expose_result(self):
        identity, _, _, _ = self.note_result()
        with patch("app.tasks.note_result.commit_proof", AsyncMock(
                side_effect=TaskError("note_result_unavailable", "Receipt unavailable"))):
            entry, = await self.project(identity)
        self.assertIn("not voice-ready", entry.summary)
        del self.db.receipts[identity]
        entry, = await self.project(identity)
        self.assertIn("unavailable", entry.summary)
        self.assertNotIn("was saved", entry.summary)


if __name__ == "__main__":
    unittest.main()
