"""Receipt-derived result/runner boundaries; no real database or provider."""
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.notes.workspace_account_note import account_note_id
from app.tasks.models import Reconciliation, TaskStatus, WorkerOutcome, digest
from app.tasks.hermes_result import inspect_expert_output
from app.tasks.note_result import NoteResult, commit_proof, note_result_source
from app.tasks.runner import TaskRunner


@asynccontextmanager
async def scope(value):
    yield value


class NoteResultTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.task_id, self.operation = "fixture-note-owner", uuid4(), uuid4()
        self.artifact, self.area = account_note_id(self.owner), uuid4()
        self.saved = NS(document_id="saved-v1", source_id=uuid4(), revision=1,
                        payload_id=uuid4(), is_current=True)
        self.action_payload = uuid4()
        self.scope = {"task_id": str(self.task_id), "artifact_id": str(self.artifact), "candidate_digest": "c" * 64}
        self.row = {"operation_id": self.operation, "operation_kind": "patch", "artifact_id": self.artifact,
            "parameters_digest": "c" * 64, "result_version_id": "saved-v1", "area_id": self.area,
            "receipt_payload_id": self.action_payload, "task_revision": 1, "scope": self.scope,
            "scope_availability": "available", "action_availability": "available", "source_id": self.saved.source_id,
            "action_receipt": {"tool": "edit_account_note", "parameters_digest": digest(self.scope), "result": {
                "outcome": "completed", "effect_committed": True, "operation_id": str(self.operation),
                "artifact": {"artifact_id": str(self.artifact), "version_id": "saved-v1"}}}}
        self.result = NoteResult(owner_id=self.owner, task_id=self.task_id, task_revision=1, attempt_id=uuid4(),
            operation_id=uuid4(), request_payload_id=uuid4(), artifact_id=self.artifact, area_id=self.area,
            effect_committed=True, outcome="completed", publication_operation_id=self.operation, version_id="saved-v1",
            source_id=self.saved.source_id, source_revision=1, source_payload_id=self.saved.payload_id,
            action_receipt_payload_id=self.action_payload)
        self.task = NS(owner_id=self.owner, id=self.task_id, revision=1, knowledge_area_id=self.area)

    async def proof(self, row=None, bound=True):
        conn = NS(fetchrow=AsyncMock(side_effect=[self.row if row is None else row, {"source": "original"}]),
                  fetchval=AsyncMock(return_value=bound))
        with patch("app.tasks.note_result.SourceRepository._version", return_value=self.saved):
            return await commit_proof(conn, self.owner, self.task_id, self.operation)

    async def test_exact_domain_action_source_and_dependency_proof(self):
        row, saved = await self.proof()
        self.assertEqual(row["operation_id"], self.operation)
        self.assertIs(saved, self.saved)

    async def test_conflict_forged_target_digest_and_detached_receipt_fail(self):
        for change in ("no_effect", "wrong_artifact", "wrong_digest", "wrong_version", "withdrawn"):
            with self.subTest(change=change):
                row = deepcopy(self.row)
                result = row["action_receipt"]["result"]
                if change == "no_effect": result.update(effect_committed=False, outcome="conflict")
                if change == "wrong_artifact": row["artifact_id"] = uuid4()
                if change == "wrong_digest": row["action_receipt"]["parameters_digest"] = "b" * 64
                if change == "wrong_version": result["artifact"]["version_id"] = "current-v2"
                if change == "withdrawn": row["action_availability"] = "removed"
                with self.assertRaises(Exception): await self.proof(row)
        with self.assertRaises(Exception): await self.proof(bound=False)

    async def source(self, payload=None, linked=3):
        conn = NS(fetchval=AsyncMock(return_value=linked))
        with patch("app.tasks.note_result.commit_proof", AsyncMock(return_value=(self.row, self.saved))):
            return await note_result_source(conn, {"source_payload_id": uuid4()}, self.task,
                                             payload or self.result.model_dump(mode="json"))

    async def test_notification_is_runtime_and_marks_later_correction_historical(self):
        output, error = await self.source()
        self.assertIsNone(error)
        self.assertEqual(output.content.authored_by, "note_service")
        self.assertEqual(output.content.version_id, "saved-v1")
        self.assertTrue(output.content.is_current)
        self.assertTrue(output.speech_ready)
        self.saved.is_current = False
        later, error = await self.source()
        self.assertIsNone(error)
        self.assertFalse(later.content.is_current)
        self.assertIn("since been updated", later.content.message)
        self.assertNotEqual(output.result_digest, later.result_digest)

    async def test_wrong_result_binding_or_missing_lineage_is_unavailable(self):
        for key, value in (("task_id", str(uuid4())), ("task_revision", 2),
                           ("version_id", "current-v2"), ("effect_committed", False),
                           ("authorship", {"kind": "expert"})):
            with self.subTest(key=key):
                data = self.result.model_dump(mode="json") | {key: value}
                output, error = await self.source(data)
                self.assertIsNone(output)
                self.assertEqual(error, "note_result_unavailable")
        self.assertIsNone((await self.source(linked=2))[0])

    async def test_generic_task_reader_does_not_relabel_runtime_status_as_expert(self):
        output = inspect_expert_output(self.result.model_dump(mode="json"), task=self.task)
        self.assertEqual(output.kind, "runtime")
        self.assertFalse(output.expert_available)
        self.assertFalse(output.speech_ready)

class NoteRunnerSettlementTests(unittest.IsolatedAsyncioTestCase):
    def setup_runner(self, *, recovering=False, handled=True):
        task = NS(backend_kind="frontier", budget=NS(active_seconds=5, lease_seconds=30), active_seconds=0,
                  pending_status=None)
        claim = NS(task=task, recovering=recovering, provider_state="live")
        outcome = WorkerOutcome(TaskStatus.FAILED, {})
        finalizer = AsyncMock(return_value=handled)
        provider = NS(prepare=NS(finish_outcome=finalizer), execute=AsyncMock(return_value=outcome),
                      reconcile=AsyncMock(return_value=Reconciliation("terminal", outcome)))
        repository = NS(limits=NS(shutdown_seconds=1), finish=AsyncMock(), abandon=AsyncMock(),
                        resolve_recovery=AsyncMock())
        runner = TaskRunner(repository, {"frontier": provider})
        runner.context = Mock(return_value=NS(claim=claim))
        return runner, repository, provider, finalizer, claim

    async def test_normal_and_recovery_use_same_owner_finalizer(self):
        for recovering in (False, True):
            with self.subTest(recovering=recovering):
                runner, repo, _, finalizer, claim = self.setup_runner(recovering=recovering)
                await runner._execute(claim)
                self.assertEqual(finalizer.await_count, 1)
                repo.finish.assert_not_awaited()
                repo.resolve_recovery.assert_not_awaited()
                repo.abandon.assert_not_awaited()

    async def test_ordinary_outcome_retains_existing_finish(self):
        runner, repo, _, finalizer, claim = self.setup_runner(handled=False)
        await runner._execute(claim)
        self.assertEqual(repo.finish.await_count, 1)

    async def test_exception_waits_for_existing_executor_recovery(self):
        runner, repo, provider, finalizer, claim = self.setup_runner()
        provider.execute.side_effect = RuntimeError("cleanup may be uncertain")
        await runner._execute(claim)
        finalizer.assert_not_awaited()
        self.assertEqual(repo.abandon.await_count, 1)
        repo.finish.assert_not_awaited()

    async def test_live_or_unknown_recovery_never_finalizes_local_effect(self):
        for state in ("running", "unknown"):
            with self.subTest(state=state):
                runner, repo, provider, finalizer, claim = self.setup_runner(recovering=True)
                provider.reconcile.return_value = Reconciliation(state)
                await runner._execute(claim)
                finalizer.assert_not_awaited()
                self.assertEqual(repo.resolve_recovery.await_count, 1)


if __name__ == "__main__":
    unittest.main()
