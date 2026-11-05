"""Canonical saved-file targets in the bounded voice snapshot, without I/O."""
from datetime import UTC, datetime
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

from app.knowledge.models import SourceError
from app.live_state.task_context import RECEIPT_FIELDS, _read, _text
from app.tasks.artifacts import result_artifacts


class ArtifactTargetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.owner, self.action, self.file_id = "fixture-owner", uuid4(), uuid4()
        self.grant = uuid4()
        self.task = NS(owner_id=self.owner, id=uuid4(), knowledge_area_id=uuid4())
        self.tasks = NS(pool=object())
        self.source = NS(source_id=uuid4(), payload_id=uuid4(), document_id="source_exact")
        evidence = dict(source_id=str(self.source.source_id), revision=1, generation=1,
                        content_digest="a" * 64, is_current=True)
        self.source.original_evidence = lambda: evidence
        self.file = NS(filename="edited.mp4", content_type="video/mp4", size_bytes=123,
                       metadata={"task_artifact": {"task_id": str(self.task.id), "action_id": str(self.action)}})
        self.receipt = {"tool": "publish_generated_file", "result": {"status": "success",
                        "file": {"id": str(self.file_id), "memory_source": evidence}}}
        self.connection = NS(fetch=AsyncMock(return_value=[{"id": self.action, "grant_id": self.grant, "content": self.receipt}]),
                             fetchrow=AsyncMock(return_value={"fixture": True}))
        self.output = NS(expert_available=True, content=NS(completed_operation_refs=["op_" + self.grant.hex]))
        self.reader = AsyncMock(return_value=(self.source, None))
        for target, value in (("app.tasks.artifacts._file_from_record", Mock(return_value=self.file)),
                              ("app.tasks.artifacts.read_file_original", self.reader)):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    async def test_existing_inbox_shape_is_unchanged(self):
        self.assertEqual(await result_artifacts(self.connection, self.tasks, self.task, self.output),
                         [{"file_id": str(self.file_id), "filename": "edited.mp4",
                           "content_type": "video/mp4", "size_bytes": 123}])

    async def test_internal_target_uses_exact_current_source_and_accepted_operation(self):
        result = await result_artifacts(self.connection, self.tasks, self.task, self.output, include_targets=True)
        self.assertEqual(result[0]["version_id"], "source_exact")
        self.assertEqual(result[0]["operation_ref"], "op_" + self.grant.hex)
        self.assertNotEqual(result[0]["operation_ref"], "op_" + self.action.hex)
        self.assertEqual(result[0]["source_payload_id"], str(self.source.payload_id))
        self.assertEqual(self.reader.await_count, 1)

    async def test_withdrawn_or_changed_source_cannot_supply_a_target(self):
        self.reader.side_effect = SourceError("source_unavailable", "Removed")
        self.assertEqual(await result_artifacts(self.connection, self.tasks, self.task, self.output,
                                               include_targets=True), [])
        self.reader.side_effect = None
        self.receipt["result"]["file"]["memory_source"] = {
            **self.source.original_evidence(), "content_digest": "b" * 64}
        self.assertEqual(await result_artifacts(self.connection, self.tasks, self.task, self.output,
                                               include_targets=True), [])

    async def test_wrong_action_lineage_is_not_a_saved_output(self):
        self.file.metadata["task_artifact"]["action_id"] = str(uuid4())
        self.assertEqual(await result_artifacts(self.connection, self.tasks, self.task, self.output,
                                               include_targets=True), [])
        self.reader.assert_not_awaited()


class TaskTargetProjectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.scope = NS(owner_id="fixture-owner", session_id=uuid4(), area_id=uuid4())
        self.task = NS(owner_id=self.scope.owner_id, session_id=self.scope.session_id,
            knowledge_area_id=self.scope.area_id, id=uuid4(), request_event_id=uuid4(), request_revision=1,
            description_payload_id=uuid4(), result_payload_id=uuid4(), updated_at=self.now, revision=1,
            title="Edit my recording", status=NS(value="succeeded"), current_attempt_id=None, fence=1,
            pending_status=None, model="fixture")
        self.task.model_dump = Mock(return_value={"id": str(self.task.id), "status": "succeeded", "revision": 1})
        self.request = NS(authorizes_request=True, owner_id=self.scope.owner_id, session_id=self.scope.session_id,
                          revision=1, payload_id=uuid4())
        self.detail = {name: None for name in RECEIPT_FIELDS}
        self.detail.update(last_event_sequence=7, result_recorded_at=self.now, manifest_payload_id=uuid4(),
                           result={"answer": "The edited video is saved.", "completed_operation_refs": ["op_fixture"]})
        self.binder = NS(scope=self.scope, clock=lambda: self.now,
            tasks=NS(live_context_in_transaction=AsyncMock(return_value=(self.task, self.detail))),
            context=NS(_record=AsyncMock(return_value=self.request)), _lock_roots=AsyncMock())
        self.target = {"file_id": str(uuid4()), "version_id": "source_exact", "operation_ref": "op_" + uuid4().hex,
                       "content_type": "video/mp4", "source_payload_id": str(uuid4())}
        self.artifacts = AsyncMock(return_value=[self.target])
        for target, value in (("app.tasks.artifacts.result_artifacts", self.artifacts),
                              ("app.tasks.hermes_result.inspect_expert_output", Mock(return_value=object()))):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)

    async def test_exact_target_and_its_payload_dependency_reach_the_snapshot(self):
        snapshot = await _read(self.binder, object(), self.task.id)
        self.assertIn(self.target["source_payload_id"], snapshot.state["roots"])
        self.assertEqual(snapshot.state["artifact_targets"][0]["version_id"], "source_exact")
        text = _text(snapshot, self.now)
        self.assertIn(self.target["file_id"], text)
        self.assertIn(self.target["operation_ref"], text)
        self.assertNotIn(self.target["source_payload_id"], text)
        self.assertIn("not permission to open", text)

    async def test_withdrawal_changes_snapshot_and_removes_stale_target(self):
        before = await _read(self.binder, object(), self.task.id)
        self.artifacts.return_value = []
        after = await _read(self.binder, object(), self.task.id)
        self.assertNotEqual(before.state, after.state)
        self.assertNotIn("artifact_targets", after.state)
        self.assertNotIn(self.target["file_id"], _text(after, self.now))

    async def test_exact_targets_survive_omission_of_a_long_prose_answer(self):
        self.detail["result"]["answer"] = "x" * 4000
        snapshot = await _read(self.binder, object(), self.task.id)
        text = _text(snapshot, self.now)
        self.assertLessEqual(len(text), 2000)
        self.assertIn(self.target["file_id"], text)
        self.assertIn("omitted", text)

    async def test_target_count_is_bounded_with_explicit_omission(self):
        self.artifacts.return_value = [dict(self.target, file_id=str(uuid4())) for _ in range(3)]
        snapshot = await _read(self.binder, object(), self.task.id)
        self.assertEqual(len(snapshot.state["artifact_targets"]), 2)
        self.assertEqual(snapshot.state["artifact_omitted_count"], 1)
        self.assertIn("Additional saved files are omitted", _text(snapshot, self.now))

    async def test_ordinary_answer_does_not_read_artifact_records(self):
        self.detail["result"]["completed_operation_refs"] = []
        snapshot = await _read(self.binder, object(), self.task.id)
        self.artifacts.assert_not_awaited()
        self.assertNotIn("artifact_targets", snapshot.state)
        self.assertIn("The edited video is saved", _text(snapshot, self.now))


if __name__ == "__main__":
    unittest.main()
