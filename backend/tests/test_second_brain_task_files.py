"""Task-granted exact file adoption and inbox links on the real isolated ledger."""
import asyncio
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from uuid import UUID, uuid4

from app.attention.repository import AttentionRepository
from app.files.tools import FileToolModule
from app.knowledge.files import SourceFileRepository
from app.knowledge.models import SourceError
from app.runtime.feature_config import TaskLimits
from app.tasks.context_repository import ContextRepository
from app.tasks.hermes_result import HermesTaskResult
from app.tasks.memory_evidence import TaskMemoryEvidence
from app.tasks.models import TaskCreate, TaskError, TaskMutation
from app.tasks.repository import TaskRepository
from app.tasks.runner import TaskRunner
from app.tasks.tool_bridge import TaskToolBridge
from app.tools.invocation import bind_tool_invocation
from evals.second_brain.file_storage_fixture import MemoryStorage
from evals.second_brain.source_fixture import source_fixture


@unittest.skipUnless(os.getenv("JARVIS_SECOND_BRAIN_TEST_DB_URL") and
                     os.getenv("JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"),
                     "Explicit manifested database required")
class TaskFileDatabaseTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.f = await self.enterAsyncContext(source_fixture(
            os.environ["JARVIS_SECOND_BRAIN_TEST_DB_URL"],
            Path(os.environ["JARVIS_SECOND_BRAIN_FIXTURE_MANIFEST_DIR"]), test_name=self.id()))
        self.tasks = TaskRepository(self.f.pool, TaskLimits(lease_seconds=300))
        self.storage = MemoryStorage()
        self.parameters = {"filename": "  progress.txt  ", "content": "\nCalibration remains unfinished.\n"}
        async with self.f.pool.acquire() as connection:
            self.session = await connection.fetchval(
                "insert into jarvis_sessions(owner_id) values($1) returning id", self.f.owner)
            payload = await self.tasks.payload(connection, self.f.owner, {"text": "Save this exact synthetic text."})
            self.event = await connection.fetchval("""insert into jarvis_context_events
                (owner_id,session_id,source_namespace,source_event_id,kind,finality,trust,payload_id)
                values($1,$2,'fixture-task-file',$3,'user_request','final','user_input',$4) returning id""",
                self.f.owner, self.session, str(uuid4()), payload)
            scope = await self.tasks.payload(connection, self.f.owner, self.parameters)
            self.grant = await connection.fetchval("""insert into jarvis_operation_grants
                (owner_id,capability,operation_key,request_event_id,request_revision,scope_revision,scope_payload_id,expires_at)
                values($1,'files.create_text',$2,$3,1,1,$4,clock_timestamp()+interval '1 hour') returning id""",
                self.f.owner, str(uuid4()), self.event, scope)
        self.task = await self.tasks.create(self.f.owner, TaskCreate(request_id=uuid4(),
            title="Synthetic retained report", requested_outcome="Save the authorized text once.",
            session_id=self.session,
            knowledge_area_id=None if self._testMethodName.endswith("default_area") else self.f.area,
            grant_ids=(self.grant,)), model="fixture-no-model")
        self.claim = await self.tasks.claim_next("task-file-fixture", ("frontier",))
        self.ctx = TaskRunner(self.tasks, {}).context(self.claim)
        await ContextRepository(self.tasks).store_manifest(self.claim, watermark=0, records=[], content={"fixture": True})
        self.evidence = TaskMemoryEvidence(self.ctx, self.f.sources, area_id=self.f.area)
        self.files = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner,
                                          area_id=self.f.area, task_evidence=self.evidence)
        self.foreground = SourceFileRepository(self.f.sources, self.storage, owner_id=self.f.owner, area_id=self.f.area)
        self.bridge = await TaskToolBridge.from_modules(self.ctx, (FileToolModule(self.files, self.storage),),
            read_tools=("list_files", "get_file_metadata"), capability_tools={"files.create_text": "create_text_file"},
            memory_evidence=self.evidence, receipt_revalidator=self.files.revalidate_receipt)
        self.reference = self.bridge.authorized_operations()[0]["operation_ref"]

    async def call(self, identity="first-call"):
        return await self.bridge.invoke("create_text_file", {"operation_ref": self.reference}, provider_call_id=identity)

    async def action(self):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchrow("select * from jarvis_task_actions where owner_id=$1", self.f.owner)

    async def file_count(self):
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval("select count(*) from jarvis_files where owner_id=$1", self.f.owner)

    async def finish(self, *, references=None):
        outcome = await HermesTaskResult(status="succeeded", answer="The authorized text is saved.",
            speech=[{"text": "The authorized text is saved."}], evidence=[], uncertainties=[],
            completed_operation_refs=[self.reference] if references is None else references).verified(self.ctx, self.bridge)
        self.assertTrue(await self.tasks.finish(self.claim, outcome))
        async with self.f.pool.acquire() as connection:
            return await connection.fetchval(
                "select id from jarvis_attention_intents where owner_id=$1 and task_id=$2", self.f.owner, self.task.id)

    async def test_exact_defaulted_grant_receipt_lineage_replay_and_inbox_link(self):
        first, replay = await self.call(), await self.call("new-native-id")
        self.assertEqual(first["status"], "success")
        self.assertEqual(first["file"], replay["file"])
        self.assertEqual(first["file"]["filename"], "progress.txt")
        self.assertEqual(first["file"]["description"], "Jarvis-created text file.")
        self.assertNotIn("storage_path", first["file"])
        self.assertIn("task_evidence", first["file"])
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(await self.file_count(), 1)
        file_id = UUID(first["file"]["id"])
        file = await self.foreground.get_file(owner_id=self.f.owner, file_id=file_id)
        self.assertEqual(file.metadata["task_artifact"]["attempt_id"], str(self.claim.attempt_id))
        self.assertEqual(await self.foreground.read_content(owner_id=self.f.owner, file_id=file_id),
                         self.parameters["content"].encode())
        intent = await self.finish()
        detail = await AttentionRepository(self.tasks).detail(self.f.owner, intent)
        self.assertEqual(detail["artifacts"], [{"file_id": str(file_id), "filename": "progress.txt",
            "content_type": "text/plain; charset=utf-8", "size_bytes": len(self.parameters["content"].encode())}])
        self.assertEqual((await self.action())["status"], "accepted")

    async def test_lost_handler_ack_after_adoption_replays_without_second_upload(self):
        handler = self.bridge._handler
        async def lose_ack(*args, **kwargs):
            await handler(*args, **kwargs)
            raise RuntimeError("Synthetic lost post-adoption acknowledgement")
        with patch.object(self.bridge, "_handler", lose_ack), self.assertRaises(RuntimeError):
            await self.call()
        self.assertEqual((await self.action())["status"], "accepted")
        result = await self.call("retry-new-id")
        self.assertTrue(result["task_operation"]["replayed"])
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(await self.file_count(), 1)

    async def test_unknown_upload_is_not_blindly_repeated_or_claimed_saved(self):
        self.storage.mode = "fail_after"
        with self.assertRaises(RuntimeError):
            await self.call()
        result = await self.call("retry-new-id")
        self.assertEqual(result["status"], "outcome_unknown")
        self.assertNotIn("file", result)
        self.assertEqual(self.storage.uploads, 1)
        self.assertEqual(await self.file_count(), 0)
        with self.assertRaises(TaskError):
            await self.bridge.verified_operation(self.reference)
        async with self.f.pool.acquire() as connection:
            operation = await connection.fetchval("select id from jarvis_file_publications where owner_id=$1", self.f.owner)
        with self.assertRaises(TaskError) as caught:
            await self.foreground.publications.reconcile(owner_id=self.f.owner, operation_id=operation)
        self.assertEqual(caught.exception.code, "file_task_adoption_required")
        self.assertEqual(await self.file_count(), 0)

    async def test_rejected_receipt_rolls_back_original_and_adoption(self):
        original = self.ctx.actions.record_receipt_in_transaction
        async def reject(*args, **kwargs):
            if args[-2] == "accepted":
                raise RuntimeError("Synthetic receipt failure")
            return await original(*args, **kwargs)
        with patch.object(self.ctx.actions, "record_receipt_in_transaction", reject), self.assertRaises(RuntimeError):
            await self.call()
        self.assertEqual(await self.file_count(), 0)
        self.assertEqual((await self.action())["status"], "outcome_unknown")
        self.assertEqual(self.storage.uploads, 1)

    async def test_deleted_file_revokes_result_and_does_not_recreate_on_receipt(self):
        result = await self.call()
        intent = await self.finish()
        await self.foreground.delete_source_file(owner_id=self.f.owner, file_id=UUID(result["file"]["id"]))
        detail = await AttentionRepository(self.tasks).detail(self.f.owner, intent)
        self.assertFalse(detail["source_available"])
        self.assertEqual(detail["artifacts"], [])
        self.assertIsNone(detail["expert_result"])
        self.assertEqual(self.storage.objects, {})

    async def test_update_during_upload_fences_adoption_and_old_links(self):
        self.storage.mode = "pause_after"
        pending = asyncio.create_task(self.call())
        try:
            await asyncio.wait_for(self.storage.entered.wait(), 10)
            await self.tasks.mutate(self.f.owner, self.task.id, "update", TaskMutation(
                request_id=uuid4(), expected_revision=1, message="Do not publish the previous text."))
        finally:
            self.storage.release.set()
        with self.assertRaises(TaskError):
            await pending
        self.assertEqual(await self.file_count(), 0)
        self.assertEqual(self.storage.uploads, 1)

    async def test_exact_scope_and_worker_destructive_tools_are_not_broadened(self):
        self.assertNotIn("delete_file", [item["name"] for item in self.bridge.catalog()])
        result = await self.bridge.invoke("create_text_file", {"operation_ref": self.reference,
            "content": "Invented replacement"}, provider_call_id="bad")
        self.assertEqual(result["status"], "failed")
        action = await self.ctx.actions.prepare(self.claim, "tool-grant:" + self.grant.hex,
            self.grant, "files.create_text", self.parameters)
        await self.ctx.actions.authorize(self.claim, action["id"])
        await self.ctx.actions.begin_submission(self.claim, action["id"])
        with bind_tool_invocation(provider_call_id="manual", request_event_id=self.event, request_revision=1,
                task=self.bridge._scope(action_id=action["id"], operation_key="tool-grant:" + self.grant.hex)):
            with self.assertRaises(TaskError):
                await self.files.create_text_file(owner_id=self.f.owner, session_id=self.session,
                    filename="progress.txt", content="Different", description="Jarvis-created text file.")
            with self.assertRaises(SourceError):
                await self.foreground.create_text_file(owner_id=self.f.owner, session_id=self.session,
                    filename="progress.txt", content="Different", description="Jarvis-created text file.")
            with self.assertRaises(TaskError):
                await self.files.delete_source_file(owner_id=self.f.owner, file_id=uuid4())
        self.assertEqual(self.storage.uploads, 0)

    async def test_answer_cannot_link_an_unclaimed_file_operation(self):
        await self.call()
        intent = await self.finish(references=[])
        self.assertEqual((await AttentionRepository(self.tasks).detail(self.f.owner, intent))["artifacts"], [])

    async def test_file_read_binds_source_without_granting_writes(self):
        file = await self.foreground.create_text_file(owner_id=self.f.owner, session_id=self.session,
            filename="existing.txt", content="Synthetic existing source", description="Fixture")
        result = await self.bridge.invoke("get_file_metadata", {"file_id": str(file.id)}, provider_call_id="read")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["file"]["task_evidence"]["source_id"], file.memory_source["source_id"])
        self.assertEqual(self.storage.uploads, 1)

    async def test_inbox_artifact_uses_recorded_default_area(self):
        result = await self.call()
        intent = await self.finish()
        detail = await AttentionRepository(self.tasks).detail(self.f.owner, intent)
        self.assertEqual([item["file_id"] for item in detail["artifacts"]], [result["file"]["id"]])
